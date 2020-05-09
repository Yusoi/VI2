#include <optix_device.h>
#include "LaunchParams.h"
#include <vec_math.h>
#include "random.h"

/* Compile with:
nvcc.exe -O3 -use_fast_math -arch=compute_30 -code=sm_30 -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.0.0\include" -I "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\include" -I "." -m 64 -ptx -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64" dof.cu -o dof.ptx
*/

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

// for this simple example, we have a single ray type
enum { PHONG_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

/**
*
* Default Material
*
*/

//closest hit radiance
extern "C" __global__ void __closesthit__radiance() {
    float3 &prd = *(float3*)getPRD<float3>();

    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // gather basic info
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute triangle normal using either shading normal or gnormal as fallback:
    const float3 &A = make_float3(sbtData.vertexD.position[index.x]);
    const float3 &B = make_float3(sbtData.vertexD.position[index.y]);
    const float3 &C = make_float3(sbtData.vertexD.position[index.z]);

    float3 n;
    float3 Ng = cross(B-A,C-A);
    if(sbtData.vertexD.normal) 
        n = make_float3((1.f-u-v) * sbtData.vertexD.normal[index.x] + u * sbtData.vertexD.normal[index.y] + v * sbtData.vertexD.normal[index.z]);
    else 
        n = Ng;

    // intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // Face forward + Normalization
    const float3 lightDir = normalize(make_float3(optixLaunchParams.global->lightPos)-pos);
    float3 Ns = normalize(n);

    float intensity = max(dot(-lightDir, Ns),0.5f);

    // Set payload
    float lightVisibility = 0.0f;
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );

    optixTrace(optixLaunchParams.traversable,
        pos,
        -lightDir,
        0.00001f,           // tmin
        1.0f,               // tmax
        0.0f,               // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SHADOW_RAY_TYPE,    // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW_RAY_TYPE,    // missSBTIndex 
        u0, u1 );

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {
        const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        if (fromTexture.x != 1.0f && fromTexture.y != 1.0f && fromTexture.z != 1.0f) {
            prd = make_float3(fromTexture) * min(intensity * lightVisibility, 1.0);
        } else {
            prd = make_float3(1.0f,1.0f,1.0f);
        }
    }
}

//any hit radiance
extern "C" __global__ void __anyhit__radiance() {
}

//miss radiance
extern "C" __global__ void __miss__radiance() {   
    float3 &prd = *(float3*)getPRD<float3>();
    // white background color
    prd = make_float3(1.0f, 1.0f, 1.0f);      
}

//closest hit shadow
extern "C" __global__ void __closesthit__shadow() {
    float &prd = *(float*)getPRD<float>();
    prd = 0.5f;
}

//any hit shadow
extern "C" __global__ void __anyhit__shadow() {
}

//miss shadow
extern "C" __global__ void __miss__shadow() {
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

/**
*
*   Raygen
*
*/

//Ray Deployment
extern "C" __global__ void __raygen__renderFrame() {

        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const auto &camera = optixLaunchParams.camera; 
        
        if (optixLaunchParams.frame.frame == 0 && ix == 0 && iy == 0) {
    
            // print info to console
            printf("===========================================\n");
            printf("Nau Ray-Tracing Debug\n");
            printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
            printf("Rays per pixel: %d \n", optixLaunchParams.frame.raysPerPixel);
            printf("===========================================\n");
        }

        /*
        if((ix == 0 && iy == 0 )||(ix == 200 && iy == 200) || (ix == 600 && iy == 600)){
            printf("X: %u, Y: %u, Camera Position: %f %f %f\n",ix,iy,camera.position.x,camera.position.y,camera.position.z);
        }*/
    
    
        // ray payload
        float3 pixelColorPRD = make_float3(1.f);
        uint32_t u0, u1;
        packPointer( &pixelColorPRD, u0, u1 );  
        float red = 0.0f, blue = 0.0f, green = 0.0f;
        
        float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);

        //Lens variables
        float aperture = optixLaunchParams.global->aperture;
        float focalDistance = optixLaunchParams.global->focalDistance;
        float lensDistance = optixLaunchParams.global->lensDistance;

        //1st step: Calculate focal plane point. "Shoots a ray" from the center of the lens to the image
        //To get the focal point, di (distance from the lens to the image) needs to the the same as dp (distance from the lens to the camera)
        float dp = lensDistance;
        float3 lensCenter = camera.position + camera.direction * dp;
        //float3 focalPoint = 



        for(int i = 0; i < raysPerPixel; i++){
            for(int j = 0; j < raysPerPixel; j++){
                float2 subpixel_jitter;
                uint32_t seed = tea<4>(ix * optixGetLaunchDimensions().x + iy, i * raysPerPixel + j);
                
                subpixel_jitter = make_float2(rnd(seed)-0.5f,rnd(seed)-0.5f);

                const float2 screen(make_float2(ix + subpixel_jitter.x,iy + subpixel_jitter.y)
                                / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
        
                if ( (ix == 0 && iy == 0) || (ix == 1000 && iy == 1000)) {
                    printf("X: %u, Y: %u, Camera Position: %f %f %f\n",ix,iy,camera.position.x,camera.position.y,camera.position.z);
                }

                //TODO: Generate a random point in the lens
                float3 curLensPoint = make_float3(0.0f,0.0f,0.0f);

                // note: nau already takes into account the field of view and ratio when computing 
                // camera horizontal and vertical
                flaot3 rayDir = normalize(focalPoint - curLensPoint);
                /*float3 rayDir = normalize(camera.direction
                                        + (screen.x ) * camera.horizontal
                                        + (screen.y ) * camera.vertical);*/
                                        
                // trace primary ray
                optixTrace(optixLaunchParams.traversable,
                        lensCenter,
                        rayDir,
                        0.f,    // tmin
                        1e10f,  // tmax
                        0.0f,   // rayTime
                        OptixVisibilityMask( 255 ),
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT,//,OPTIX_RAY_FLAG_DISABLE_ANYHIT
                        PHONG_RAY_TYPE,             // SBT offset
                        RAY_TYPE_COUNT,               // SBT stride
                        PHONG_RAY_TYPE,             // missSBTIndex 
                        u0, u1 );
        
                red += pixelColorPRD.x/(raysPerPixel*raysPerPixel);
                green += pixelColorPRD.y/(raysPerPixel*raysPerPixel);
                blue += pixelColorPRD.z/(raysPerPixel*raysPerPixel);
            }
        }

        
    
        //convert float (0-1) to int (0-255)
        const int r = int(255.0f*red);
        const int g = int(255.0f*green);
        const int b = int(255.0f*blue);
        // convert to 32-bit rgba value 
        const uint32_t rgba = 0xff000000
          | (r<<0) | (g<<8) | (b<<16);
        // compute index
        const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;
        // write to output buffer
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}

