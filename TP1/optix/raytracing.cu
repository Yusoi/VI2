#include <optix_device.h>
#include "LaunchParams.h"
#include <vec_math.h>
#include "random.h"

/* Compile with:
nvcc.exe -O3 -use_fast_math -arch=compute_30 -code=sm_30 -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.0.0\include" -I "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\include" -I "." -m 64 -ptx -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64" raytracing.cu -o raytracing.ptx
*/

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

// for this simple example, we have a single ray type
enum { PHONG_RAY_TYPE = 0, SHADOW_RAY_TYPE,  RAY_TYPE_COUNT };

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
    const float3 surfPos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // Face forward + Normalization
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);
    float3 Ns = normalize(n);

    float intensity = max(dot(-lightDir, Ns),0.5f);

    // Set payload
    float lightVisibility = 1.0f;
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );

    //Trace shadow ray
    optixTrace(optixLaunchParams.traversable,
               surfPos,
               -lightDir,
               0.001f,      // tmin
               1e20f,  // tmax
               0.0f,       // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               SHADOW_RAY_TYPE,            // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SHADOW_RAY_TYPE,            // missSBTIndex 
               u0, u1 );

    // Lambert Diffuse
    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {
      const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
      float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      prd = make_float3(fromTexture) * min(intensity * lightVisibility, 1.0);
    }
    
    else{
        // Final shading: ambient, directional ambient and shadowing
        prd = sbtData.color * min(intensity * lightVisibility, 1.0);
    }
}

//any hit radiance
extern "C" __global__ void __anyhit__radiance() {

}

//miss radiance
extern "C" __global__ void __miss__radiance() {
    float3 &prd = *(float3*)getPRD<float3>();
    // white background color
    prd = make_float3(0.0f, 0.8f, 0.0f);    
}

//closest hit shadow
extern "C" __global__ void __closesthit__shadow() {
    float &prd = *(float*)getPRD<float>();
    prd = 0.5f;
}

//any hit shadow
extern "C" __global__ void __anyhit__shadow() {
    //Not to fill, direct lighting
}

//miss shadow
extern "C" __global__ void __miss__shadow() {
    // we didn't hit anything, so the light is visible
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

//----------
//closest hit radiance para grades
extern "C" __global__ void __closesthit__radiance_grade() {
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
    const float3 surfPos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // Face forward + Normalization
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);
    float3 Ns = normalize(n);

    float intensity = max(dot(-lightDir, Ns),0.5f);

    const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
    float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
    // Lambert Diffuse
    if (sbtData.hasTexture && sbtData.vertexD.texCoord0 && fromTexture.x != 1.0f && fromTexture.y != 1.0f && fromTexture.z != 1.0f) {

        // Set payload
        float lightVisibility = 1.0f;
        uint32_t u0, u1;
        packPointer( &lightVisibility, u0, u1 );

        //Trace shadow ray
        optixTrace(optixLaunchParams.traversable,
            surfPos,
            -lightDir,
            0.001f,      // tmin
            1e20f,  // tmax
            0.0f,       // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SHADOW_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            SHADOW_RAY_TYPE,            // missSBTIndex 
            u0, u1 );

        prd = make_float3(fromTexture) * min(intensity * lightVisibility, 1.0);

    } else {

         // ray payload
        float3 glass_color_PRD = make_float3(1.f);
        uint32_t u0, u1;
        packPointer( &glass_color_PRD, u0, u1 ); 

        optixTrace(optixLaunchParams.traversable,
            surfPos,
            optixGetWorldRayDirection(),
            0.001f,      // tmin
            1e20f,  // tmax
            0.0f,       // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            PHONG_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            PHONG_RAY_TYPE,            // missSBTIndex 
            u0, u1);
    
        prd = glass_color_PRD;

    }
}

//any hit radiance para grades
extern "C" __global__ void __anyhit__radiance_grade() {
    //Not to fill, direct lighting
}

//miss radiance para grades
extern "C" __global__ void __miss__radiance_grade() {
    
}

//closest hit shadow para grades
extern "C" __global__ void __closesthit__shadow_grade() {
    float &prd = *(float*)getPRD<float>();
    prd = 0.5f;
}

//any hit shadow para grades
extern "C" __global__ void __anyhit__shadow_grade() {
    //Not to fill, direct lighting
}

//miss shadow para grades
extern "C" __global__ void __miss__shadow_grade() {
    //TODO
    // we didn't hit anything, so the light is visible
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

/**
*
*   VIDRO
*
*/

//closest hit radiance para vidros
extern "C" __global__ void __closesthit__radiance_vidro() {
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

    // intersection position
    const float3 surfPos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // ray payload
    float3 glass_color_PRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &glass_color_PRD, u0, u1 );  

    optixTrace(optixLaunchParams.traversable,
        surfPos,
        optixGetWorldRayDirection(),
        0.001f,      // tmin
        1e20f,  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        PHONG_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        PHONG_RAY_TYPE,            // missSBTIndex 
        u0, u1);

    prd = glass_color_PRD;
}

//any hit radiance para vidros
extern "C" __global__ void __anyhit__radiance_vidro() {
    //Not to fill, direct lighting
}

//miss radiance para vidros
extern "C" __global__ void __miss__radiance_vidro() {
    float3 &prd = *(float3*)getPRD<float3>();
    // white background color
    prd = make_float3(0.0f, 0.8f, 0.0f); 
}

//closest hit shadow para vidros
extern "C" __global__ void __closesthit__shadow_vidro() {
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

//any hit shadow para vidros
extern "C" __global__ void __anyhit__shadow_vidro() {
    //Not to fill, direct lighting
}

//miss shadow para vidros
extern "C" __global__ void __miss__shadow_vidro() {
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

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
            const float4 &ld = optixLaunchParams.global->lightDir;
            printf("LightPos: %f, %f %f %f\n", ld.x,ld.y,ld.z,ld.w);
            printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
            printf("Rays per pixel squared: %d \n", optixLaunchParams.frame.raysPerPixel);
            printf("===========================================\n");
        }
    
    
        // ray payload
        float3 pixelColorPRD = make_float3(1.f);
        uint32_t u0, u1;
        packPointer( &pixelColorPRD, u0, u1 );  
        float red = 0.0f, blue = 0.0f, green = 0.0f;
        

        const float2 screen(make_float2(ix+.5f,iy+.5f)
                        / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);

        // note: nau already takes into account the field of view and ratio when computing 
        // camera horizontal and vertival
        float3 rayDir = normalize(camera.direction
                                + (screen.x ) * camera.horizontal
                                + (screen.y ) * camera.vertical);
            
        // trace primary ray
        optixTrace(optixLaunchParams.traversable,
                camera.position,
                rayDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,//,OPTIX_RAY_FLAG_DISABLE_ANYHIT
                PHONG_RAY_TYPE,             // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                PHONG_RAY_TYPE,             // missSBTIndex 
                u0, u1 );

        red += pixelColorPRD.x;
        green += pixelColorPRD.y;
        blue += pixelColorPRD.z;
    
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

