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
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);
    float3 Ns = normalize(n);

    float intensity = max(dot(-lightDir, Ns),0.5f);

    // Set payload
    float lightVisibility = 0.0f;
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );

    //Trace shadow ray
    optixTrace(optixLaunchParams.traversable,
               pos,
               -lightDir,
               0.001f,      // tmin
               1e10f,  // tmax
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
        if (fromTexture.x != 1.0f && fromTexture.y != 1.0f && fromTexture.z != 1.0f) {
            prd = make_float3(fromTexture) * min(intensity * lightVisibility, 1.0);
        } else {
            prd = make_float3(1.0f,1.0f,1.0f);
        }
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
    // we didn't hit anything, so the light is visible
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

/**
*
* Grades
*
*/

//closest hit radiance para grades
extern "C" __global__ void __closesthit__radiance_grade() {
    float3 &prd = *(float3*)getPRD<float3>();

    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // Intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    // Light Direction
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);

    // gather basic info
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
    float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);

    // Lambert Diffuse
    if (fromTexture.x != 1.0f && fromTexture.y != 1.0f && fromTexture.z != 1.0f) {
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

        // Face forward + Normalization
        float3 Ns = normalize(n);

        float intensity = max(dot(-lightDir, Ns),0.5f);

        // Set payload
        float lightVisibility = 0.0f;
        uint32_t u0, u1;
        packPointer( &lightVisibility, u0, u1 );

        //Trace shadow ray
        optixTrace(optixLaunchParams.traversable,
                   pos,
                   -lightDir,
                   0.001f,      // tmin
                   1e10f,  // tmax
                   0.0f,       // rayTime
                   OptixVisibilityMask( 255 ),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   SHADOW_RAY_TYPE,            // SBT offset
                   RAY_TYPE_COUNT,               // SBT stride
                   SHADOW_RAY_TYPE,            // missSBTIndex 
                   u0, u1 );

        const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        if (fromTexture.x != 1.0f && fromTexture.y != 1.0f && fromTexture.z != 1.0f) {
            prd = make_float3(fromTexture) * min(intensity * lightVisibility, 1.0);
        } else {
            prd = make_float3(1.0f,1.0f,1.0f);
        }

    } else {
        float3 transparent = make_float3(1.0f);
        uint32_t u0, u1;
        packPointer( &transparent, u0, u1 );

        //Trace shadow ray
        optixTrace(optixLaunchParams.traversable,
                   pos,
                   optixGetWorldRayDirection(),
                   0.001f,      // tmin
                   1e20f,  // tmax
                   0.0f,       // rayTime
                   OptixVisibilityMask( 255 ),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   PHONG_RAY_TYPE,            // SBT offset
                   RAY_TYPE_COUNT,               // SBT stride
                   PHONG_RAY_TYPE,            // missSBTIndex 
                   u0, u1 ); 

        prd = transparent;
    }

    
}

//any hit radiance para grades
extern "C" __global__ void __anyhit__radiance_grade() {
}

//miss radiance para grades
extern "C" __global__ void __miss__radiance_grade() {
}

//closest hit shadow para grades
extern "C" __global__ void __closesthit__shadow_grade() {
    float &prd = *(float*)getPRD<float>();

    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // Intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    // Light Direction
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);

    // gather basic info
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
    float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);

    // Lambert Diffuse
    if (fromTexture.x != 1.0f && fromTexture.y != 1.0f && fromTexture.z != 1.0f) {
        prd = 0.5f;
    } else {
        float transparent = 1.0f;
        uint32_t u0, u1;
        packPointer( &transparent, u0, u1 );

        //Trace shadow ray
        optixTrace(optixLaunchParams.traversable,
                   pos,
                   -lightDir,
                   0.001f,      // tmin
                   1e10f,  // tmax
                   0.0f,       // rayTime
                   OptixVisibilityMask( 255 ),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   SHADOW_RAY_TYPE,            // SBT offset
                   RAY_TYPE_COUNT,               // SBT stride
                   SHADOW_RAY_TYPE,            // missSBTIndex 
                   u0, u1 ); 

        prd = transparent;
    }
}

//any hit shadow para grades
extern "C" __global__ void __anyhit__shadow_grade() {
}

//miss shadow para grades
extern "C" __global__ void __miss__shadow_grade() {
}

/**
*
* Azulejo
*
*/

//closest hit radiance para azulejos
extern "C" __global__ void __closesthit__radiance_azulejo() {
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
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);
    float3 Ns = normalize(n);

    float intensity = max(dot(-lightDir, Ns),0.5f);

    const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
    float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
    
    float3 rayDir = reflect(optixGetWorldRayDirection(), Ns);

    // Phong
    // ray payload
    float3 lightVisibility = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 ); 

    optixTrace(optixLaunchParams.traversable,
        pos,
        rayDir,
        0.04f,      // tmin
        1e10f,  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE,
        PHONG_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        PHONG_RAY_TYPE,            // missSBTIndex 
        u0, u1);
    
        prd = make_float3(fromTexture) * lightVisibility;
}

//any hit radiance para azulejos
extern "C" __global__ void __anyhit__radiance_azulejo() {
}

//miss radiance para azulejos
extern "C" __global__ void __miss__radiance_azulejo() {
}

//closest hit shadow para azulejos
extern "C" __global__ void __closesthit__shadow_azulejo() {
    float &prd = *(float*)getPRD<float>();
    prd = 0.5f;
}

//any hit shadow para azulejos
extern "C" __global__ void __anyhit__shadow_azulejo() {
}

//miss shadow para azulejos
extern "C" __global__ void __miss__shadow_azulejo() {
}

/**
*
*   Vidro
*
*/

//closest hit radiance para vidros
extern "C" __global__ void __closesthit__radiance_vidro() {
    float3 &prd = *(float3*)getPRD<float3>();

    // Intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    // Light Direction
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);

    float3 transparent = make_float3(1.0f);
    uint32_t u0, u1;
    packPointer( &transparent, u0, u1 );

    //Trace shadow ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        optixGetWorldRayDirection(),
        0.001f,      // tmin
        1e10f,  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        PHONG_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        PHONG_RAY_TYPE,            // missSBTIndex 
        u0, u1 ); 

    prd = transparent;
}

//any hit radiance para vidros
extern "C" __global__ void __anyhit__radiance_vidro() {
}

//miss radiance para vidros
extern "C" __global__ void __miss__radiance_vidro() {
}

//closest hit shadow para vidros
extern "C" __global__ void __closesthit__shadow_vidro() {   
    float &prd = *(float*)getPRD<float>();

    // Intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    // Light Direction
    const float3 lightDir = make_float3(optixLaunchParams.global->lightDir);

    // Set payload
    float lightVisibility = 0.0f;
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );

    //Trace shadow ray
    optixTrace(optixLaunchParams.traversable,
               pos,
               -lightDir,
               0.001f,      // tmin
               1e10f,  // tmax
               0.0f,       // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               SHADOW_RAY_TYPE,            // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SHADOW_RAY_TYPE,            // missSBTIndex 
               u0, u1 ); 

    prd = lightVisibility;
}

//any hit shadow para vidros
extern "C" __global__ void __anyhit__shadow_vidro() {
}

//miss shadow para vidros
extern "C" __global__ void __miss__shadow_vidro() {
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
                1e10f,  // tmax
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

