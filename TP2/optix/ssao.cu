#include <optix_device.h>
#include "LaunchParams.h"
#include <vec_math.h>
#include "random.h"

/* Compile with:
nvcc.exe -O3 -use_fast_math -arch=compute_30 -code=sm_30 -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.0.0\include" -I "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\include" -I "." -m 64 -ptx -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64" ssao.cu -o ssao.ptx
*/

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

// for this simple example, we have a single ray type
enum { PHONG_RAY_TYPE = 0, SHADOW_RAY_TYPE, SAMPLE_RAY_TYPE, RAY_TYPE_COUNT };

/**
*
* Default Material
*
*/

//closest hit radiance
extern "C" __global__ void __closesthit__radiance() {
}

//any hit radiance
extern "C" __global__ void __anyhit__radiance() {
}

//miss radiance
extern "C" __global__ void __miss__radiance() {   
}

//closest hit shadow
extern "C" __global__ void __closesthit__shadow() {
}

//any hit shadow
extern "C" __global__ void __anyhit__shadow() {
}

//miss shadow
extern "C" __global__ void __miss__shadow() {
}

//closest hit sample
extern "C" __global__ void __closesthit__sample() {
}

//any hit sample
extern "C" __global__ void __anyhit__sample() {
}

//miss sample
extern "C" __global__ void __miss__sample() {
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
        
        float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);

        const float2 screen(make_float2(ix+.5f,iy+.5f)
                        / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);

        // note: nau already takes into account the field of view and ratio when computing 
        // camera horizontal and vertical
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

