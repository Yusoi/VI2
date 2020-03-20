#include <optix_device.h>
#include "LaunchParams.h"
#include <vec_math.h>

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

    float3 Ns;
    float3 Ng = cross(B-A,C-A);
    if(sbtData.vertexD.normal) 
        Ns = make_float3((1.f-u-v) * sbtData.vertexD.normal[index.x] + u * sbtData.vertexD.normal[index.y] + v * sbtData.vertexD.normal[index.z]);
    else 
        Ns = Ng;
    
    // intersection position
    const float3 surfPos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // Face forward + Normalizatio
    const float3 rayDir = optixGetWorldRayDirection();
    const float3 lightDir = normalize(make_float3(optixLaunchParams.global->lightDir));
    if (dot(rayDir,Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);
    
    if (dot(Ng,Ns) < 0.f)
        Ns -= 2.f*dot(Ng,Ns)*Ng;
    Ns = normalize(Ns);

    float intensity = max(dot(lightDir, Ns),0.0f);

    // Set payload
    float lightVisibility = 1.0f;
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );
    //Trace shadow ray
    optixTrace(optixLaunchParams.traversable,
               surfPos,
               lightDir,
               0.001f,      // tmin
               1.0f,  // tmax
               0.0f,       // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_NONE,
               SHADOW_RAY_TYPE,            // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SHADOW_RAY_TYPE,            // missSBTIndex 
               u0, u1 );

    // Lambert Diffuse
    float3 diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {
      const float4 tc = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x] + u * sbtData.vertexD.texCoord0[index.y] + v * sbtData.vertexD.texCoord0[index.z];
      float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      prd = make_float3(fromTexture) * min(intensity * lightVisibility + 0.0, 1.0);
    }

    
    // Final shading: ambient, directional ambient and shadowing
    const float cosDN = 0.1f + .8f*fabsf(dot(rayDir,Ns));
    prd = (.1f + (.2f + .8f*lightVisibility) * cosDN) * diffuseColor;
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
    prd = 0.0f;
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

//-----------------
// Luzes
extern "C" __global__ void __closesthit__light() {

    float3 &prd = *(float3*)getPRD<float3>();
    prd = make_float3(1.0f, 1.0f, 1.0f);
}


extern "C" __global__ void __anyhit__light() {
}


extern "C" __global__ void __miss__light() {
}


extern "C" __global__ void __closesthit__light_shadow() {
    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

// Ignore intersections
extern "C" __global__ void __anyhit__light_shadow() {
}

// miss empty for background
extern "C" __global__ void __miss__light_shadow() {
}

//----------
//closest hit radiance para grades
extern "C" __global__ void __closesthit__radiance_grade() {

}

//any hit radiance para grades
extern "C" __global__ void __anyhit__radiance_grade() {

}

//miss radiance para grades
extern "C" __global__ void __miss__radiance_grade() {
    
}

//closest hit shadow para grades
extern "C" __global__ void __closesthit__shadow_grade() {

}

//any hit shadow para grades
extern "C" __global__ void __anyhit__shadow_grade() {

}

//miss shadow para grades
extern "C" __global__ void __miss__shadow_grade() {

}

//closest hit radiance para vidros
extern "C" __global__ void __closesthit__radiance_vidro() {

}

//any hit radiance para vidros
extern "C" __global__ void __anyhit__radiance_vidro() {

}

//miss radiance para vidros
extern "C" __global__ void __miss__radiance_vidro() {
    
}

//closest hit shadow para vidros
extern "C" __global__ void __closesthit__shadow_vidro() {

}

//any hit shadow para vidros
extern "C" __global__ void __anyhit__shadow_vidro() {

}

//miss shadow para vidros
extern "C" __global__ void __miss__shadow_vidro() {

}

//Ray Deployment
extern "C" __global__ void __raygen__renderFrame() {

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;
    // ray payload
    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );
    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    const float2 screen(make_float2(ix+.5f,iy+.5f)
                        / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
    // note: nau already takes into account the field of view when computing
    // camera horizontal and vertical
    float3 rayDir = normalize(camera.direction
                              + screen.x * camera.horizontal
                              + screen.y * camera.vertical);

    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        camera.position,
        rayDir,
        0.f, // tmin
        1e20f, // tmax
        0.0f, // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
        PHONG_RAY_TYPE, // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        PHONG_RAY_TYPE, // missSBTIndex
        u0, u1 );
    
    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*pixelColorPRD.x);
    const int g = int(255.0f*pixelColorPRD.y);
    const int b = int(255.0f*pixelColorPRD.z);
    // convert to 32-bit rgba value
    const uint32_t rgba = 0xff000000
    | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix+iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    
    if(optixLaunchParams.frame.frame == 0 && ix == 0 && iy == 0){
        // print info to console
        printf("===========================================\n");
        printf("Nau Ray-Tracing Hello World\n");
        printf("Launch size: %i x %i\n", ix, iy);
        printf("Camera Direction: %f %f %f\n",
        optixLaunchParams.camera.direction.x,
        optixLaunchParams.camera.direction.y,
        optixLaunchParams.camera.direction.z);
        printf("===========================================\n");
    }

}

