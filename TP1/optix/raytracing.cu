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
enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{ 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}

//closest hit lambert
extern "C" __global__ void __closesthit__lambert() {
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

    vec3f Ng = cross(B-A,C-A);
    if(sbtData) 
        Ns = ((1.f-u-v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z]);
    else 
        Ns = Ng;
    
    // Face forward + Normalization
    const vec3f rayDir = optixGetWorldRayDirection();
    
    if (dot(rayDir,Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);
    
    if (dot(Ng,Ns) < 0.f) Ns -= 2.f*dot(Ng,Ns)*Ng;
    Ns = normalize(Ns);

    // Lambert Diffuse
    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
      const vec2f tc = (1.f-u-v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];
      vec4f fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      diffuseColor *= (vec3f)fromTexture;
    }

    // Shadow
    const vec3f surfPos = (1.f-u-v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];
    const vec3f lightPos(-907.108f, 2205.875f, -400.0267f);
    const vec3f lightDir = lightPos - surfPos;

    // Trace Shadow Ray
    vec3f lightVisibility = 0.f;

    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );
    optixTrace(optixLaunchParams.traversable,
               surfPos + 1e-3f * Ng,
               lightDir,
               1e-3f,      // tmin
               1.f-1e-3f,  // tmax
               0.0f,       // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               SHADOW_RAY_TYPE,            // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SHADOW_RAY_TYPE,            // missSBTIndex 
               u0, u1 );

    // Final shading: ambient, directional ambient and shadowing
    const float cosDN = 0.1f + .8f*fabsf(dot(rayDir,Ns));
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = (.1f + (.2f + .8f*lightVisibility) * cosDN) * diffuseColor;
}

//any hit lambert
extern "C" __global__ void __anyhit__lambert() {

}

//miss lambert
extern "C" __global__ void __miss__lambert() {

}

//closest hit phong
extern "C" __global__ void __closesthit__phong() {

}

//any hit phong
extern "C" __global__ void __anyhit__phong() {

}

//miss phong
extern "C" __global__ void __miss__phong() {

}

//closest hit lambert para grades
extern "C" __global__ void __closesthit__lambert_grade() {

}

//any hit lambert para grades
extern "C" __global__ void __anyhit__lambert_grade() {

}

//miss lambert para grades
extern "C" __global__ void __miss__lambert_grade() {
    
}

//closest hit phong para grades
extern "C" __global__ void __closesthit__phong_grade() {

}

//any hit phong para grades
extern "C" __global__ void __anyhit__phong_grade() {

}

//miss phong para grades
extern "C" __global__ void __miss__phong_grade() {

}

//closest hit lambert para vidros
extern "C" __global__ void __closesthit__lambert_vidro() {

}

//any hit lambert para vidros
extern "C" __global__ void __anyhit__lambert_vidro() {

}

//miss lambert para vidros
extern "C" __global__ void __miss__lambert_vidro() {
    
}

//closest hit phong para vidros
extern "C" __global__ void __closesthit__phong_vidro() {

}

//any hit phong para vidros
extern "C" __global__ void __anyhit__phong_vidro() {

}

//miss phong para vidros
extern "C" __global__ void __miss__phong_vidro() {

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
        SURFACE_RAY_TYPE, // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        SURFACE_RAY_TYPE, // missSBTIndex
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

