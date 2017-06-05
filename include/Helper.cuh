#ifndef HELPER_CUH
#define HELPER_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
#include <cstdio>
#include <cstdlib>
// system headers

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

inline __host__ __device__ float3 operator*(float3 &a, uint3 b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __host__ __device__ float3 operator/(float a, float3 b)
{
    return make_float3(a/b.x, a/b.y, a/b.z);
}

inline __host__ __device__ float3 operator/(float3 &a, float3 &b)
{
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

#endif
