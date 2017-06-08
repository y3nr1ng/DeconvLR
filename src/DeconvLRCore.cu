// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
// 3rd party libraries headers
#include <cufft.h>
// standard libraries headers
#include <cstdint>
// system headers

namespace Kernel {

inline int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <typename T_out, typename T_in>
__global__
void convertTypeKernel(T_out *dst, T_in *src,
                       const int nx, const int ny, const int nz,
                       const size_t pitchDst, const size_t pitchSrc) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < nx * ny * nz; 
         i += blockDim.x * gridDim.x) {
        int z = i / nx / ny;
        int y = (i / nx) % ny;
        int x = i % nx;
        int nPitchSrc = pitchSrc / sizeof(T_in);
        int nPitchDst = pitchDst / sizeof(T_out);
        T_in* ptrSrc = src + (z * ny + y) * nPitchSrc + x;
        T_out* ptrDst = dst + (z * ny + y) * nPitchDst + x;
        *ptrDst = (T_out)(*ptrSrc);
    }
}

template <typename T_out, typename T_in>
__host__
void convertType(T_out *dst, T_in *src,
                 const cudaExtent extDst, const cudaExtent extSrc) {
    int nSMs;
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, 0);
    int nx = extSrc.width / sizeof(T_in);
    int ny = extSrc.height;
    int nz = extSrc.depth;
    convertTypeKernel<T_out, T_in><<<32*nSMs, 256>>>(dst, src, nx, ny, nz,
                                                     extDst.width, extSrc.width);
}

// explicit instantiation
template void convertType(cufftReal *dst, uint16_t *src,
                          const cudaExtent extDst, const cudaExtent extSrc);

}
