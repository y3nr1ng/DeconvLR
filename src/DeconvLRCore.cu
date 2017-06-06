// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
// 3rd party libraries headers
#include <cufft.h>
// standard libraries headers
// system headers

namespace Kernel {

inline int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <typename T_out, typename T_in>
__global__
void convertTypeKernel(T_out *dst, T_in *src, const cudaExtent size) {
    //TODO fill the blank here, type convert form T_in to T_out
}

template <typename T_out, typename T_in>
__host__
void convertType(T_out *dst, T_in *src, const cudaExtent size) {
    int nSMs;
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, 0);

    convertTypeKernel<T_out, T_in><<<32*nSMs, 256>>>(dst, src, size);
}

// explicit instantiation
template void convertType(cufftReal *dst, uint16_t *src, const cudaExtent size);

}
