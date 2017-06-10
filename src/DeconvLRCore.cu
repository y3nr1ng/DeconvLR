// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cufft.h>
// standard libraries headers
#include <cstdint>
// system headers

namespace PSF {

cudaArray_t d_psf = nullptr;
texture<texture, 3, cudaReadModeElementType> psfTex;

void bindData(
    const float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
) {
    // create cudaArray for the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(
        32, 0, 0, 0, cudaChannelFormatKindFloat
    );
    cudaExtent extent = make_cudaExtent(
        nx, ny, nz
    );
    cudaErrChk(cudaMalloc3DArray(
        &d_psf,
        &desc,
        extent,
        cudaArrayDefault
    ));

    // copy data from host to device
    cudaMemcpy3DParms parms = {0};
    parms.srcPtr = make_cudaPitchedPtr(
        h_psf,
        nx * sizeof(float), nx, ny
    ),
    parms.dstArray = d_psf,
    parms.extent = extent,
    parms.kind = cudaMemcpyHostToDevice
    cudaErrChk(cudaMemcpy3D(&parms));

    // bind the texture
    cudaErrChk(cudaBindTextureToArray(
        psfTex,     // texture to bind
        d_psf,      // memory array on device
        &desc       // channel format
    ));
}

void findCenter(float *cx, float *cy, float *cz) {

}

void alignCenter() {

}

void release() {
    if (d_psf != nullptr) {
        cudaErrChk(cudaFreeArray(d_psf));
        // unbind the texture
        cudaErrChk(cudaUnbindTexture(psfTex));
    }
}

}

namespace Kernel {

texture<cufftComplex, 2, cudaReadModeElementType> otfTex;

__host__
void interpolateOTF() {

}

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

/*
 * Explicit instantiation
 */
template void convertType(cufftReal *dst, uint16_t *src,
                          const cudaExtent extDst, const cudaExtent extSrc);

}
