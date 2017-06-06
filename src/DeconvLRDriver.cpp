// corresponded header file
#include "DeconvLRDriver.hpp"
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cufft.h>
// standard libraries headers
#include <exception>
#include <cmath>
// system headers

struct DeconvLR::Impl {
    Impl() {

    }
    ~Impl() {

    }

    // volume size
    dim3 volumeSize;
    // voxel ratio = raw voxel size / PSF voxel size
    float3 voxelRatio;

    /*
     * Device pointers
     */
    cudaPitchedPtr *otf;
};

DeconvLR::DeconvLR()
    : pimpl(std::make_unique<Impl>()) {
}

DeconvLR::~DeconvLR() {

}

void DeconvLR::setResolution(
    const float dx, const float dy, const float dz,
    const float dpx, const float dpy, const float dpz
) {
    pimpl->voxelRatio = make_float3(dx/dpx, dy/dpy, dz/dpz);
}

void DeconvLR::setVolumeSize(
    const size_t nx, const size_t ny, const size_t nz
) {
    if (nx > 4096 or ny > 4096 or nz > 4096) {
        throw std::range_error("volume size exceeds maximum constraints");
    }
    pimpl->volumeSize.x = nx;
    pimpl->volumeSize.y = ny;
    pimpl->volumeSize.z = nz;
}

void DeconvLR::setPSF(const ImageStack<uint16_t> &psf) {
    uint16_t *hPsf;
    dim3 psfSize(psf.nx(), psf.ny(), psf.nz());

    // pin down the host memory
    cudaErrChk(cudaHostRegister(
        psf.data(),
        psfSize.x * psfSize.y * psfSize.z * sizeof(uint16_t),
        cudaHostRegisterMapped
    ));
    cudaErrChk(cudaHostGetDevicePointer(
        &hPsf,    // device pointer for mapped address
        psf.data(), // requested host pointer
        0
    ));

    /*
     * Convert from uint16_t to cufftReal
     */
    cufftReal *dPsf;
    //TODO

    /*
     * cuFFT R2C
     */
    cudaPitchedPtr otfLut;
    cufftHandle otfFFTHandle;

    // create workspace for the OTF
    cudaExtent otfSize = make_cudaExtent(
        psfSize.x * sizeof(cufftComplex),   // width in bytes
        psfSize.y,
        std::floor((float)psfSize.z / 2) + 1
    );
    cudaErrChk(cudaMalloc3D(&otfLut, otfSize));

    // plan and execute FFT
    cudaErrChk(cufftPlan3d(
        &otfFFTHandle,
        otfSize.width, otfSize.height, otfSize.depth,
        CUFFT_R2C
    ));
    cudaErrChk(cufftExecR2C(otfFFTHandle, dPsf, (cufftComplex *)otfLut.ptr));

    // release the resources
    cudaErrChk(cufftDestroy(otfFFTHandle));

    // convert to CUDA array
    cudaArray_t otfLutArray;
    cudaChannelFormatDesc formDesc = cudaCreateChannelDesc(
        32, 0, 0, 0,
        cudaChannelFormatKindFloat
    );

    cudaErrChk(cudaMalloc3DArray(
        &otfLutArray,
        &formDesc,
        otfSize,
        cudaArrayDefault
    ));

    cudaMemcpy3DParms parms = {0};
    parms.srcPtr = otfLut;
    parms.dstArray = otfLutArray;
    parms.extent = otfSize;
    parms.kind = cudaMemcpyDeviceToDevice;
    cudaErrChk(cudaMemcpy3D(&parms));

    // release the template OTF
    cudaErrChk(cudaFree(otfLut.ptr));

    /*
     * Interpolate to volume size
     */
    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = otfLutArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    // out-of-border pixels are 0
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    // access by [0, 1]
    texDesc.readMode = cudaReadModeNormalizedFloat;

    // bind to texture object
    cudaTextureObject_t otfTex = 0;
    cudaErrChk(cudaCreateTextureObject(&otfTex, &resDesc, &texDesc, NULL));

    // interpolate
    //TODO

    /*
     * Cleanup
     */
    // release the texture resources
    cudaErrChk(cudaDestroyTextureObject(otfTex));
    cudaErrChk(cudaFreeArray(otfLutArray));
    // unregister the host memory region
    cudaErrChk(cudaHostUnregister(psf.data()));
}

void DeconvLR::process(
    ImageStack<uint16_t> &output,
    const ImageStack<uint16_t> &input
) {

}
