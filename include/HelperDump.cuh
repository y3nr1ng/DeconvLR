#ifndef HELPER_DUMP_CUH
#define HELPER_DUMP_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cufft.h>

#define cimg_use_tiff
#include "CImg.h"
using namespace cimg_library;
// standard libraries headers
#include <string>
#include <cstring>
// system headers

namespace DumpData {

namespace {

__global__
void abs_kernel(
    cufftReal *odata,
    const cufftComplex *idata,
    const size_t nx, const size_t ny, const size_t nz
) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    // skip out-of-bound threads
    if (ix >= nx or iy >= ny or iz >= nz) {
        return;
    }

    int idx = iz * (nx*ny) + iy * nx + ix;
    odata[idx] = cuCabsf(idata[idx]);
}

}

namespace Device {

void real(
    std::string fname,
    const cufftReal *d_idata,
    const size_t nx, const size_t ny, const size_t nz
) {
    CImg<float> data(nx, ny, nz);
    const size_t size = nx * ny * nz * sizeof(float);

    // pinned down the host memory region
    float *d_odata;
    cudaErrChk(cudaHostRegister(h_odata, size, cudaHostRegisterMapped));
    cudaErrChk(cudaHostGetDevicePointer(&d_odata, h_odata, 0));

    // copy from device to host
    cudaErrChk(cudaMemcpy(d_odata, d_idata, dataSize, cudaMemcpyDeviceToHost));

    // release the resources
    cudaErrChk(cudaHostUnregister(h_odata));

    // save the result to file
    data.saveAs(fname.c_str());
}

void complex(
    std::string fname,
    const cufftComplex *d_idata,
    const size_t nx, const size_t ny, const size_t nz
) {
    CImg<float> data(nx, ny, nz);
    const size_t size = nx * ny * nz * sizeof(float);

    // pinned down the host memory region
    float *d_odata;
    cudaErrChk(cudaHostRegister(h_odata, size, cudaHostRegisterMapped));
    cudaErrChk(cudaHostGetDevicePointer(&d_odata, h_odata, 0));

    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    abs_kernel<<<nblocks, nthreads>>>(
        d_odata,
        d_idata,
        nx, ny, nz
    );
    cudaErrChk(cudaPeekAtLastError());

    // release the resources
    cudaErrChk(cudaHostUnregister(h_odata));

    // save the result to file
    data.saveAs(fname.c_str());
}

}

namespace Host {

void real(
    std::string fname,
    cufftReal *h_idata,
    const size_t nx, const size_t ny, const size_t nz
) {
    CImg<float> data(nx, ny, nz);
    const size_t size = nx * ny * nz * sizeof(float);

    // copy to image data region
    std::memcpy(data.data(), h_idata, size);

    // save the result to file
    data.saveAs(fname.c_str());
}

}

}

#endif
