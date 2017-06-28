// corresponded header file
// necessary project headers
#include "DeconvRLImpl.cuh"
#include "Helper.cuh"
#include "DumpData.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cufft.h>

#define cimg_use_tiff
#include "CImg.h"
using namespace cimg_library;
// standard libraries headers
#include <cstdint>
// system headers

namespace DeconvRL {

namespace PSF {

namespace {

cudaArray_t psfRes = nullptr;
texture<float, cudaTextureType3D, cudaReadModeElementType> psfTexRef;

struct SubConstant
    : public thrust::unary_function<float, float> {
    SubConstant(const float c_)
        : c(c_) {
    }

    __host__ __device__
    float operator()(const float &p) const {
        float o = p-c;
        return (o < 0) ? 0 : o;
    }

private:
    const float c;
};

__global__
void createGrid_kernel(
    int3 *d_grid,
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
    d_grid[idx] = make_int3(ix, iy, iz);
}

struct MultiplyWeighting
    : public thrust::unary_function<int3, float4> {
    MultiplyWeighting(
        const float *data,
        const size_t nx_, const size_t ny_, const size_t nz_
    )
        : d_weight(data), nx(nx_), ny(ny_), nz(nz_) {
    }

    __host__ __device__
    float4 operator()(const int3 &p) const {
        const int idx = p.z * (nx*ny) + p.y * nx + p.x;
        const float w = d_weight[idx];
        return make_float4(p.x*w, p.y*w, p.z*w, w);
    }

private:
    const float *d_weight;
    size_t nx, ny, nz;
};

__global__
void alignCenter_kernel(
    float *odata,
    const size_t nx, const size_t ny, const size_t nz,
    const float ox, const float oy, const float oz
) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    // skip out-of-bound threads
    if (ix >= nx or iy >= ny or iz >= nz) {
        return;
    }

    // normalized coordinate
    float fx = (ix+ox+0.5f) / nx;
    float fy = (iy+oy+0.5f) / ny;
    float fz = (iz+oz+0.5f) / nz;

    // sampling from the texture
    // (coordinates are backtracked to the deviated ones)
    int idx = iz * (nx*ny) + iy * nx + ix;
    odata[idx] = tex3D(psfTexRef, fx, fy, fz);
}

}

PSF::PSF(
    float *h_psf_,
    const size_t npx_, const size_t npy_, const size_t npz_
) : h_psf(h_psf_), npx(npx_), npy(npy_), npz(npz_) {
    nelem = npx * npy * npz;

    // pinned down the host memory region
    cudaErrChk(cudaHostRegister(
        h_psf,
        nelem * sizeof(float),
        cudaHostRegisterMapped
    ));
    // retrieve the device address
    cudaErrChk(cudaHostGetDevicePointer(&d_psf, h_psf, 0));
}

PSF::~PSF() {
    cudaErrChk(cudaHostUnregister(h_psf));
}

void PSF::alignCenter() {
    float3 centroid = findCentroid();
    fprintf(
        stderr,
        "[INF] centroid = (%.2f, %.2f, %.2f)\n",
        centroid.x, centroid.y, centroid.z
    );

    /*
     * Bind the data source to the texture.
     */
    // create cudaArray for the texture.
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(
        32, 0, 0, 0, cudaChannelFormatKindFloat
    );
    cudaExtent extent = make_cudaExtent(npx, npy, npz);
    cudaErrChk(cudaMalloc3DArray(
        &psfRes,
        &desc,      // pixel channel description
        extent,     // array dimension
        cudaArrayDefault
    ));

    // copy the data to cudaArray_t
    cudaMemcpy3DParms parms = {0};
    parms.srcPtr = make_cudaPitchedPtr(h_psf, npx * sizeof(float), npx, npy);
    parms.dstArray = psfRes;
    parms.extent = extent;
    parms.kind = cudaMemcpyHostToDevice;
    cudaErrChk(cudaMemcpy3D(&parms));

    // reconfigure the texture
    psfTexRef.normalized = true;
    // sampled data is interpolated
    psfTexRef.filterMode = cudaFilterModeLinear;
    // wrap around the texture if exceeds border limit
    psfTexRef.addressMode[0] = cudaAddressModeWrap;
    psfTexRef.addressMode[1] = cudaAddressModeWrap;
    psfTexRef.addressMode[2] = cudaAddressModeWrap;

    // start the binding
    cudaErrChk(cudaBindTextureToArray(psfTexRef, psfRes));

    /*
     * Execute the alignment kernel.
     */
    // coordinate of the center of the volume
    const float3 center = make_float3(
        (npx-1)/2.0f, (npy-1)/2.0f, (npz-1)/2.0f
    );
    // offset
    const float3 offset = centroid - center;
    fprintf(stderr, "[DBG] offset = (%.2f, %.2f, %.2f)\n", offset.x, offset.y, offset.z);

    // begin resample the kernel
    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(npx, nthreads.x), DIVUP(npy, nthreads.y), DIVUP(npz, nthreads.z)
    );
    alignCenter_kernel<<<nblocks, nthreads>>>(
        d_psf,
        npx, npy, npz,
        offset.x, offset.y, offset.z
    );
    cudaErrChk(cudaPeekAtLastError());

    /*
     * Release the resources.
     */
    cudaErrChk(cudaUnbindTexture(psfTexRef));
    cudaErrChk(cudaFreeArray(psfRes));

    DumpData::Host::real("psf_aligned.tif", h_psf, npx, npy, npz);
}

void PSF::createOTF(
    cufftComplex *d_otf,
    const size_t nx, const size_t ny, const size_t nz
) {
    /*
     * Prepare FFT environment.
     */
    cufftHandle otfHdl;
    cudaErrChk(cufftPlan3d(&otfHdl, nz, ny, nx, CUFFT_R2C));
    // estimate resource requirements
    size_t size;
    cudaErrChk(cufftGetSize3d(otfHdl, nz, ny, nx, CUFFT_R2C, &size));
    fprintf(stderr, "[DBG] requires %ld bytes to generate an OTF\n", size);

    /*
     * Execute the conversion.
     */
    cudaErrChk(cufftExecR2C(otfHdl, d_psf, d_otf));

    // release FFT resource
    cudaErrChk(cufftDestroy(otfHdl));

    DumpData::Device::complex("otf_dump.tif", d_otf, nx/2+1, ny, nz);
}

// center the PSF to its potential centroid
float3 PSF::findCentroid() {
    const size_t size = nelem * sizeof(float);

    /*
     * Create temporary PSF to find the centroid.
     */
    float *d_tmp;
    cudaErrChk(cudaMalloc(&d_tmp, size));
    // copy the raw PSF to temporary PSF
    cudaErrChk(cudaMemcpy(d_tmp, h_psf, size, cudaMemcpyHostToDevice));

    // estimate and remove the background, clamp at [0, +inf)
    const float bkgLvl = estimateBackground();
    fprintf(stderr, "[INF] background level = %.2f\n", bkgLvl);
    thrust::transform(
        thrust::device,
        d_tmp, d_tmp+nelem,
        d_tmp,
        SubConstant(bkgLvl)
    );

    /*
     * Generate 3-D grid for weighting.
     */
    int3 *d_grid;
    cudaErrChk(cudaMalloc(&d_grid, nelem * sizeof(int3)));
    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(npx, nthreads.x), DIVUP(npy, nthreads.y), DIVUP(npz, nthreads.z)
    );
    createGrid_kernel<<<nblocks, nthreads>>>(d_grid, npx, npy, npz);
    cudaErrChk(cudaPeekAtLastError());

    /*
     * Calculate the centroid along weighted grid points using cleaned PSF.
     */
    float4 result = thrust::transform_reduce(
        thrust::device,
        d_grid, d_grid+nelem,
        MultiplyWeighting(d_tmp, npx, npy, npz),
        make_float4(0),
        thrust::plus<float4>()
    );

    float3 centroid = make_float3(
        result.x/result.w, result.y/result.w, result.z/result.w
    );

    // free the weight computation resources
    cudaErrChk(cudaFree(d_grid));
    cudaErrChk(cudaFree(d_tmp));

    return centroid;
}

float PSF::estimateBackground() {
    float sum = thrust::reduce(
        thrust::device,
        d_psf, d_psf+nelem,
        0,
        thrust::plus<float>()
    );
    return sum/nelem;
}

}

}
