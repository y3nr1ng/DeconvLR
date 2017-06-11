// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cufft.h>
// standard libraries headers
#include <cstdint>
// system headers

#define DIVUP(x, y) ((x+y-1)/y)

namespace PSF {

// deviated PSF
cudaArray_t d_psfDev = nullptr;
texture<float, cudaTextureType3D, cudaReadModeElementType> psfTexRef;

namespace {
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

    // sampling from the texture
    // (coordinates are backtracked to the deviated ones)
    int idx = iz * (nx*ny) + iy * nx + ix;
    odata[idx] = tex3D(psfTexRef, ix+ox+0.5f, iy+oy+0.5f, iz+oz+0.5f);
}
}

float estimateBackground(thrust::device_vector<float> &data) {
    float sum = thrust::reduce(
        thrust::device,
        data.begin(), data.end(),
        0,
        thrust::plus<float>()
    );
    return sum / data.size();
}

void removeBackground(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
) {
    // transfer to device
    const size_t nelem = nx * ny * nz;
    thrust::device_vector<float> d_psf(h_psf, h_psf+nelem);

    // estimate and remove the background, clamp at [0, +inf)
    const float bkgLvl = estimateBackground(d_psf);
    fprintf(stderr, "[DEBUG] background level = %.2f\n", bkgLvl);
    thrust::transform(
        d_psf.begin(), d_psf.end(),
        d_psf.begin(),
        SubConstant(bkgLvl)
    );

    // copy back to host
    thrust::copy(
        d_psf.begin(), d_psf.end(), h_psf
    );
}

float3 findCentroid(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
) {
    // pinned down the host memory region
    float *d_psf;
    const size_t nelem = nx * ny * nz;
    cudaErrChk(cudaHostRegister(
        h_psf,
        nelem * sizeof(float),
        cudaHostRegisterMapped
    ));
    cudaErrChk(cudaHostGetDevicePointer(&d_psf, h_psf, 0));

    // create a 3-D grid for weighting
    int3 *d_grid;
    cudaErrChk(cudaMalloc(&d_grid, nelem * sizeof(int3)));
    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    createGrid_kernel<<<nblocks, nthreads>>>(d_grid, nx, ny, nz);
    cudaErrChk(cudaPeekAtLastError());

    // calculate the centroid along grid points
    float4 result = thrust::transform_reduce(
        thrust::device,
        d_grid, d_grid + nelem,
        MultiplyWeighting(d_psf, nx, ny, nz),
        make_float4(0, 0, 0, 0),
        thrust::plus<float4>()
    );

    float3 centroid = make_float3(
        result.x/result.w, result.y/result.w, result.z/result.w
    );

    // release the resources
    cudaErrChk(cudaFree(d_grid));
    cudaErrChk(cudaHostUnregister(h_psf));

    return centroid;
}

void bindData(
    float *h_psf,
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
        &d_psfDev,
        &desc,
        extent,
        cudaArrayDefault
    ));

    // copy data from host to device
    cudaMemcpy3DParms parms = {0};
    parms.srcPtr = make_cudaPitchedPtr(
        h_psf,
        nx * sizeof(float), nx, ny
    );
    parms.dstArray = d_psfDev;
    parms.extent = extent;
    parms.kind = cudaMemcpyHostToDevice;
    cudaErrChk(cudaMemcpy3D(&parms));

    // texture coordinates are not normalized
    psfTexRef.normalized = false;   //TODO use normalized coordinate
    // sampled data is interpolated
    psfTexRef.filterMode = cudaFilterModeLinear;
    // wrap around the texture if exceeds border limit
    psfTexRef.addressMode[0] = cudaAddressModeWrap;
    psfTexRef.addressMode[1] = cudaAddressModeWrap;
    psfTexRef.addressMode[2] = cudaAddressModeWrap;

    // bind the texture
    cudaErrChk(cudaBindTextureToArray(
        psfTexRef,  // texture to bind
        d_psfDev,   // memory array on device
        desc        // channel format
    ));
}

void alignCenter(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz,
    const float3 centroid
) {
    // coordinate of the center of the volume
    const float3 center = make_float3(
        (nx-1)/2.0f, (ny-1)/2.0f, (nz-1)/2.0f
    );
    // offset
    const float3 offset = centroid - center;

    fprintf(stderr, "[DEBUG] offset = (%.2f, %.2f, %.2f)\n", offset.x, offset.y, offset.z);

    // pinned down the host memory region
    float *d_psf;
    cudaErrChk(cudaHostRegister(
        h_psf,
        nx * ny * nz * sizeof(float),
        cudaHostRegisterMapped
    ));
    cudaErrChk(cudaHostGetDevicePointer(&d_psf, h_psf, 0));

    // begin resample the kernel
    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    alignCenter_kernel<<<nblocks, nthreads>>>(
        d_psf,
        nx, ny, nz,
        offset.x, offset.y, offset.z
    );
    cudaErrChk(cudaPeekAtLastError());

    // release the resources
    cudaErrChk(cudaHostUnregister(h_psf));
}

void release() {
    // unbind the texture
    cudaErrChk(cudaUnbindTexture(psfTexRef));
    cudaErrChk(cudaFreeArray(d_psfDev));
}

}

namespace OTF {

// OTF template, used for interpolation
cudaArray_t d_otfTpl = nullptr;
texture<cufftComplex, cudaTextureType3D, cudaReadModeElementType> otfTexRef;

namespace {
__global__
void interpolate_kernel(
    cufftComplex *odata,
    const size_t nx, const size_t ny, const size_t nz,      // full size
    const size_t ntx, const size_t nty, const size_t ntz,   // template size
    const float dx, const float dy, const float dz          // voxel ratio
) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    // skip out-of-bound threads
    if (ix >= nx or iy >= ny or iz >= nz) {
        return;
    }

    // shift to center, (0, N-1) -> (-N/2, N/2+1)
    float fx = ix - (nx-1)/2.0f;
    float fy = iy - (ny-1)/2.0f;
    float fz = iz - (nz-1)/2.0f;
    // dilate to the coordinate of the template OTF
    fx *= dx;
    fy *= dy;
    fz *= dz;
    // shift back to origin, (-M/2, M/2+1) -> (0, M-1)
    fx += (ntx-1)/2.0f;
    fy += (nty-1)/2.0f;
    fz += (ntz-1)/2.0f;
    // wrap around if exceeds the size
    if (fx > ntx) {
        fx -= ntx;
    }
    if (fy > nty) {
        fy -= nty;
    }
    if (fz > ntz) {
        fz -= ntz;
    }
ˋ
    // wrap around
    if (fz < 0) {
        fz += ntz;
    }

    // sampling from the texture
    // (coordinates are backtracked to the deviated ones)
    int idx = iz * (nx*ny) + iy * nx + ix;
    odata[idx] = tex3D(otfTexRef, fx+0.5f, fy+0.5f, fz+0.5f);
}

__global__
void magnitude_kernel(
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
    float re = idata[idx].x;
    float im = idata[idx].y;
    odata[idx] = std::sqrt(re*re + im*im);
}
}

void fromPSF(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
) {
    // pinned down the host memory region
    float *d_psf;
    cudaErrChk(cudaHostRegister(
        h_psf,
        nx * ny * nz * sizeof(float),
        cudaHostRegisterMapped
    ));
    cudaErrChk(cudaHostGetDevicePointer(&d_psf, h_psf, 0));

    // create FFT plan
    cufftHandle otfHdl;
    cudaErrChk(cufftPlan3d(
        &otfHdl,
        nz, ny, nx,
        CUFFT_R2C
    ));
    // estimate resource requirements
    size_t wsSz;
    cudaErrChk(cufftGetSize3d(
        otfHdl,
        nz, ny, nx,
        CUFFT_R2C,
        &wsSz
    ));
    fprintf(stderr, "[DEBUG] PSF -> OTF requires %ld bytes\n", wsSz);

    // allocate device memory to buffer the result
    cufftComplex *d_otf;
    cudaErrChk(cudaMalloc(
        &d_otf,
        (nx/2+1) * ny * nz * sizeof(cufftComplex)
    ));

    // begin PSF to OTF
    cudaErrChk(cufftExecR2C(otfHdl, d_psf, d_otf));

    // release resources regarding the PSF
    cudaErrChk(cufftDestroy(otfHdl));
    cudaErrChk(cudaHostUnregister(h_psf));

    // bind OTF to texture as template
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(
        32, 32, 0, 0, cudaChannelFormatKindFloat
    );
    cudaExtent extent = make_cudaExtent(
        (nx/2+1), ny, nz
    );
    cudaErrChk(cudaMalloc3DArray(
        &d_otfTpl,
        &desc,
        extent,
        cudaArrayDefault
    ));

    // copy data from host to device
    cudaMemcpy3DParms parms = {0};
    parms.srcPtr = make_cudaPitchedPtr(
        d_otf,
        (nx/2+1) * sizeof(cufftComplex), (nx/2+1), ny
    );
    parms.dstArray = d_otfTpl;
    parms.extent = extent;
    parms.kind = cudaMemcpyDeviceToDevice;
    cudaErrChk(cudaMemcpy3D(&parms));

    // texture coordinates are not normalized
    otfTexRef.normalized = false;
    // sampled data is interpolated
    otfTexRef.filterMode = cudaFilterModeLinear;
    // wrap around the texture if exceeds border limit
    otfTexRef.addressMode[0] = cudaAddressModeBorder;
    otfTexRef.addressMode[1] = cudaAddressModeBorder;
    otfTexRef.addressMode[2] = cudaAddressModeBorder;

    // bind the texture
    cudaErrChk(cudaBindTextureToArray(
        otfTexRef,  // texture to bind
        d_otfTpl,   // memory array on device
        desc        // channel format
    ));

    // release the resources
    cudaErrChk(cudaFree(d_otf));
}

void interpolate(
    cufftComplex *d_otf,
    const size_t nx, const size_t ny, const size_t nz,      // full size
    const size_t ntx, const size_t nty, const size_t ntz,   // template size
    const float dx, const float dy, const float dz          // voxel ratio
) {
    // start the interpolation
    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    interpolate_kernel<<<nblocks, nthreads>>>(
        d_otf,
        nx, ny, nz,
        ntx, nty, ntz,
        dx, dy, dz
    );
    cudaErrChk(cudaPeekAtLastError());
}

void release() {
    // unbind the texture
    cudaErrChk(cudaUnbindTexture(otfTexRef));
    cudaErrChk(cudaFreeArray(d_otfTpl));
}

void dumpTemplate(
    float *h_otf,
    const size_t nx, const size_t ny, const size_t nz
) {
    // pinned down the host memory region
    float *d_otfDump;
    cudaErrChk(cudaHostRegister(
        h_otf,
        nx * ny * nz * sizeof(float),
        cudaHostRegisterMapped
    ));
    cudaErrChk(cudaHostGetDevicePointer(&d_otfDump, h_otf, 0));

    // create linear template OTF buffer space
    cufftComplex *d_otfLinTpl;
    cudaErrChk(cudaMalloc(
        &d_otfLinTpl,
        nx * ny * nz * sizeof(cufftComplex)
    ));

    // copy out the template to linear mode
    cudaMemcpy3DParms parms = {0};
    parms.srcArray = d_otfTpl;
    parms.dstPtr = make_cudaPitchedPtr(
        d_otfLinTpl,
        nx * sizeof(cufftComplex), (nx/2+1), ny
    );
    parms.extent = make_cudaExtent(
        nx, ny, nz
    );
    parms.kind = cudaMemcpyDeviceToDevice;
    cudaErrChk(cudaMemcpy3D(&parms));

    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    magnitude_kernel<<<nblocks, nthreads>>>(
        d_otfDump,
        d_otfLinTpl,
        nx, ny, nz
    );
    cudaErrChk(cudaPeekAtLastError());

    // release the resources
    cudaErrChk(cudaFree(d_otfLinTpl));
    cudaErrChk(cudaHostUnregister(h_otf));
}

void dumpComplex(
    float *h_odata,
    const cufftComplex *d_idata,
    const size_t nx, const size_t ny, const size_t nz
) {
    // pinned down the host memory region
    float *d_odata;
    cudaErrChk(cudaHostRegister(
        h_odata,
        nx * ny * nz * sizeof(float),
        cudaHostRegisterMapped
    ));
    cudaErrChk(cudaHostGetDevicePointer(&d_odata, h_odata, 0));

    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    magnitude_kernel<<<nblocks, nthreads>>>(
        d_odata,
        d_idata,
        nx, ny, nz
    );
    cudaErrChk(cudaPeekAtLastError());

    // release the resources
    cudaErrChk(cudaHostUnregister(h_odata));
}

}
