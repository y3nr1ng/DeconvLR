// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
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

struct WeightedSum
    : public thrust::binary_function<float4, float4, float4> {
    __host__ __device__
    float4 operator()(const float4 &a, const float4 &b) const {
        return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
    }
};

__global__
void alignCenter_kernel(
    float *d_odata,
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
    d_odata[idx] = tex3D(psfTexRef, ix+ox+0.5f, iy+oy+0.5f, iz+oz+0.5f);
}
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
        WeightedSum()
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
    psfTexRef.normalized = false;
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

namespace Kernel {

texture<cufftComplex, 2, cudaReadModeElementType> otfTex;

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
