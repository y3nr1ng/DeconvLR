// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
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

#define DIVUP(x, y) ((x+y-1)/y)

namespace PSF {

// deviated PSF
cudaArray_t d_psfDev = nullptr;
texture<float, cudaTextureType3D, cudaReadModeElementType> psfTexRef;

namespace {
//TODO use template to determine the cutoff
//TODO rename to signify clamping
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
    //TODO don't modify the original data

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

    psfTexRef.normalized = true;
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
void fftshift3_kernel(
    cufftReal *odata,
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
    float flip = 1 - 2*(~(((ix+iy)&1) ^ (iz&1)));
    odata[idx] *= flip;
}

__global__
void interpolate_kernel(
    cufftComplex *odata,
    const size_t nx, const size_t ny, const size_t nz,      // full size
    const size_t ntx, const size_t nty, const size_t ntz,   // template size
    const float dx, const float dy, const float dz,
    const float dtx, const float dty, const float dtz
) {
    // index
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    // skip out-of-bound threads
    if (ix >= nx or iy >= ny or iz >= nz) {
        return;
    }

    // convert to spatial frequency
    float fx = ix / (nx*dx);
    float fy = iy / (ny*dy);
    float fz = iz / (nz*dz);
    // wrap if half if greater than half
    //if (fx > 1/(2*dx)) {
    //    fx = (nx-ix) / (nx*dx);
    //}
    if (fy > 1/(2*dy)) {
        fy = (ny-iy) / (ny*dy);
    }
    if (fz > 1/(2*dz)) {
        fz = (nz-iz) / (nz*dz);
    }

    // convert to index in the templated OTF
    fx *= ntx*dtx;
    fy *= nty*dty;
    fz *= ntz*dtz;

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

    /*
    // fftshift
    dim3 nthreads(16, 16, 4);
    dim3 nblocks(
        DIVUP(nx, nthreads.x), DIVUP(ny, nthreads.y), DIVUP(nz, nthreads.z)
    );
    fftshift3_kernel<<<nblocks, nthreads>>>(
        d_psf,
        nx, ny, nz
    );
    */

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
    const float dx, const float dy, const float dz,
    const float dtx, const float dty, const float dtz
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
        dx, dy, dz,
        dtx, dty, dtz
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

namespace Core {

namespace RL {

enum class ConvType {
    PLAIN = 1, CONJUGATE
};

namespace {
// generic complex number operation
template <ConvType type>
struct MultiplyAndScale
    : public thrust::binary_function<cuComplex, cuComplex, cuComplex> {
    MultiplyAndScale(const float c_)
        : c(c_) {
        if (type == ConvType::CONJUGATE) {
            printf("CONJUGATE\n");
        } else {
            printf("PLAIN\n");
        }
    }

    __host__ __device__
    cuComplex operator()(const cuComplex &a, const cuComplex &b) const {
        if (type == ConvType::CONJUGATE) {
            return cuCmulf(a, cuConjf(b))/c;
        } else {
            return cuCmulf(a, b)/c;
        }
    }

private:
    const float c;
};

template <ConvType type>
void filter(
    cufftReal *odata, cufftReal *idata, const cufftComplex *otf,
    Core::RL::Parameters &parm
) {
    fprintf(stderr, "[DEBUG] +++ ENTER RL::(anon)::step() +++\n");

    const size_t nelem = (parm.nx/2+1) * parm.ny * parm.nz;
    cufftComplex *buffer = (cufftComplex *)parm.FFTBuffer.complexA;

    // convert to frequency space
    cudaErrChk(cufftExecR2C(
        parm.fftHandle.forward,
        idata,
        buffer
    ));

    // element-wise multiplication and scale down
    thrust::transform(
        thrust::device,
        buffer, buffer+nelem,       // first input sequence
        otf,                        // second input sequence
        buffer,                     // output sequence
        MultiplyAndScale<type>(1.0f/nelem)
    );

    // convert back to real space
    cudaErrChk(cufftExecC2R(
        parm.fftHandle.reverse,
        buffer,
        odata
    ));

    fprintf(stderr, "[DEBUG] +++ EXIT RL::(anon)::step() +++\n");
}

thrust::divides<float> DivfOp;
thrust::multiplies<float> MulfOp;

}

void step(
    float *odata, float *idata,
    Core::RL::Parameters &parm
) {
    fprintf(stderr, "[DEBUG] +++ ENTER RL::step() +++\n");

    const size_t nelem = parm.nelem;
    cufftReal *buffer = (cufftReal *)parm.FFTBuffer.complexA;

    cufftComplex *otf = parm.otf;

    /*
    CImg<float> dump(parm.nx, parm.ny, parm.nz);
    Common::dumpDeviceReal(
        dump.data(),
        odata,
        dump.width(), dump.height(), dump.depth()
    );
    dump.save_tiff("dump.tif");
    */

    /*
     * \hat{f_{k+1}} =
     *     \hat{f_k} \left(
     *         h \ast \frac{g}{h \otimes \hat{f_k}}
     *     \right)
     */

    // reblur the image
    filter<ConvType::PLAIN>(odata, const_cast<cufftReal *>(idata), otf, parm);
    //filter<ConvType::PLAIN>(buffer, idata, otf, parm);
    /*
    fprintf(stderr, "B\n");
    // error
    thrust::transform(
        thrust::device,
        parm.raw,  parm.raw+nelem,  // first input sequence
        buffer,                     // second input sequence
        buffer,                     // output sequence
        DivfOp
    );
    fprintf(stderr, "C\n");
    filter<ConvType::CONJUGATE>(buffer, buffer, otf, parm);
    fprintf(stderr, "D\n");
    // latent image
    thrust::transform(
        thrust::device,
        idata, idata+nelem,         // first input sequence
        buffer,                     // second input sequence
        odata,                      // output sequence
        MulfOp
    );
    fprintf(stderr, "E\n");
    */

    fprintf(stderr, "[DEBUG] +++ EXIT RL::step() +++\n");
}

}

namespace Biggs {

namespace {

}

void step(
    float *odata, const float *idata,
    Core::RL::Parameters &parm
) {
    // execute an iteration of RL
    //RL::step();

    // find the update direction

    // calculate acceleration factor

    // re-estimate prediction
}

}

}

namespace Common {

namespace {

template <typename T>
struct ToFloat
    : public thrust::unary_function<const T, float> {
    __host__ __device__
    float operator()(const T &v) const {
        return (float)v;
    }
};

}

void ushort2float(float *odata, const uint16_t *idata, const size_t nelem) {
    thrust::transform(
        thrust::device,
        idata, idata + nelem,   // input
        odata,                  // output
        ToFloat<uint16_t>()
    );
}

void dumpDeviceReal(
    float *h_odata,
    const cufftReal *d_idata,
    const size_t nx, const size_t ny, const size_t nz
) {
    const size_t dataSize = nx * ny * nz * sizeof(float);

    // pinned down the host memory region
    float *d_odata;
    cudaErrChk(cudaHostRegister(h_odata, dataSize, cudaHostRegisterMapped));
    cudaErrChk(cudaHostGetDevicePointer(&d_odata, h_odata, 0));

    // copy from device to host
    cudaErrChk(cudaMemcpy(d_odata, d_idata, dataSize, cudaMemcpyDeviceToHost));

    // release the resources
    cudaErrChk(cudaHostUnregister(h_odata));
}

}
