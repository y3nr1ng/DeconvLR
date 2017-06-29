// corresponded header file
#include "DeconvRLDriver.hpp"
// necessary project headers
#include "DeconvRLImpl.cuh"
#include "Helper.cuh"
#include "DumpData.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
#include <exception>
#include <cstdio>
// system headers

namespace DeconvRL {

struct DeconvRL::Impl {
    Impl()
        : iterations(1) {

    }

    ~Impl() {
        // TODO free iterParms
    }

    // volume size
    dim3 volumeSize;
    // voxel size
    struct {
        float3 raw;
        float3 psf;
    } voxelSize;

    /*
     * Algorithm configurations.
     */
    int iterations;
    Core::Parameters iterParms;
};

// C++14 feature
template<typename T, typename ... Args>
std::unique_ptr<T> make_unique(Args&& ... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
}

DeconvRL::DeconvRL()
    : pimpl(make_unique<Impl>()) {
}

DeconvRL::~DeconvRL() {

}

void DeconvRL::setResolution(
    const float dx, const float dy, const float dz,
    const float dpx, const float dpy, const float dpz
) {
    /*
     * Spatial frequency ratio (along one dimension)
     *
     *       1/(NS * DS)   NP   DP   NP
     *   R = ----------- = -- * -- = -- * r
     *       1/(NP * DP)   NS   DS   NS
     *
     *   NS, sample size
     *   DS, sample voxel size
     *   NP, PSF size
     *   DP, PSF voxel size
     *   r, voxel ratio
     */
    pimpl->voxelSize.raw = make_float3(dx, dy, dz);
    pimpl->voxelSize.psf = make_float3(dpx, dpy, dpz);
}

void DeconvRL::setVolumeSize(
    const size_t nx, const size_t ny, const size_t nz
) {
    //TODO probe for device specification
    if (nx > 2048 or ny > 2048 or nz > 2048) {
        throw std::range_error("volume size exceeds maximum constraints");
    }
    pimpl->volumeSize.x = nx;
    pimpl->volumeSize.y = ny;
    pimpl->volumeSize.z = nz;

    fprintf(
        stderr,
        "[INF] volume size = %ux%ux%u\n",
        pimpl->volumeSize.x, pimpl->volumeSize.y, pimpl->volumeSize.z
    );
}

//TODO remove ImageStack dependency
void DeconvRL::setPSF(const ImageStack<uint16_t> &psf_u16) {
    /*
     * Ensure we are working with floating points.
     */
    ImageStack<float> psf(psf_u16);
    fprintf(
        stderr,
        "[INF] PSF size = %ldx%ldx%ld\n",
        psf.nx(), psf.ny(), psf.nz()
    );

    /*
     * Generate the OTF.
     */
    PSF::PSF psfProc(psf.data(), psf.nx(), psf.ny(), psf.nz());
    psfProc.alignCenter(
        pimpl->volumeSize.x, pimpl->volumeSize.y, pimpl->volumeSize.z
    );

    // allocate memory space for OTF
    cudaErrChk(cudaMalloc(
        &pimpl->iterParms.otf,
        (pimpl->volumeSize.x/2+1) * pimpl->volumeSize.y * pimpl->volumeSize.z * sizeof(cufftComplex)
    ));
    // create the OTF
    psfProc.createOTF(pimpl->iterParms.otf);
    fprintf(stderr, "[INF] OTF established\n");
}

void DeconvRL::initialize() {
    const dim3 volumeSize = pimpl->volumeSize;
    Core::Parameters &iterParms = pimpl->iterParms;

    /*
     * Load dimension information into the iteration parameter.
     */
    iterParms.nx = volumeSize.x;
    iterParms.ny = volumeSize.y;
    iterParms.nz = volumeSize.z;
    iterParms.nelem = volumeSize.x * volumeSize.y * volumeSize.z;

    /*
     * Create FFT plans.
     */
     // FFT plans for estimation
     cudaErrChk(cufftPlan3d(
         &iterParms.fftHandle.forward,
         volumeSize.z, volumeSize.y, volumeSize.x,
         CUFFT_R2C
     ));
     cudaErrChk(cufftPlan3d(
         &iterParms.fftHandle.reverse,
         volumeSize.z, volumeSize.y, volumeSize.x,
         CUFFT_C2R
     ));

     //TODO attach callback device functions

     /*
      * Estimate memory usage from FFT procedures.
      */

     /*
      * Allocate device staging area.
      */
      size_t realSize =
          volumeSize.x * volumeSize.y * volumeSize.z * sizeof(cufftReal);
      size_t complexSize =
          (volumeSize.x/2+1) * volumeSize.y * volumeSize.z * sizeof(cufftComplex);

     // template
     cudaErrChk(cudaMalloc((void **)&iterParms.raw, realSize));

     // IO buffer
     cudaErrChk(cudaMalloc((void **)&iterParms.ioBuffer.input, realSize));
     cudaErrChk(cudaMalloc((void **)&iterParms.ioBuffer.output, realSize));

     // FFT Buffer
     cudaErrChk(cudaMalloc((void **)&iterParms.filterBuffer.complexA, complexSize));

     // RL Buffer
     cudaErrChk(cudaMalloc((void **)&iterParms.RLBuffer.realA, realSize));

     // prediction buffer
     cudaErrChk(cudaMalloc((void **)&iterParms.predBuffer.prevIter, realSize));
     cudaErrChk(cudaMalloc((void **)&iterParms.predBuffer.prevPredChg, realSize));
}

//TODO scale output from float to uint16
void DeconvRL::process(
	ImageStack<float> &odata,
	const ImageStack<uint16_t> &idata
) {
    Core::Parameters &iterParms = pimpl->iterParms;
    const size_t nelem = iterParms.nelem;

    // register the input data memory region on host as pinned
    cudaErrChk(cudaHostRegister(
        idata.data(),
        nelem * sizeof(float),
        cudaHostRegisterMapped
    ));

    // retrieve the host pointer
    uint16_t *d_idata = nullptr;
    cudaErrChk(cudaHostGetDevicePointer(&d_idata, idata.data(), 0));

    /*
     * Copy the data to buffer area along with type casts.
     */
    fprintf(stderr, "[DBG] %ld elements to type cast\n", nelem);
    Common::ushort2float(
        iterParms.ioBuffer.input,   // output
        d_idata,                    // input
        nelem
    );

    // duplicate the to store a copy of raw data
    cudaErrChk(cudaMemcpy(
        iterParms.raw,
        iterParms.ioBuffer.input,
        nelem * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    /*
     * Release the pinned memory region.
     */
    cudaErrChk(cudaHostUnregister(idata.data()));

    /*
     * Execute the core functions.
     */
    const int nIter = pimpl->iterations;
    for (int iIter = 1; iIter <= nIter; iIter++) {
        //Core::RL::step(
        Core::Biggs::step(
            iterParms.ioBuffer.output,  // output
            iterParms.ioBuffer.input,   // input
            iterParms
        );
        // swap A, B buffer
        std::swap(iterParms.ioBuffer.input, iterParms.ioBuffer.output);

        fprintf(stderr, "[INF] %d/%d\n", iIter, nIter);
    }

    // swap back to avoid confusion
    std::swap(iterParms.ioBuffer.input, iterParms.ioBuffer.output);
    // copy back to host
    cudaErrChk(cudaMemcpy(
        odata.data(),
        iterParms.ioBuffer.output,
        nelem * sizeof(cufftReal),
        cudaMemcpyDeviceToHost
    ));
}

}
