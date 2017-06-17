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
#include <cstdio>
// system headers

struct DeconvLR::Impl {
    Impl() {

    }

    ~Impl() {
        if (d_otf != nullptr) {
            cudaErrChk(cudaFree(d_otf));
        }
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

    /*
     * Device pointers
     */
    cufftComplex *d_otf = nullptr;

    /*
     * Handles
     */
    cufftHandle est_fft = 0;
    cufftHandle est_ifft = 0;
    cufftHandle err_fft = 0;
    cufftHandle err_ifft = 0;

    Core::RL::Parameters iterParms;
};

// C++14 feature
template<typename T, typename ... Args>
std::unique_ptr<T> make_unique(Args&& ... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
}

DeconvLR::DeconvLR()
    : pimpl(make_unique<Impl>()) {
}

DeconvLR::~DeconvLR() {

}

void DeconvLR::setResolution(
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

void DeconvLR::setVolumeSize(
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
        "[INFO] volume size = %ldx%ldx%ld\n",
        pimpl->volumeSize.x, pimpl->volumeSize.y, pimpl->volumeSize.z
    );
}

/*
 * ===========
 * PSF AND OTF
 * ===========
 */
void DeconvLR::setPSF(const ImageStack<uint16_t> &psf_u16) {
    fprintf(stderr, "[DEBUG] --> setPSF()\n");

    /*
     * Ensure we are working with floating points.
     */
    ImageStack<float> psf(psf_u16);
    fprintf(
        stderr,
        "[INFO] PSF size = %ldx%ldx%ld\n",
        psf.nx(), psf.ny(), psf.nz()
    );

    /*
     * Align the PSF to center.
     */
    PSF::removeBackground(
        psf.data(),
        psf.nx(), psf.ny(), psf.nz()
    );
    float3 centroid = PSF::findCentroid(
        psf.data(),
        psf.nx(), psf.ny(), psf.nz()
    );
    fprintf(
        stderr,
        "[INFO] centroid = (%.2f, %.2f, %.2f)\n",
        centroid.x, centroid.y, centroid.z
    );

    /*
     * Shift the PSF around the centroid.
     */
    PSF::bindData(
        psf.data(),
        psf.nx(), psf.ny(), psf.nz()
    );
    PSF::alignCenter(
        psf.data(),
        psf.nx(), psf.ny(), psf.nz(),
        centroid
    );
    fprintf(stderr, "[DEBUG] PSF aligned to center\n");
    PSF::release();

    psf.saveAs("psf_aligned.tif");

    /*
     * Generate OTF texture.
     */
    OTF::fromPSF(
        psf.data(),
        psf.nx(), psf.ny(), psf.nz()
    );
    fprintf(stderr, "[DEBUG] template OTF generated\n");

    CImg<float> otfTpl(psf.nx()/2+1, psf.ny(), psf.nz());
    OTF::dumpTemplate(
        otfTpl.data(),
        otfTpl.width(), otfTpl.height(), otfTpl.depth()
    );
    otfTpl.save_tiff("otf_template.tif");

    // allocate OTF memory
    cudaErrChk(cudaMalloc(
        &pimpl->d_otf,
        (pimpl->volumeSize.x/2+1) * pimpl->volumeSize.y * pimpl->volumeSize.z * sizeof(cufftComplex)
    ));
    // start the interpolation
    OTF::interpolate(
        pimpl->d_otf,
        pimpl->volumeSize.x/2+1, pimpl->volumeSize.y, pimpl->volumeSize.z,
        psf.nx()/2+1, psf.ny(), psf.nz(),
        pimpl->voxelSize.raw.x, pimpl->voxelSize.raw.y, pimpl->voxelSize.raw.z,
        pimpl->voxelSize.psf.x, pimpl->voxelSize.psf.y, pimpl->voxelSize.psf.z
    );
    OTF::release();
    fprintf(stderr, "[INFO] OTF established\n");

    CImg<float> otfCalc(pimpl->volumeSize.x/2+1, pimpl->volumeSize.y, pimpl->volumeSize.z);
    OTF::dumpComplex(
        otfCalc.data(),
        pimpl->d_otf,
        otfCalc.width(), otfCalc.height(), otfCalc.depth()
    );
    otfCalc.save_tiff("otf_interp.tif");

	fprintf(stderr, "[DEBUG] setPSF() -->\n");
}

void DeconvLR::initialize() {
    dim3 volumeSize = pimpl->volumeSize;

    /*
     * Create FFT plans if not exists.
     */
     // FFT plans for estimation
     cudaErrChk(cufftPlan3d(
         &pimpl->est_fft,
         volumeSize.z, volumeSize.y, volumeSize.x,
         CUFFT_R2C
     ));
     cudaErrChk(cufftPlan3d(
         &pimpl->est_ifft,
         volumeSize.z, volumeSize.y, volumeSize.x,
         CUFFT_C2R
     ));
     // FFT plans for error
     cudaErrChk(cufftPlan3d(
         &pimpl->err_fft,
         volumeSize.z, volumeSize.y, volumeSize.x,
         CUFFT_R2C
     ));
     cudaErrChk(cufftPlan3d(
         &pimpl->err_ifft,
         volumeSize.z, volumeSize.y, volumeSize.x,
         CUFFT_C2R
     ));

     //TODO attach callback device functions

     /*
      * Estimate memory usage from FFT procedures.
      */

     /*
      * Allocate host and device staging area.
      */
}

void DeconvLR::process(
	ImageStack<uint16_t> &odata_u16,
	const ImageStack<uint16_t> &idata_u16
) {
    /*
     * Ensure we are working with floating points.
     */
    ImageStack<float> idata(idata_u16);

    /*
     * Execute the core functions.
     */
    const int nIter = pimpl->iterations;
    for (int iIter = 0: i < nIter; i++) {
        Core::Biggs::step();
    }
    // copy back the data

}
