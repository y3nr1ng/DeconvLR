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
        if (d_otf.ptr) {
            cudaErrChk(cudaFree(d_otf.ptr));
        }
	}

	// volume size
	dim3 volumeSize;
	// voxel ratio = raw voxel size / PSF voxel size
	float3 voxelRatio;

	/*
	 * Device pointers
	 */
	cudaPitchedPtr d_otf = {0};
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
	pimpl->voxelRatio = make_float3(dx/dpx, dy/dpy, dz/dpz);
}

void DeconvLR::setVolumeSize(
	const size_t nx, const size_t ny, const size_t nz
) {
	if (nx > 2048 or ny > 2048 or nz > 2048) {
		throw std::range_error("volume size exceeds maximum constraints");
	}
	pimpl->volumeSize.x = nx;
	pimpl->volumeSize.y = ny;
	pimpl->volumeSize.z = nz;
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

    /*
     * Align the PSF to center.
     */
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

    /*
     * Generate OTF texture.
     */
    OTF::fromPSF(
        psf.data(),
        psf.nx(), psf.ny(), psf.nz()
    );
    fprintf(stderr, "[DEBUG] template OTF generated\n");

    CImg<float> dump(psf.nx()/2+1, psf.ny(), psf.nz());
    OTF::dumpTemplate(dump.data(), dump.width(), dump.height(), dump.depth());
    dump.save_tiff("dump.tif");

    OTF::interpolate(
        pimpl->d_otf,
        pimpl->volumeSize.x, pimpl->volumeSize.y, pimpl->volumeSize.z,
        psf.nx(), psf.ny(), psf.nz(),
        pimpl->voxelRatio.x, pimpl->voxelRatio.y, pimpl->voxelRatio.z
    );
    OTF::release();

	fprintf(stderr, "[DEBUG] setPSF() -->\n");
}

void DeconvLR::process(
	ImageStack<uint16_t> &output,
	const ImageStack<uint16_t> &input
) {

}
