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

	}

	// volume size
	dim3 volumeSize;
	// voxel ratio = raw voxel size / PSF voxel size
	float3 voxelRatio;

	/*
	 * Device pointers
	 */
	cudaPitchedPtr otf;
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

    // type conversion to ensure we are working with float
    ImageStack<float> psf(psf_u16);

    // upload PSF to device
    PSF::bindData(
        h_psf.data(),
        psf.nx(), psf.ny(), psf.nz()
    );

    PSF::release();

	fprintf(stderr, "[DEBUG] setPSF() -->\n");
}

void centerPSF(float *d_psf, const float *h_psf) {

    // find center of the PSF
    // bind uncentered PSF to texture
    // retrieve centered PSF
}

void createOTFTexture() {

}

void DeconvLR::process(
	ImageStack<uint16_t> &output,
	const ImageStack<uint16_t> &input
) {

}
