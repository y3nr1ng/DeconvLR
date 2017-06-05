// corresponded header file
#include "DeconvLRDriver.hpp"
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
// system headers

struct DeconvLR::Impl {
    Impl() {

    }
    ~Impl() {

    }

    // volume size
    float3 volumeSize;
    // voxel ratio = raw voxel size / PSF voxel size
    float3 voxelRatio;
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
    pimpl->volumeSize = make_float3(nx, ny, nz);
}

void DeconvLR::setPSF(const ImageStack<uint16_t> &psf) {
    // load the psf
    // estimate the ratio
    // resample
    // copy to texture memory
}

void DeconvLR::process(
    ImageStack<uint16_t> &output,
    const ImageStack<uint16_t> &input
) {

}
