// corresponded header file
#include "DeconvLRDriver.hpp"
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
// system headers

class DeconvLR::Impl {
public:
    Impl() {

    }
    ~Impl() {

    }

private:
    DeconvLRCore device;
};

DeconvLR::DeconvLR()
    : pimpl(std::make_unique<Impl>()) {
}

DeconvLR::~DeconvLR() {

}

void DeconvLR::setOTF(
    const ImageStack<float> &otf,
    const float dr, const float dz
) {
    
}
