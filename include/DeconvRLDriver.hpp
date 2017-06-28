#ifndef DECONV_RL_DRIVER_HPP
#define DECONV_RL_DRIVER_HPP

// corresponded header file
// necessary project headers
#include "ImageStack.hpp"
// 3rd party libraries headers
// standard libraries headers
#include <memory>
// system headers

namespace DeconvRL {

class DeconvRL {
public:
    DeconvRL();
    ~DeconvRL();

    void setResolution(
        const float dx, const float dy, const float dz,
        const float dpx = 1.0f, const float dpy = 1.0f, const float dpz = 1.0f
    );
    void setVolumeSize(const size_t nx, const size_t ny, const size_t nz);
    void setPSF(const ImageStack<uint16_t> &psf);

    // allocate host and device resources
    void initialize();
    // start the RL core routines
    void process(
        ImageStack<float> &output,
        const ImageStack<uint16_t> &input
    );

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}

#endif
