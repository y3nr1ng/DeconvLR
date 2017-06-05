#ifndef DECONV_LR_DRIVER_HPP
#define DECONV_LR_DRIVER_HPP

// corresponded header file
// necessary project headers
#include "ImageStack.hpp"
// 3rd party libraries headers
// standard libraries headers
#include <memory>
// system headers

class DeconvLR {
public:
    DeconvLR();
    ~DeconvLR();

    // update resource allocation
    void setOTF(const ImageStack<float> &otf,
                const float dr = 1.0f, const float dz = 1.0f);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

#endif
