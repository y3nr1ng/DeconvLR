#ifndef DECONV_LR_DRIVER_HPP
#define DECONV_LR_DRIVER_HPP

// corresponded header file
// necessary project headers
// 3rd party libraries headers
// standard libraries headers
#include <memory>
// system headers

class DeconvLR {
public:
    DeconvLR();
    ~DeconvLR();

    // update resource allocation
    void updatePlans();

    // upload the image to device
    void upload();
    // run the deconvolution procedure
    void run();
    // download the image from device
    void download();

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

void square_calc_demo(void);

#endif
