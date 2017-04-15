#ifndef DECONV_LR_DRIVER_HPP
#define DECONV_LR_DRIVER_HPP

#include <memory>

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
