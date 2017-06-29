// corresponded header file
// necessary project headers
#include "ImageStack.hpp"
#include "DeconvRLDriver.hpp"
// 3rd party libraries headers
// standard libraries headers
#include <cstdint>
#include <iostream>
// system headers

int main(void)
{
    TIFFSetWarningHandler(NULL);

    std::string origImgFile = "data/cell/sample.tif";
    std::string psfFile = "data/cell/psf.tif";

    // scan the folder
    // search and load the otf
    ImageStack<uint16_t> psf(psfFile);
    // init the deconvlr
    DeconvRL::DeconvRL deconvWorker;
    deconvWorker.setResolution(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    // iterate through the images
    //      open the image
    const ImageStack<uint16_t> input(origImgFile);
    ImageStack<float> output(input, 0);
    //      use the first image to init the resources
    deconvWorker.setVolumeSize(input.nx(), input.ny(), input.nz());
    deconvWorker.setPSF(psf);
    deconvWorker.initialize();
    //      run the deconv
    deconvWorker.process(output, input);
    //      save the image
    output.saveAs("result.tif");
    // save the log
    // release the resources

    return 0;
}
