// corresponded header file
// necessary project headers
#include "ImageStack.hpp"
#include "DeconvLRDriver.hpp"
// 3rd party libraries headers
// standard libraries headers
#include <cstdint>
#include <iostream>
// system headers

int main(void)
{
    TIFFSetWarningHandler(NULL);

    std::string origImgFile = "data/bigradient/bigradient_conv.tif";
    std::string otfFile = "data/bigradient/psf.tif";

    // scan the folder
    // search and load the otf
    ImageStack<float> otf(otfFile);
    otf.debug();
    // init the deconvlr
    DeconvLR deconWorker;
    // iterate through the images
    //      open the image
    ImageStack<uint16_t> origImg(origImgFile);
    //      use the first image to init the resources
    deconWorker.setOTF(otf);
    //      run the deconv
    //      save the image
    // save the log
    // release the resources

    return 0;
}
