// corresponded header file
// necessary project headers
#include "DeconvLRDriver.hpp"
#include "ImageStack.hpp"
// 3rd party libraries headers
// standard libraries headers
#include <iostream>
// system headers

int main(void)
{
    TIFFSetWarningHandler(NULL);
    std::string folder = "data/04132017_cell2_zp5um_F-tractin-mCh-GCamp6s";

    // scan the folder
    // search and load the otf
    // init the deconvlr
    // iterate through the images
    //      open the image
    //      run the deconv
    //      save the image
    // save the log
    // release the resources

    square_calc_demo();

    ImageStack im;

    return 0;
}
