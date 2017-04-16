// corresponded header file
#include "ImageStack.hpp"
// necessary project headers
// 3rd party libraries headers
#define cimg_use_tiff
#include "CImg.h"
using namespace cimg_library;
// standard libraries headers
// system headers

ImageStack::ImageStack()
    : file(NULL), buffer(NULL) {
}

ImageStack::ImageStack(std::string path, Mode mode)
    : ImageStack() {
    open(path, mode);
}

ImageStack::~ImageStack() {
    if (file) {
        TIFFClose(file);
    }
    if (buffer) {
        // free the memory
    }
}

void ImageStack::open(std::string path, Mode _mode) {
    // close before reopen

    if (mode == Mode::WRITE) {
        file = TIFFOpen(path.c_str(), "w");
    } else if (mode == Mode::READ) {
        file = TIFFOpen(path.c_str(), "r");
    }
}

void ImageStack::close() {
    if (mode == Mode::WRITE) {
        flush();
    } else {

    }
}

template <typename T>
T * ImageStack::data() const {
    return (T *)buffer;
}

void ImageStack::readStackInfo() {

}

void ImageStack::flush() {
    // fluh the memory contents to file
}
