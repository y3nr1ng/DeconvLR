#ifndef IMAGE_STACK_HPP
#define IMAGE_STACK_HPP

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>

#define cimg_use_tiff
#include "CImg.h"
using namespace cimg_library;
// standard libraries headers
#include <string>
#include <exception>
#include <iostream>
// system headers

namespace fs = boost::filesystem;

template <typename T>
class ImageStack {
public:
    ImageStack(const fs::path path_)
        : path(path_) {
        try {
            image.assign(path.c_str());
        } catch(CImgIOException &err) {
            throw std::runtime_error("unable to open image");
        }
    }

    void debug() {
        std::cout << "file: " << path << std::endl;
        image.display();
    }

    T * data() const {
        return image.data();
    }

    size_t nx() const {
        return image._width;
    }

    size_t ny() const {
        return image._height;
    }

    size_t nz() const {
        return image._depth;
    }

private:
    const fs::path path;
    CImg<T> image;
};

#endif
