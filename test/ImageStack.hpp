#ifndef IMAGE_STACK_HPP
#define IMAGE_STACK_HPP

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <tiffio.h>
// standard libraries headers
#include <cstdint>
#include <string>
// system headers

struct ImageStack {
    enum class Mode : uint8_t {
        READ = 1,
        WRITE
    };

    ImageStack();
    ImageStack(std::string path, Mode mode);
    ~ImageStack();

    void open(std::string path, Mode mode);
    void close();

    template <typename T>
    T * data() const;

private:
    void readStackInfo();
    void flush();

    TIFF *file;
    Mode mode;
    uint32_t width, height, nPages;
    uint8_t *buffer;
};

#endif
