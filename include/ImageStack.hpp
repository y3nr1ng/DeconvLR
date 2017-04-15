#ifndef IMAGE_STACK_HPP
#define IMAGE_STACK_HPP

#include <cstdint>

struct ImageStack {
    ImageStack();
    ~ImageStack();

    uint16_t *data;

private:
    void open();
    void close();

    void read();
    void flush();
};

#endif
