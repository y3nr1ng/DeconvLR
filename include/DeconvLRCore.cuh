#ifndef DECONV_LR_CORE_CUH
#define DECONV_LR_CORE_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
// standard libraries headers
// system headers

namespace Kernel {

// place the original psf data into texture memory for interpolation
void uploadRawPSF(
    const uint16_t *hPsf,
    const size_t nx, const size_t ny, const size_t nz
);

}

#endif
