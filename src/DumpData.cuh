#ifndef HELPER_DUMP_CUH
#define HELPER_DUMP_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cufft.h>
// standard libraries headers
#include <string>
// system headers

namespace DumpData {

namespace Device {

void real(
    std::string fname,
    const cufftReal *d_idata,
    const size_t nx, const size_t ny, const size_t nz
);

void complex(
    std::string fname,
    const cufftComplex *d_idata,
    const size_t nx, const size_t ny, const size_t nz
);

}

namespace Host {

void real(
    std::string fname,
    cufftReal *h_idata,
    const size_t nx, const size_t ny, const size_t nz
);

}

}

#endif
