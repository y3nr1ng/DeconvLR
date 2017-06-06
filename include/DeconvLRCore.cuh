#ifndef DECONV_LR_CORE_CUH
#define DECONV_LR_CORE_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
// system headers

namespace Kernel {

template <typename T_out, typename T_in>
__host__
void convertType(T_out *dst, T_in *src, const cudaExtent size);

}

#endif
