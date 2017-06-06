// corresponded header file
// necessary project headers
#include "DeconvLRCore.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
// system headers

namespace Kernel {

inline int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

}
