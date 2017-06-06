#ifndef HELPER_CUH
#define HELPER_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cufft.h>
// standard libraries headers
#include <cstdio>
#include <cstdlib>
// system headers

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(
    cudaError_t code,
    const char *file, int line,
    bool abort=true
) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Runtime: %s\n.. %s ln%d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

inline void cudaAssert(
    cufftResult_t code,
    const char *file, int line,
    bool abort=true
) {
    if (code != CUFFT_SUCCESS) {
        fprintf(stderr,"CUDA FFT: %d %s %d\n", code, file, line);
        if (abort) {
            exit(code);
        }
    }
}

#endif
