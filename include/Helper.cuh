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

/*
 * CUDA Runtime
 */
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


/*
 * cuFFT
 */
static const char * cufftGetErrorString(cufftResult error) {
	switch (error) {
        case CUFFT_SUCCESS:
            return "the cuFFT operation was successful";

        case CUFFT_INVALID_PLAN:
            return "cuFFT was passed an invalid plan handle";

        case CUFFT_ALLOC_FAILED:
            return "cuFFT failed to allocate GPU or CPU memory";

        case CUFFT_INVALID_VALUE:
            return "user specified an invalid pointer or parameter";

        case CUFFT_INTERNAL_ERROR:
            return "driver or internal cuFFT library error";

        case CUFFT_EXEC_FAILED:
            return "failed to execute an FFT on the GPU";

        case CUFFT_SETUP_FAILED:
            return "the cuFFT library failed to initialize";

        case CUFFT_INVALID_SIZE:
            return "user specified an invalid transform size";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "missing parameters in call";

        case CUFFT_INVALID_DEVICE:
            return "execution of a plan was on different GPU than plan creation";

        case CUFFT_PARSE_ERROR:
            return "internal plan database error";

        case CUFFT_NO_WORKSPACE:
            return "no workspace has been provided prior to plan execution";

        case CUFFT_NOT_IMPLEMENTED:
            return "function does not implement functionality for parameters given";

        case CUFFT_NOT_SUPPORTED:
            return "operation is not supported for parameters given";

        default:
            return "<unknown>";
    }
}
inline void cudaAssert(
	cufftResult_t code,
	const char *file, int line,
	bool abort=true
	) {
	if (code != CUFFT_SUCCESS) {
		fprintf(stderr,"cuFFT: %s\n.. %s ln%d\n", cufftGetErrorString(code), file, line);
		if (abort) {
			exit(code);
		}
	}
}

#endif
