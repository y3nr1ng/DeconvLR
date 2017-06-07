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
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
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
