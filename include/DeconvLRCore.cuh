#ifndef DECONV_LR_CORE_CUH
#define DECONV_LR_CORE_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
// system headers

namespace PSF {

/*
 * Find center of the PSF data.
 */
float3 findCentroid(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
);

/*
 * Bind host PSF image data for further processing.
 */
void bindData(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
);

/*
 * Move centroid of the 3-D PSF to center of the volume.
 */
void alignCenter();

/*
 * Release the resources used by the PSF functions.
 */
void release();

}

namespace Kernel {

template <typename T_out, typename T_in>
__host__
void convertType(T_out *dst, T_in *src, const cudaExtent size);

}

#endif
