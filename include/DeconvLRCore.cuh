#ifndef DECONV_LR_CORE_CUH
#define DECONV_LR_CORE_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cufft.h>
// standard libraries headers
// system headers

namespace PSF {

/*
 * Remove constant background noise.
 */
 void removeBackground(
     float *h_psf,
     const size_t nx, const size_t ny, const size_t nz
);

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
 void alignCenter(
     float *h_psf,
     const size_t nx, const size_t ny, const size_t nz,
     const float3 centroid
 );

/*
 * Release the resources used by the PSF functions.
 */
void release();

}

namespace OTF {

void fromPSF(
    float *h_psf,
    const size_t nx, const size_t ny, const size_t nz
);

void interpolate(
    cufftComplex *d_otf,
    const size_t nx, const size_t ny, const size_t nz,      // full size
    const size_t ntx, const size_t nty, const size_t ntz,   // template size
    const float dx, const float dy, const float dz,
    const float dtx, const float dty, const float dtz
);

void release();

void dumpTemplate(
    float *h_otf,
    const size_t nx, const size_t ny, const size_t nz
);

void dumpComplex(
    float *h_odata,
    const cufftComplex *d_idata,
    const size_t nx, const size_t ny, const size_t nz
);

}

namespace Core {

// Note: buffers must be able to handle in-place FFT transform
union InPlaceType {
    cufftReal *real;
    cufftComplex *complex;
};

namespace RL {

/**
 * Parameter class that holds all constant and temporary variables during the
 * Richardson-Lucy iteration steps.
 */
struct Parameters {
    /**
     *  The original image.
     */
    float *raw;

    /**
     * Converted OTF, not conjugated.
     */
    float *otf;

    /**
     * Dimension of the image in real space.
     */
    size_t nx, ny, nz;

    /**
     * cuFFT handles for forward (R2C) and reverse (C2R) FFT operations.
     */
    struct {
        cufftHandle forward;
        cufftHandle reverse;
    } fftHandle;

    /**
     * Intermediate buffers, maximum size is used - whether it is the padded
     * cufftComplex array or the full-sized cufftReal array.
     */
    InPlaceType bufferA, bufferB;
};

}

namespace Biggs {

}

}

#endif
