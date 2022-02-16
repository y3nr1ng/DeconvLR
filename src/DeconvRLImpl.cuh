#ifndef DECONV_LR_CORE_CUH
#define DECONV_LR_CORE_CUH

// corresponded header file
// necessary project headers
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cufft.h>
// standard libraries headers
#include <cstdint>
// system headers

namespace DeconvRL {

namespace PSF {

class PSF {
public:
    PSF(
        float *h_psf,
        const size_t npx, const size_t npy, const size_t npz = 1
    );
    ~PSF();

    /**
     * @brief Center the centroid of the provided PSF.
     *
     * The method calls upon the the estimateBackground method to estimate and
     * remove the potential bacground noises by its mean. Later, centroid of the
     * PSF is calculated and used to circular shift the original PSF to its
     * align with its center.
     *
     * @see findCentroid, estimateBackground
     */
    void alignCenter(const size_t nx, const size_t ny, const size_t nz = 1);

    /**
     * @brief Convert the PSF to OTF.
     *
     * Convert the PSF to an OTF by a FFT. Caller has to allocate the OTF
     * pointer with a proper memory space, (nx/2+1)*ny*nz*sizeof(cufftComplex).
     *
     * @param d_otf The converted OTF.
     * @param nx Number of elements in the X dimension (fastest variation).
     * @param ny Number of elements in the Y dimension.
     * @param nz Number of elements in the Z dimension (slowest variation).
     *
     * @see
     */
    void createOTF(cufftComplex *d_otf);

private:
    float3 findCentroid();
    float estimateBackground();
    void padPSF(const size_t nx, const size_t ny, const size_t nz);

    // PSF memory, host side and mirrored device address
    float *d_psf;

    // size of the PSF
    size_t npx, npy, npz;
    size_t nelem;
};

}

namespace Core {

/**
 * @brief Brief introduction to the function.
 *
 * Description of what the function does
 * @param PARAM1 Description of the first parameter of the function.
 * @return Describe what the function returns.
 * @see FUNCTION
 */

/**
 * Parameter class that holds all constant and temporary variables during the
 * Richardson-Lucy iteration steps.
 */
struct Parameters {
    //TODO destructor to free the memory region
    //TODO destructor to free the FFT handles

    /**
     * Dimension of the image in real space.
     */
    size_t nx, ny, nz;
    // product of nx, ny and nz
    size_t nelem;

    // original image
    float *raw;

    // OTF
    cufftComplex *otf;

    /**
     * cuFFT handles for forward (R2C) and reverse (C2R) FFT operations.
     */
    struct {
        cufftHandle forward;
        cufftHandle reverse;
    } fftHandle;

    /**
     * I/O buffer to interface with the host.
     */
    struct {
        cufftReal *input;
        cufftReal *output;
    } ioBuffer;

    /**
     * Intermediate buffers, maximum size is used, aka padded input data size.
     */
    struct {
        cufftComplex *complexA;
    } filterBuffer;

    struct {
        cufftReal *realA;
    } RLBuffer;

    /**
     * Prediction buffers
     */
    struct {
        float *prevIter;
        float *prevPredChg;
    } predBuffer;
};

namespace RL {

/**
 * @brief One iteration in the Richardson-Lucy algorithm.
 *
 * DESCRIPTION
 * @param odata Result from current iteration.
 * @param idata Result of previous iteration.
 * @param parm Algorithm related parameters.
 * @return
 * @see
 */
void step(
    float *odata, const float *idata,
    Core::Parameters &parms
);

}

namespace Biggs {

/**
 * @brief One iteration in the accelerated Richardson-Lucy algorithm.
 *
 * DESCRIPTION
 * @param odata Result from current iteration.
 * @param idata Result of previous iteration.
 * @param parm Algorithm related parameters.
 * @return
 * @see
 */
void step(
    float *odata, const float *idata,
    Core::Parameters &parms
);

}

}

namespace Common {

void ushort2float(float *odata, const uint16_t *idata, const size_t nelem);

}

}

#endif
