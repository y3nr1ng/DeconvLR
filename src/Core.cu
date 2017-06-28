// corresponded header file
// necessary project headers
#include "DeconvRLImpl.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cufft.h>
// standard libraries headers
#include <cstdint>
// system headers

namespace DeconvRL {

namespace Core {

namespace RL {

enum class ConvType {
    PLAIN = 1, CONJUGATE
};

namespace {
// generic complex number operation
template <ConvType type>
struct MultiplyAndScale
    : public thrust::binary_function<cuComplex, cuComplex, cuComplex> {
    MultiplyAndScale(const float c_)
        : c(c_) {
    }

    __host__ __device__
    cuComplex operator()(const cuComplex &a, const cuComplex &b) const {
        if (type == ConvType::CONJUGATE) {
            return cuCmulf(a, cuConjf(b))/c;
        } else {
            return cuCmulf(a, b)/c;
        }
    }

private:
    const float c;
};

template <ConvType type>
void filter(
    cufftReal *odata, const cufftReal *idata, const cufftComplex *otf,
    Core::RL::Parameters &parm
) {
    const size_t nelem = (parm.nx/2+1) * parm.ny * parm.nz;
    cufftComplex *buffer = (cufftComplex *)parm.filterBuffer.complexA;

    // convert to frequency space
    cudaErrChk(cufftExecR2C(
        parm.fftHandle.forward,
        const_cast<cufftReal *>(idata),
        buffer
    ));

    // element-wise multiplication and scale down
    thrust::transform(
        thrust::device,
        buffer, buffer+nelem,       // first input sequence
        otf,                        // second input sequence
        buffer,                     // output sequence
        MultiplyAndScale<type>(1.0f/nelem)
    );

    // convert back to real space
    cudaErrChk(cufftExecC2R(
        parm.fftHandle.reverse,
        buffer,
        odata
    ));
}

thrust::divides<float> DivfOp;
thrust::multiplies<float> MulfOp;

}

void step(
    float *odata, const float *idata,
    Core::RL::Parameters &parms
) {
    fprintf(stderr, "[DBG] +++ ENTER RL::step() +++\n");

    const size_t nelem = parms.nelem;
    cufftReal *buffer = parms.RLBuffer.realA;

    cufftComplex *otf = parms.otf;

    /*
     * \hat{f_{k+1}} =
     *     \hat{f_k} \left(
     *         h \ast \frac{g}{h \otimes \hat{f_k}}
     *     \right)
     */

    // reblur the image
    fprintf(stderr, " 1");
    filter<ConvType::PLAIN>(buffer, idata, otf, parms);
    fprintf(stderr, " 2");
    // error
    thrust::transform(
        thrust::device,
        parms.raw,  parms.raw+nelem,
        buffer,
        buffer, // output
        DivfOp
    );
    fprintf(stderr, " 3");
    filter<ConvType::PLAIN>(buffer, buffer, otf, parms);
    fprintf(stderr, " 4");
    // latent image
    thrust::transform(
        thrust::device,
        idata, idata+nelem,
        buffer,
        odata,  // output
        MulfOp
    );
    fprintf(stderr, " 5\n");

    fprintf(stderr, "[DBG] +++ EXIT RL::step() +++\n");
}

}

namespace Biggs {

namespace {

}

void step(
    float *odata, const float *idata,
    Core::RL::Parameters &parm
) {
    // execute an iteration of RL
    //RL::step();

    // find the update direction

    // calculate acceleration factor

    // re-estimate prediction
}

}

}

namespace Common {

namespace {

template <typename T>
struct ToFloat
    : public thrust::unary_function<const T, float> {
    __host__ __device__
    float operator()(const T &v) const {
        return (float)v;
    }
};

}

void ushort2float(float *odata, const uint16_t *idata, const size_t nelem) {
    thrust::transform(
        thrust::device,
        idata, idata + nelem,   // input
        odata,                  // output
        ToFloat<uint16_t>()
    );
}

}

}
