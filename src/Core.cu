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
#include <thrust/inner_product.h>
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
struct MultiplyAndScale
    : public thrust::binary_function<cuComplex, cuComplex, cuComplex> {
    MultiplyAndScale(const float c_)
        : c(c_) {
    }

    __host__ __device__
    cuComplex operator()(const cuComplex &a, const cuComplex &b) const {
        return cuCmulf(a, b)/c;
    }

private:
    const float c;
};

void filter(
    cufftReal *odata, const cufftReal *idata, const cufftComplex *otf,
    Core::Parameters &parm
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
        MultiplyAndScale(1.0f/parm.nelem)
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
    Core::Parameters &parms
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
    filter(buffer, idata, otf, parms);
    // error
    thrust::transform(
        thrust::device,
        parms.raw,  parms.raw+nelem,
        buffer,
        buffer, // output
        DivfOp
    );
    filter(buffer, buffer, otf, parms);
    // latent image
    thrust::transform(
        thrust::device,
        idata, idata+nelem,
        buffer,
        odata,  // output
        MulfOp
    );

    fprintf(stderr, "[DBG] +++ EXIT RL::step() +++\n");
}

}

namespace Biggs {

namespace {

struct ScaleAndAdd
    : public thrust::binary_function<float, float, float> {
    ScaleAndAdd(const float alpha_)
        : alpha(alpha_) {
    }

    __host__ __device__
    float operator()(const float &a, const float &b) const {
        // apply positivity constraint after SAXPY
        //return fmaxf(a + alpha*b, 0.0f);
        return a + alpha*b;
    }

private:
    const float alpha;
};

}

void step(
    float *odata, const float *idata,
    Core::Parameters &parm
) {
    // borrow space from odata, rename to avoid confusion
    float* iter = odata;
    // calcualte x_k
    RL::step(iter, idata, parm);

    // extract the definition
    float *prevIter = parm.predBuffer.prevIter;
    float *prevPredChg = parm.predBuffer.prevPredChg;

    // updateDir borrow buffer from prevIter
    float* updateDir = prevIter;
    // h_k in the paper
    // update_direction = prev_iter - iter;
    thrust::transform(
        thrust::device,
        iter, iter+parm.nelem,
        prevIter,
        updateDir,
        thrust::minus<float>()
    );

    // reuse space of idata
    float *predChg = const_cast<float *>(idata);
    // calculate g_{k - 1} = x_k - y_{k - 1}.
    // pred_change = iter - prev_pred;
    thrust::transform(
        thrust::device,
        iter, iter+parm.nelem,
        idata,
        predChg,
        thrust::minus<float>()
    );

    // calculate alpha (acceleration factor).
    float den = thrust::inner_product(
        thrust::device,
        predChg, predChg+parm.nelem,
        prevPredChg,
        0
    );
    float nom = (
        thrust::inner_product(
            thrust::device,
            prevPredChg, prevPredChg+parm.nelem,
            prevPredChg,
            0
        ) + std::numeric_limits<float>::epsilon()
    );
    float alpha = den / nom;
    fprintf(stderr, "[DBG] fraction [%f/%f = %f]\n", den, nom, alpha);

    // stability enforcement
    alpha = std::max(std::min(alpha, 1.0f), 0.0f);
    fprintf(stderr, "[INF] alpha = %f\n", alpha);

    // save current predictions
    cudaErrChk(cudaMemcpy(
        prevIter,
        iter,
        parm.nelem * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));
    cudaErrChk(cudaMemcpy(
        prevPredChg,
        predChg,
        parm.nelem * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    // calculate y_k
    // odata = iter + alpha * update_direction;
    thrust::transform(
        thrust::device,
        iter, iter+parm.nelem,
        updateDir,
        odata,
        ScaleAndAdd(alpha)
    );
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
