// corresponded header file
#include "DeconvLRDriver.hpp"
// necessary project headers
#include "DeconvLRCore.cuh"
#include "Helper.cuh"
// 3rd party libraries headers
#include <cuda_runtime.h>
#include <cufft.h>
// standard libraries headers
#include <exception>
#include <cmath>
#include <cstdio>
// system headers

struct DeconvLR::Impl {
	Impl() {

	}
	~Impl() {

	}

	// volume size
	dim3 volumeSize;
	// voxel ratio = raw voxel size / PSF voxel size
	float3 voxelRatio;

	/*
	 * Device pointers
	 */
	cudaPitchedPtr otf;
};

// C++14 feature
template<typename T, typename ... Args>
std::unique_ptr<T> make_unique(Args&& ... args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
}

DeconvLR::DeconvLR()
	: pimpl(make_unique<Impl>()) {
}

DeconvLR::~DeconvLR() {

}

void DeconvLR::setResolution(
	const float dx, const float dy, const float dz,
	const float dpx, const float dpy, const float dpz
) {
	pimpl->voxelRatio = make_float3(dx/dpx, dy/dpy, dz/dpz);
}

void DeconvLR::setVolumeSize(
	const size_t nx, const size_t ny, const size_t nz
) {
	if (nx > 2048 or ny > 2048 or nz > 2048) {
		throw std::range_error("volume size exceeds maximum constraints");
	}
	pimpl->volumeSize.x = nx;
	pimpl->volumeSize.y = ny;
	pimpl->volumeSize.z = nz;
}

void DeconvLR::setPSF(const ImageStack<uint16_t> &psf_u16) {
	fprintf(stderr, "[DEBUG] --> setPSF()\n");

    /*
     * Convert to float
     */
    ImageStack<cufftReal> psf(psf_u16);
	fprintf(stderr, "[DEBUG] type conversion completed\n");

	cufftReal *hPsf;
	cudaExtent psfExtent = make_cudaExtent(
		psf.nx() * sizeof(cufftReal),   // width in bytes
		psf.ny(),
		psf.nz()
	);

	/*
     * Pin the host memory region
	 */
	cudaErrChk(cudaHostRegister(
        psf.data(),
        psfExtent.width * psfExtent.height * psfExtent.depth,
        cudaHostRegisterMapped
    ));
	cudaErrChk(cudaHostGetDevicePointer(
        &hPsf, 		// device pointer for mapped address
        psf.data(), // requested host pointer
        0
    ));
	fprintf(stderr, "[DEBUG] host memory pinned\n");

    /*
     * Create OTF
     */
    // allocate linear space for the template OTF
    cudaPitchedPtr otfTplLin;
    cudaExtent otfTplExtent = make_cudaExtent(
		psf.nx() * sizeof(cufftComplex),   // width in bytes
		psf.ny(),
		psf.nz()/2+1
	);
    cudaErrChk(cudaMalloc3D(&otfTplLin, otfTplExtent));
    fprintf(stderr, "[DEBUG] linear array for template OTF created\n");

    // plan and execute FFT
    cufftHandle otfFFTHandle;
	cudaErrChk(cufftPlan3d(
        &otfFFTHandle,
        otfTplExtent.width, otfTplExtent.height, otfTplExtent.depth,
        CUFFT_R2C
    ));
    fprintf(stderr, "[DEBUG] R2C FFT planned\n");
	cudaErrChk(cufftExecR2C(
        otfFFTHandle,
        hPsf,                           // input
        (cufftComplex *)otfTplLin.ptr   // output
    ));
    fprintf(stderr, "[DEBUG] OTF = FFT(PSF)\n");
    // wait for the FFT to finish
    cudaErrChk(cudaDeviceSynchronize());
    
    // unpin the PSF memory region
    cudaErrChk(cudaHostUnregister(psf.data()));

    /*
     * Copy to cudaArray
     */
    // allocate cudaArray for template OTF
    cudaArray_t otfTpl;
    // cufftComplex is essentially a float2
    cudaChannelFormatDesc formDesc = cudaCreateChannelDesc(
        32, 32, 0, 0,
        cudaChannelFormatKindFloat
    );
	// width field in elements
	otfTplExtent.width = psf.nx();
	cudaErrChk(cudaMalloc3DArray(
        &otfTpl,
        &formDesc,
        otfTplExtent,
        cudaArrayDefault
	));

    // copy 3-D data
    cudaMemcpy3DParms parms = {0};
    parms.srcPtr = otfTplLin;
    parms.dstArray = otfTpl;
    parms.extent = otfTplExtent;
    parms.kind = cudaMemcpyDeviceToDevice;
    cudaErrChk(cudaMemcpy3D(&parms));
    fprintf(stderr, "[DEBUG] prepare template OTF binding\n");

    // release the linear template OTF
    cudaErrChk(cudaFree(otfTplLin.ptr));

    /*
     * Interpolate the OTF
     */
    // call core routine
    fprintf(stderr, "[DEBUG] template OTF uploaded as texture\n");

    fprintf(stderr, "[DEBUG] interpolation completed\n");

    // free the template OTF
    fprintf(stderr, "[DEBUG] texture freed\n");

	fprintf(stderr, "[DEBUG] setPSF() -->\n");
}

void DeconvLR::process(
	ImageStack<uint16_t> &output,
	const ImageStack<uint16_t> &input
) {

}
