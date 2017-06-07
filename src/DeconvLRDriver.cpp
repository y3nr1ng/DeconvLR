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
	if (nx > 4096 or ny > 4096 or nz > 4096) {
		throw std::range_error("volume size exceeds maximum constraints");
	}
	pimpl->volumeSize.x = nx;
	pimpl->volumeSize.y = ny;
	pimpl->volumeSize.z = nz;
}

void DeconvLR::setPSF(const ImageStack<uint16_t> &psf) {
	fprintf(stderr, "[DEBUG] --> setPSF()\n");

	uint16_t *hPsf;
	cudaExtent psfSize = make_cudaExtent(
		psf.nx() * sizeof(uint16_t),   // width in bytes
		psf.ny(),
		psf.nz()
	);

	// pin down the host memory
	cudaErrChk(cudaHostRegister(
        psf.data(),
        psfSize.width * psfSize.height * psfSize.depth,
        cudaHostRegisterMapped
   ));
	cudaErrChk(cudaHostGetDevicePointer(
        &hPsf, // device pointer for mapped address
        psf.data(), // requested host pointer
        0
    ));

	fprintf(stderr, "[DEBUG] host memory pinned\n");

	/*
	 * Convert from uint16_t to cufftReal
	 */
	cudaPitchedPtr dPsf;
	// update pitch to base on cufftReal
	psfSize.width = psf.nx() * sizeof(cufftReal);
	// create workspace for the PSF
	cudaErrChk(cudaMalloc3D(&dPsf, psfSize));

	Kernel::convertType<cufftReal, uint16_t>(
		(cufftReal *)dPsf.ptr,
		hPsf,
		psfSize
	);

	fprintf(stderr, "[DEBUG] type conversion completed\n");

	/*
	 * cuFFT R2C
	 */
	cudaPitchedPtr otfLut;
	cufftHandle otfFFTHandle;

	// create workspace for the OTF
	cudaExtent otfSize = make_cudaExtent(
		psf.nx() * sizeof(cufftComplex),   // width in bytes
		psf.ny(),
		std::floor((float)psf.nz() / 2) + 1
    );
	cudaErrChk(cudaMalloc3D(&otfLut, otfSize));

	fprintf(stderr, "[DEBUG] template OTF workspace created\n");

	// plan and execute FFT
	cudaErrChk(cufftPlan3d(
        &otfFFTHandle,
        otfSize.width, otfSize.height, otfSize.depth,
        CUFFT_R2C
    ));
	cudaErrChk(cufftExecR2C(
        otfFFTHandle,
        (cufftReal *)dPsf.ptr,      // input
        (cufftComplex *)otfLut.ptr  // output
    ));

	fprintf(stderr, "[DEBUG] OTF created\n");

	// release the cuFFT handle and integer PSF
	cudaErrChk(cufftDestroy(otfFFTHandle));
	cudaErrChk(cudaFree(dPsf.ptr));

	// convert to CUDA array
	cudaArray_t otfLutArray;
	cudaChannelFormatDesc formDesc = cudaCreateChannelDesc(
		32, 0, 0, 0,
		cudaChannelFormatKindFloat
	);

	cudaErrChk(cudaMalloc3DArray(
        &otfLutArray,
        &formDesc,
        otfSize,
        cudaArrayDefault
	));

	fprintf(stderr, "[DEBUG] OTF template cudaArray allocated\n");

	cudaMemcpy3DParms parms = {0};
	parms.srcPtr = otfLut;
	parms.dstArray = otfLutArray;
	parms.extent = otfSize;
	parms.kind = cudaMemcpyDeviceToDevice;
	cudaErrChk(cudaMemcpy3D(&parms));

	// release the template OTF
	cudaErrChk(cudaFree(otfLut.ptr));

	fprintf(stderr, "[DEBUG] OTF template copied, D2D\n");

	/*
	 * Interpolate to volume size
	 */
	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = otfLutArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	// out-of-border pixels are 0
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	// access by [0, 1]
	texDesc.readMode = cudaReadModeNormalizedFloat;

	// bind to texture object
	cudaTextureObject_t otfTex = 0;
	cudaErrChk(cudaCreateTextureObject(&otfTex, &resDesc, &texDesc, NULL));

	fprintf(stderr, "[DEBUG] template bind to a texture object\n");

	// update OTF size to full volume
	otfSize.width = pimpl->volumeSize.x * sizeof(cufftComplex);
	otfSize.height = pimpl->volumeSize.y;
	otfSize.depth = pimpl->volumeSize.z;
	// allocate OTF workspace
	cudaErrChk(cudaMalloc3D(&pimpl->otf, otfSize));

	fprintf(stderr, "[DEBUG] pimpl OTF workspace allocated\n");

	// interpolate
	//TODO interpolate otfTex using pimpl->voxelRatio onto pimpl->otf

	fprintf(stderr, "[DEBUG] interpolation completed\n");

	/*
	 * Cleanup
	 */
	// release the texture resources
	cudaErrChk(cudaDestroyTextureObject(otfTex));
	cudaErrChk(cudaFreeArray(otfLutArray));
	// unregister the host memory region
	cudaErrChk(cudaHostUnregister(psf.data()));

	fprintf(stderr, "[DEBUG] setPSF() -->\n");
}

void DeconvLR::process(
	ImageStack<uint16_t> &output,
	const ImageStack<uint16_t> &input
) {

}
