# DeconvLR
DeconvLR is a open source CUDA implementation of accelerated Richard-Lucy Deconvolution algorithm regularized with total variation loss. This library is developed to recovered blurred image due to the spreading of point source in optical system. As far as we know, there is no other fully functional open source GPU accelerated implementation. This project is aim to develope an open source, high efficient library to process high resolution images of high quality.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
You need the following packages to get started.

***nix**
```
make
g++ <= 5
CMake >= 3.6
Boost >= 1.59
CUDA >= 8.0
```

**Windows**

**TODO** I haven't exactly tested this on Windows. DLL export symbols are needed in the public header.

### Build
1. Please clone this repository
   ```bash
   git clone https://github.com/liuyenting/DeconvLR.git
   ```
   or download and extract the tarball from [release page](https://github.com/liuyenting/DeconvLR/releases).
   ```bash
   tar zxvf DeconRL.tar.gz
   ```
2. Go to source directory and create a new build output directory. 
   ```bash
   cd DeconvLR
   mkdir build
   ```
3. We use `cmake` to do the heavy lifting.
   ```bash
   cd build
   cmake ..
   ```
   if everything runs smoothly, we can proceed with
   ```bash
   make
   ```

## Running the demo
**TODO** Explain how to run the demo.

Asides from the demo, this library is intended to use as 
```c++
std::string origImgFile = "data/bigradient/sample.tif";
std::string psfFile = "data/bigradient/psf_n15_z5.tif";

// load psf
ImageStack psf(psfFile);

// init the deconvlr
DeconvLR deconvWorker;
deconvWorker.setResolution(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

// open the image
const ImageStack<uint16_t> input(origImgFile);
ImageStack<uint16_t> output(input, 0);

// use the first image to init the resources
deconvWorker.setVolumeSize(input.nx(), input.ny(), input.nz());
deconvWorker.setPSF(psf);

// run the deconv
deconvWorker.process(output, input);
```

## Benchmark
**TODO** move benchmakr images from gh-page (in docs folder) to here.

## Authors
* **Liu, Yen-Ting** - *Initial work* - [liuyenting](https://github.com/liuyenting/)
* **Chiang, Tin-Ray** - *Initial work* - [CTinRay](https://github.com/CTinRay)

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details

## References
* William Hadley Richardson (1972), "Bayesian-Based Iterative Method of Image Restoration*," J. Opt. Soc. Am. 62, 55-59.
* Lucy, L. B. (1974). "An iterative technique for the rectification of observed distributions". Astronomical Journal. 79 (6): 745–754.
* Biggs, D. S., & Andrews, M. (1997). Acceleration of iterative image restoration algorithms. Applied optics, 36(8), 1766-1775.
* Dey, N., Blanc-Féraud, L., Zimmer, C., Roux, P., Kam, Z., Olivo-Marin, J. C., & Zerubia, J. (2004). 3D microscopy deconvolution using Richardson-Lucy algorithm with total variation regularization (Doctoral dissertation, INRIA).
