// corresponded header file
// necessary project headers
#include "DeconvLRCore.h"
// 3rd party libraries headers
#include <cuda_runtime.h>
// standard libraries headers
#include <iostream>
// system headers

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(void)
{
  float *a_h, *a_d;  // Pointer to host & device arrays
  const int N = 10;  // Number of elements in arrays
  size_t size = N * sizeof(float);
  a_h = (float *)malloc(size);        // Allocate array on host

  cudaMalloc((void **) &a_d, size);   // Allocate array on device
  // Initialize host array and copy it to CUDA device
  for (int i=0; i<N; i++) {
      a_h[i] = (float)i;
  }
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

  // Do calculation on device:
  square_array(a_d, N);

  // Retrieve result from device and store it in host array
  gpuErrchk(cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost));
  // Print results
  for (int i=0; i<N; i++) {
      printf("%d %f\n", i, a_h[i]);
  }

  // Cleanup
  free(a_h);
  gpuErrchk(cudaFree(a_d));
}
