#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 4

__device__ int mandel(float c_re, float c_im, int count) {
  if (c_re * c_re + c_im * c_im <= .25f * .25f) {
    return count;
  }

  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f) break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX,
                             float stepY, int maxIteration, int resX,
                             int* dest) {
  // To avoid error caused by the floating number, use the following pseudo code
  //
  // float x = lowerX + thisX * stepX;
  // float y = lowerY + thisY * stepY;

  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
  int thisY = blockIdx.y * blockDim.y + threadIdx.y;

  float x = lowerX + thisX * stepX;
  float y = lowerY + thisY * stepY;

  int index = thisY * resX + thisX;
  dest[index] = mandel(x, y, maxIteration);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  int sizeInBytes = sizeof(int) * resX * resY;

  int* deviceImg;
  cudaHostRegister(img, sizeInBytes, cudaHostRegisterMapped);
  cudaHostGetDevicePointer((void**)&deviceImg, img, 0);

  dim3 blockShape(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridShape(qresX / blockShape.x, resY / blockShape.y);
  mandelKernel<<<gridShape, blockShape>>>(lowerX, lowerY, stepX, stepY,
                                          maxIterations, resX, deviceImg);

  cudaDeviceSynchronize();

  cudaHostUnregister(img);

  cudaMemcpy(img, deviceImg, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceImg);
}
