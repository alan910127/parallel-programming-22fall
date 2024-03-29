#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__device__ int mandel(float c_re, float c_im, int count) {
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
                             float stepY, int maxIteration, size_t pitch,
                             int* dest) {
  // To avoid error caused by the floating number, use the following pseudo code
  //
  // float x = lowerX + thisX * stepX;
  // float y = lowerY + thisY * stepY;

  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
  int thisY = blockIdx.y * blockDim.y + threadIdx.y;

  float x = lowerX + thisX * stepX;
  float y = lowerY + thisY * stepY;

  int* row = (int*)((char*)dest + thisY * pitch);
  row[thisX] = mandel(x, y, maxIteration);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  int rowSize = sizeof(int) * resX;
  int sizeInBytes = rowSize * resY;

  int* deviceImg;
  size_t pitch;
  cudaMallocPitch((void**)&deviceImg, &pitch, rowSize, resY);

  dim3 blockShape(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridShape(resX / blockShape.x, resY / blockShape.y);
  mandelKernel<<<gridShape, blockShape>>>(lowerX, lowerY, stepX, stepY,
                                          maxIterations, pitch, deviceImg);

  cudaDeviceSynchronize();

  int* hostImg;
  cudaHostAlloc((void**)&hostImg, sizeInBytes, cudaHostAllocDefault);

  cudaMemcpy2D(hostImg, rowSize, deviceImg, pitch, rowSize, resY,
               cudaMemcpyDeviceToHost);
  memcpy(img, hostImg, sizeInBytes);

  cudaFreeHost(hostImg);
  cudaFree(deviceImg);
}
