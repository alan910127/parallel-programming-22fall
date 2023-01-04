#include <CL/cl.h>
#include <cuda.h>

#define MAX_FILTER_SIZE 7
#define BLOCK_SIZE 16
#define LOCAL_OFFSET (MAX_FILTER_SIZE / 2)
#define PADDED_SIZE (BLOCK_SIZE + LOCAL_OFFSET * 2)

static inline int roundUpDiv(int number, int base) {
  return ((number + base - 1) / base);
}

__constant__ float filter[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

__global__ void convolution(int imageWidth, int imageHeight, float *inputImage,
                            float *outputImage, int filterWidth) {
  __shared__ float sharedImage[PADDED_SIZE][PADDED_SIZE];

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ix >= imageWidth || iy >= imageHeight) return;

  int lx = threadIdx.x;
  int ly = threadIdx.y;

  int halfFilterSize = filterWidth / 2;

  // copy image from global memory to local memory
  for (int i = -halfFilterSize; i <= halfFilterSize; i += halfFilterSize) {
    for (int j = -halfFilterSize; j <= halfFilterSize; j += halfFilterSize) {
      int x = ix + i, y = iy + j;
      if (x < 0 || x >= imageWidth || y < 0 || y >= imageHeight) continue;

      int localX = lx + i + LOCAL_OFFSET, localY = ly + j + LOCAL_OFFSET;
      sharedImage[localX][localY] = inputImage[y * imageWidth + x];
    }
  }

  __threadfence_block();

  int startX = -min(ix, halfFilterSize);
  int endX = min(imageWidth - ix - 1, halfFilterSize);
  int startY = -min(iy, halfFilterSize);
  int endY = min(imageHeight - iy - 1, halfFilterSize);

  float sum = 0.0f;
  for (int i = startX; i <= endX; ++i) {
    for (int j = startY; j <= endY; ++j) {
      int x = lx + i + LOCAL_OFFSET, y = ly + j + LOCAL_OFFSET;
      int fx = halfFilterSize + i, fy = halfFilterSize + j;
      sum += sharedImage[x][y] * filter[fy * filterWidth + fx];
    }
  }

  outputImage[iy * imageWidth + ix] = sum;
}

void hostFE(int filterWidth, float *hostFilter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *, cl_context *,
            cl_program *) {
  int imageSize = imageWidth * imageHeight * sizeof(float);
  int filterSize = filterWidth * filterWidth * sizeof(float);

  float *deviceInputImage;
  cudaMalloc((void **)&deviceInputImage, imageSize);
  float *deviceOutputImage;
  cudaMalloc((void **)&deviceOutputImage, imageSize);

  cudaMemcpy(deviceInputImage, inputImage, imageSize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(filter, hostFilter, filterSize);

  dim3 blockShape(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridShape(roundUpDiv(imageWidth, blockShape.x),
                 roundUpDiv(imageHeight, blockShape.y));

  convolution<<<gridShape, blockShape>>>(imageWidth, imageHeight,
                                         deviceInputImage, deviceOutputImage,
                                         filterWidth);

  cudaDeviceSynchronize();

  cudaMemcpy(outputImage, deviceOutputImage, imageSize, cudaMemcpyDeviceToHost);

  // cudaFree(deviceInputImage);
  // cudaFree(deviceOutputImage);
}