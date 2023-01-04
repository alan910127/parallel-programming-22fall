#define MAX_FILTER_SIZE 7
#define GROUP_SIZE 16
#define LOCAL_OFFSET (MAX_FILTER_SIZE / 2)
#define PADDED_SIZE (GROUP_SIZE + LOCAL_OFFSET * 2)

__kernel void convolution(int imageWidth, int imageHeight,
                          __global float* inputImage,
                          __global float* outputImage, int filterWidth,
                          __constant float* filter) {
  __local float localImage[PADDED_SIZE][PADDED_SIZE];

  int ix = get_global_id(0);
  int iy = get_global_id(1);

  if (ix >= imageWidth || iy >= imageHeight) return;

  int lx = get_local_id(0);
  int ly = get_local_id(1);

  int halfFilterSize = filterWidth / 2;

  // copy image from global memory to local memory
  for (int i = -halfFilterSize; i <= halfFilterSize; i += halfFilterSize) {
    for (int j = -halfFilterSize; j <= halfFilterSize; j += halfFilterSize) {
      int x = ix + i, y = iy + j;
      if (x < 0 || x >= imageWidth || y < 0 || y >= imageHeight) continue;

      int localX = lx + i + LOCAL_OFFSET, localY = ly + j + LOCAL_OFFSET;
      localImage[localX][localY] = inputImage[y * imageWidth + x];
    }
  }

  mem_fence(CLK_LOCAL_MEM_FENCE);

  int startX = -min(ix, halfFilterSize);
  int endX = min(imageWidth - ix - 1, halfFilterSize);
  int startY = -min(iy, halfFilterSize);
  int endY = min(imageHeight - iy - 1, halfFilterSize);

  float sum = 0.0f;
  for (int i = startX; i <= endX; ++i) {
    for (int j = startY; j <= endY; ++j) {
      int x = lx + i + LOCAL_OFFSET, y = ly + j + LOCAL_OFFSET;
      int fx = halfFilterSize + i, fy = halfFilterSize + j;
      sum += localImage[x][y] * filter[fy * filterWidth + fx];
    }
  }

  outputImage[iy * imageWidth + ix] = sum;
}
