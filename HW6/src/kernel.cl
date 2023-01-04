__kernel void convolution(int imageWidth, int imageHeight,
                          __global float* inputImage,
                          __global float* outputImage, int filterWidth,
                          __constant float* filter) {
  int ix = get_global_id(0);
  int iy = get_global_id(1);

  int halfFilterSize = filterWidth / 2;

  float sum = 0.0f;
  for (int i = -halfFilterSize; i <= halfFilterSize; ++i) {
    for (int j = -halfFilterSize; j <= halfFilterSize; ++j) {
      int x = ix + i, y = iy + j;
      int fx = i + halfFilterSize, fy = j + halfFilterSize;
      if (x < 0 || x >= imageWidth) continue;
      if (y < 0 || y >= imageHeight) continue;
      sum += inputImage[y * imageWidth + x] * filter[fy * filterWidth + fx];
    }
  }

  outputImage[iy * imageWidth + ix] = sum;
}
