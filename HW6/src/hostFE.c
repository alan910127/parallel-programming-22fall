#include "hostFE.h"

#include <stdio.h>
#include <stdlib.h>

#include "helper.h"

#define GROUP_SIZE 16

#if 1
#undef CHECK
#define CHECK(status, cmd) /* skip */
#endif

static inline int roundUp(int number, int base) {
  return ((number + base - 1) / base) * base;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
  cl_int status;

  int filterSize = filterWidth * filterWidth * sizeof(float);
  int imageSize = imageWidth * imageHeight * sizeof(float);

  cl_command_queue commandQueue =
      clCreateCommandQueue(*context, *device, 0, &status);
  CHECK(status, "clCreateCommandQueue");

  cl_mem inputImageObject =
      clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, &status);
  CHECK(status, "clCreateBuffer");
  cl_mem outputImageObject =
      clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status);
  CHECK(status, "clCreateBuffer");
  cl_mem filterObject =
      clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
  CHECK(status, "clCreateBuffer");

  status = clEnqueueWriteBuffer(commandQueue, inputImageObject, CL_TRUE, 0,
                                imageSize, (void *)inputImage, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");
  status = clEnqueueWriteBuffer(commandQueue, filterObject, CL_TRUE, 0,
                                filterSize, (void *)filter, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");

  cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
  CHECK(status, "clCreateKernel");

  clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&imageWidth);
  clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&imageHeight);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&inputImageObject);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outputImageObject);
  clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&filterWidth);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&filterObject);

  int roundedWidth = roundUp(imageWidth, GROUP_SIZE);
  int roundedHeight = roundUp(imageHeight, GROUP_SIZE);

  size_t globalWorksize[] = {roundedWidth, roundedHeight};
  size_t localWorksize[] = {GROUP_SIZE, GROUP_SIZE};

  status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorksize,
                                  localWorksize, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  status =
      clEnqueueReadBuffer(commandQueue, outputImageObject, CL_TRUE, 0,
                          imageSize, (void *)outputImage, NULL, NULL, NULL);
  CHECK(status, "clEnqueueReadBuffer");

  // status = clReleaseKernel(kernel);
  // CHECK(status, "clReleaseKernel");
  // status = clReleaseMemObject(filterObject);
  // CHECK(status, "clReleaseMemObject");
  // status = clReleaseMemObject(inputImageObject);
  // CHECK(status, "clReleaseMemObject");
  // status = clReleaseMemObject(outputImageObject);
  // CHECK(status, "clReleaseMemObject");
  // status = clReleaseCommandQueue(*commandQueue);
  // CHECK(status, "clReleaseCommandQueue");
}
