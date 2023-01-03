#include "hostFE.h"

#include <stdio.h>
#include <stdlib.h>

#include "helper.h"

static inline cl_command_queue createCommandQueue(cl_context *context,
                                                  cl_device_id *device);
static inline cl_mem createBuffer(cl_context *context, cl_mem_flags flags,
                                  size_t size, void *hostPtr);
static inline cl_kernel createKernel(cl_program *program,
                                     const char *kernelName);

static inline void copyData(cl_command_queue *commandQueue, cl_mem *memObject,
                            size_t size, void *hostPtr);

static inline void launchKernel(cl_command_queue *commandQueue,
                                cl_kernel *kernel);

static inline void releaseCommandQueue(cl_command_queue *commandQueue);
static inline void releaseMemObject(cl_mem *memObject);
static inline void releaseKernel(cl_kernel *kernel);

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
  cl_int status;
  int filterSize = filterWidth * filterWidth * sizeof(float);
  int imageSize = imageWidth * imageHeight * sizeof(float);

  auto commandQueue = createCommandQueue(*context, *device);

  auto inputImageObject =
      createBuffer(*context, CL_MEM_READ_ONLY, imageSize, inputImage);
  auto outputImageObject =
      createBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, outputImage);
  auto filterObject =
      createBuffer(*context, CL_MEM_READ_ONLY, filterSize, filter);

  copyData(commandQueue, inputImageObject, imageSize, (void *)inputImage);
  copyData(commandQueue, filterObject, filterSize, (void *)filter);

  auto kernel = createKernel(*program, "convolution");

  clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&imageWidth);
  clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&imageHeight);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&inputImageObject);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outputImageObject);
  clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&filterWidth);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&filterObject);

  launchKernel(commandQueue, kernel);

  release(kernel);
  release(filterObject);
  release(inputImageObject);
  release(outputImageObject);
  release(commandQueue);
}

static inline cl_command_queue createCommandQueue(cl_context *context,
                                                  cl_device_id *device) {
  cl_int status;
  auto commandQueue = clCreateCommandQueue(*context, *device, 0, &status);
  CHECK(status, "clCreateCommandQueue");
  return commandQueue;
}

static inline cl_mem createBuffer(cl_context *context, cl_mem_flags flags,
                                  size_t size, void *hostPtr) {
  cl_int status;
  auto memObject = clCreateBuffer(*context, flags, size, hostPtr, &status);
  CHECK(status, "clCreateBuffer");
  return memObject;
}

static inline cl_kernel createKernel(cl_program *program,
                                     const char *kernelName) {
  cl_int status;
  auto kernel = clCreateKernel(*program, kernelName, &status);
  CHECK(status, "clCreateKernel");
  return kernel;
}

static inline void copyData(cl_command_queue *commandQueue, cl_mem *memObject,
                            size_t size, void *hostPtr) {
  cl_int status = clEnqueueWriteBuffer(*commandQueue, *memObject, CL_TRUE, 0,
                                       size, hostPtr, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");
}

static inline void launchKernel(cl_command_queue *commandQueue,
                                cl_kernel *kernel) {
  // TODO: find out better work size
  size_t globalWorkSize[] = {imageWidth, imageHeight};
  size_t localWorkSize[] = {25, 25};

  cl_int status =
      clEnqueueNDRangeKernel(*commandQueue, *kernel, 2, 0, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");
}

static inline void releaseCommandQueue(cl_command_queue *commandQueue) {
  cl_int status = clReleaseCommandQueue(*commandQueue);
  CHECK(status, "clReleaseCommandQueue");
}

static inline void releaseMemObject(cl_mem *memObject) {
  cl_int status = clReleaseMemObject(*memObject);
  CHECK(status, "clReleaseMemObject");
}

static inline void releaseKernel(cl_kernel *kernel) {
  cl_int status = clReleaseKernel(*kernel);
  CHECK(status, "clReleaseKernel");
}
