#include "hostFE.h"

#include <stdio.h>
#include <stdlib.h>

#include "helper.h"

static inline cl_command_queue createCommandQueue(cl_context *context,
                                                  cl_device_id *device);
static inline cl_mem createBuffer(cl_context *context, cl_mem_flags flags,
                                  size_t size);
static inline cl_kernel createKernel(cl_program *program,
                                     const char *kernelName);

static inline void copyToDevice(cl_command_queue *commandQueue,
                                cl_mem *memObject, size_t size, void *hostPtr);
static inline void copyToHost(cl_command_queue *commandQueue, cl_mem *memObject,
                              size_t size, void *hostPtr);

static inline void launchKernel(cl_command_queue *commandQueue,
                                cl_kernel *kernel, const size_t *globalWorksize,
                                const size_t *localWorksize);

static inline void releaseCommandQueue(cl_command_queue *commandQueue);
static inline void releaseMemObject(cl_mem *memObject);
static inline void releaseKernel(cl_kernel *kernel);

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
  cl_int status;
  int filterSize = filterWidth * filterWidth * sizeof(float);
  int imageSize = imageWidth * imageHeight * sizeof(float);

  cl_command_queue commandQueue = createCommandQueue(context, device);

  cl_mem inputImageObject = createBuffer(context, CL_MEM_READ_ONLY, imageSize);
  cl_mem outputImageObject =
      createBuffer(context, CL_MEM_WRITE_ONLY, imageSize);
  cl_mem filterObject = createBuffer(context, CL_MEM_READ_ONLY, filterSize);

  copyToDevice(&commandQueue, &inputImageObject, imageSize, (void *)inputImage);
  copyToDevice(&commandQueue, &filterObject, filterSize, (void *)filter);

  cl_kernel kernel = createKernel(program, "convolution");

  clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&imageWidth);
  clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&imageHeight);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&inputImageObject);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outputImageObject);
  clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&filterWidth);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&filterObject);

  // TODO: find out better work size
  size_t globalWorksize[] = {imageWidth, imageHeight};
  size_t localWorksize[] = {25, 25};
  launchKernel(&commandQueue, &kernel, globalWorksize, localWorksize);

  copyToHost(&commandQueue, &outputImageObject, imageSize, (void *)outputImage);

  releaseKernel(&kernel);
  releaseMemObject(&filterObject);
  releaseMemObject(&inputImageObject);
  releaseMemObject(&outputImageObject);
  releaseCommandQueue(&commandQueue);
}

static inline cl_command_queue createCommandQueue(cl_context *context,
                                                  cl_device_id *device) {
  cl_int status;
  cl_command_queue commandQueue =
      clCreateCommandQueue(*context, *device, 0, &status);
  CHECK(status, "clCreateCommandQueue");
  return commandQueue;
}

static inline cl_mem createBuffer(cl_context *context, cl_mem_flags flags,
                                  size_t size) {
  cl_int status;
  cl_mem memObject = clCreateBuffer(*context, flags, size, NULL, &status);
  CHECK(status, "clCreateBuffer");
  return memObject;
}

static inline cl_kernel createKernel(cl_program *program,
                                     const char *kernelName) {
  cl_int status;
  cl_kernel kernel = clCreateKernel(*program, kernelName, &status);
  CHECK(status, "clCreateKernel");
  return kernel;
}

static inline void copyToDevice(cl_command_queue *commandQueue,
                                cl_mem *memObject, size_t size, void *hostPtr) {
  cl_int status = clEnqueueWriteBuffer(*commandQueue, *memObject, CL_TRUE, 0,
                                       size, hostPtr, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");
}

static inline copyToHost(cl_command_queue *commandQueue, cl_mem *memObject,
                         size_t size, void *hostPtr) {
  cl_int status = clEnqueueReadBuffer(*commandQueue, *memObject, CL_TRUE, 0,
                                      size, hostPtr, NULL, NULL, NULL);
  CHECK(status, "clEnqueueReadBuffer");
}

static inline void launchKernel(cl_command_queue *commandQueue,
                                cl_kernel *kernel, const size_t *globalWorksize,
                                const size_t *localWorksize) {
  cl_int status =
      clEnqueueNDRangeKernel(*commandQueue, *kernel, 2, 0, globalWorksize,
                             localWorksize, 0, NULL, NULL);
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
