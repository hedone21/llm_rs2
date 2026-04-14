/*
 * microbench_launch.c — raw OpenCL per-kernel-launch overhead baseline.
 *
 * 10000회 noop 커널 launch 시간을 측정하여 per-launch 오버헤드(μs)를 구한다.
 * Rust ocl::core 버전과 동일한 구조로 작성해 wrapper 오버헤드를 비교한다.
 *
 * Android 빌드:
 *   $TOOLCHAIN/bin/aarch64-linux-android28-clang -O3 \
 *     -o microbench_launch_c microbench_launch.c -lOpenCL
 *
 * 실행 (LD_LIBRARY_PATH로 /vendor/lib64/libOpenCL.so 또는 /data/local/tmp 지정):
 *   LD_LIBRARY_PATH=/data/local/tmp ./microbench_launch_c
 */

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 10000
#define WARMUP 1000

static const char *kernel_src =
    "__kernel void noop(__global int *dummy) {\n"
    "    (void)dummy;\n"
    "}\n";

static void die(const char *msg, cl_int err) {
    fprintf(stderr, "FATAL: %s (cl_int=%d)\n", msg, (int)err);
    exit(1);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

int main(void) {
    cl_int err;

    /* 1. Platform + Device */
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) die("clGetPlatformIDs", err);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) die("clGetDeviceIDs", err);

    char pname[256] = {0}, dname[256] = {0};
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("C raw OpenCL microbench\n");
    printf("Platform: %s\n", pname);
    printf("Device:   %s\n", dname);

    /* 2. Context + Queue (profiling OFF) */
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateContext", err);

    /* production 경로와 동일: profiling flag OFF */
    cl_command_queue queue =
#if defined(CL_VERSION_2_0)
        clCreateCommandQueueWithProperties(ctx, device, NULL, &err);
#else
        clCreateCommandQueue(ctx, device, 0, &err);
#endif
    if (err != CL_SUCCESS) die("clCreateCommandQueue", err);

    /* 3. Program + Kernel */
    const char *srcs[1] = {kernel_src};
    size_t lens[1] = {strlen(kernel_src)};
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, lens, &err);
    if (err != CL_SUCCESS) die("clCreateProgramWithSource", err);

    err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        die("clBuildProgram", err);
    }

    cl_kernel kernel = clCreateKernel(prog, "noop", &err);
    if (err != CL_SUCCESS) die("clCreateKernel", err);

    /* 4. Dummy buffer */
    cl_mem dummy_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS) die("clCreateBuffer", err);

    /* 5. Warmup */
    size_t gws[1] = {1};
    for (int i = 0; i < WARMUP; ++i) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dummy_buf);
        if (err != CL_SUCCESS) die("clSetKernelArg(warmup)", err);
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel(warmup)", err);
    }
    clFinish(queue);

    /* 6. Measured region */
    double t_start = now_us();
    for (int i = 0; i < N; ++i) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dummy_buf);
        if (err != CL_SUCCESS) die("clSetKernelArg(measure)", err);
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel(measure)", err);
    }
    clFinish(queue);
    double t_end = now_us();

    double total_us = t_end - t_start;
    double per_us = total_us / (double)N;

    printf("N = %d\n", N);
    printf("total = %.2f ms\n", total_us / 1000.0);
    printf("per-launch = %.3f us (includes clSetKernelArg + clEnqueueNDRangeKernel + avg clFinish/N)\n",
           per_us);
    printf("(kernel GPU time per call estimate: negligible, noop)\n");

    /* Cleanup */
    clReleaseMemObject(dummy_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
