#ifndef BILINEAR_SAMPLER_KERNEL
#define BILINEAR_SAMPLER_KERNEL

#include "NvInfer.h"

int custom_bilinear_sampler(
    cudaStream_t stream,
    int B, int H, int W, int D,
    const void* imageC1,
    const void* coords,
    void* sample
);

#endif