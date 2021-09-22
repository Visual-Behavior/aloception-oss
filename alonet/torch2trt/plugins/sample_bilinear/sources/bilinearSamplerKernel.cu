#include <bilinearSamplerKernel.h>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

template<typename T>
__device__ T saturate(T x, const T v_min, const T v_max){
    if(x < v_min) return v_min;
    if(x > v_max) return v_max;
    return x;
}

template<typename T> // T should be fp32 or fp16 (float or half)
__global__ void biliner_sampler_kernel(
    int B, int H, int W, int D,
    const T* imageC1, // (B, H, W, 1)
    const T* coords, // (B, D, D, 2)
    T* output // (B, D, D, 1)
)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int batch = blockIdx.x;

    // Get coords
    T x = coords[batch*D*D*2 + j*D*2 + i*2];
    T y = coords[batch*D*D*2 + j*D*2 + i*2 + 1];
    // bool sample_is_inside = (x >= -1) && (x <= 1) && (y >= -1) && (y <= 1);
    // adapt coords for torch.grid_sampler
    x = (x + 1.0) * (T)(W-1) / 2.0;
    y = (y + 1.0) * (T)(H-1) / 2.0;
    bool can_be_sampled = (x >= -1 && x <= W && y >= -1 && y <= H);
    // Clip coords and get 4 coords
    x = saturate<T>(x, -1.0, (T)(W));
    y = saturate<T>(y, -1.0, (T)(H));
    T x0 = floor(x);
    T x1 = ceil(x);
    T y0 = floor(y);
    T y1 = ceil(y);
    // Calculate bilinear coeffs
    T c00 = (y1 - y)*(x1 - x);
    T c01 = (y1 - y)*(x - x1 + 1);
    T c10 = (y - y1 + 1)*(x1 - x);
    T c11 = (y - y1 + 1)*(x - x1 + 1);
    // Get 4 samples
    T s00 = 0;
    if (can_be_sampled && x >= 0 && y >= 0)
    {
        const int ptr00 = batch*H*W + W*(int)y0 + (int)x0;
        s00 = imageC1[ptr00];
    }

    T s01 = 0;
    if (can_be_sampled && x <= W -1 && y >= 0)
    {
        const int ptr01 = batch*H*W + W*(int)y0 + (int)x1;
        s01 = imageC1[ptr01];
    }

    T s10 = 0;
    if (can_be_sampled && x >= 0 && y <= H - 1)
    {
        const int ptr10 = batch*H*W + W*(int)y1 + (int)x0;
        s10 = imageC1[ptr10];
    }

    T s11 = 0;
    if (can_be_sampled && x <= W -1 && y <= H - 1)
    {
        const int ptr11 = batch*H*W + W*(int)y1 + (int)x1;
        s11 = imageC1[ptr11];
    }
    output[batch*D*D + j*D + i] = (c00 * s00 + c01 * s01 + c10 * s10 + c11 * s11);
}

int custom_bilinear_sampler(
    cudaStream_t stream,
    int B, int H, int W, int D,
    const void* imageC1, // shape (B, H, W, 1)
    const void* coords, // shape (B, D, D, 2)
    void* sample
)
{
    assertm(D*D <= 1024, "Max block volume is 1024\n");
    dim3 blockSize(D, D);
    dim3 gridSize(B, 1);
    biliner_sampler_kernel<<<gridSize, blockSize, 0, stream>>>(
        B, H, W, D,
        static_cast<const float*>(imageC1),
        static_cast<const float*>(coords),
        static_cast<float*>(sample));
    return 0;
}
