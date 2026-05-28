// ============================================================================
// cuda-kernel-lab : fused row-wise layernorm
//
// LayerNorm over the last dimension of a (rows x cols) row-major float32
// tensor: out[r,c] = (x[r,c] - mean_r) / sqrt(var_r + eps). No affine
// (weight/bias) so it matches torch.nn.functional.layer_norm(x, (cols,)) with
// no weight or bias. var is the biased (population) variance, matching torch.
//
// One thread block per row. Mean and variance are obtained in a single pass
// using sum and sum-of-squares, each finished with one block-wide reduction,
// so the row is streamed from global memory a small constant number of times.
// Memory bound -> figure of merit is achieved HBM bandwidth, not FLOPS.
// Validated against torch in bench/bench_fused.py.
//
// Usage:  ./layernorm <rows> <cols> <in.f32> <out.f32>   -> prints timing JSON
// ============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err__));                                    \
      exit(EXIT_FAILURE);                                                    \
    }                                                                         \
  } while (0)

constexpr int BLOCK = 256;
constexpr int WARMUP_ITERS = 10;
constexpr int TIMED_ITERS = 50;
constexpr float EPS = 1e-5f;  // matches torch default

__device__ float block_sum(float val, float *smem) {
  smem[threadIdx.x] = val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  return smem[0];
}

__global__ void layernorm_kernel(const float *x, float *out, int rows,
                                 int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *xr = x + (size_t)row * cols;
  float *outr = out + (size_t)row * cols;
  __shared__ float smem[BLOCK];

  // Single streaming pass for sum and sum-of-squares.
  float local_sum = 0.0f, local_sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = xr[c];
    local_sum += v;
    local_sq += v * v;
  }
  const float sum = block_sum(local_sum, smem);
  __syncthreads();
  const float sumsq = block_sum(local_sq, smem);

  const float mean = sum / cols;
  const float var = sumsq / cols - mean * mean;  // biased variance, like torch
  const float inv_std = rsqrtf(var + EPS);

  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    outr[c] = (xr[c] - mean) * inv_std;
}

// float4-vectorized variant for cols % 4 == 0 (16-byte aligned rows).
__global__ void layernorm_kernel_vec4(const float *x, float *out, int rows,
                                      int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float4 *xr = reinterpret_cast<const float4 *>(x + (size_t)row * cols);
  float4 *outr = reinterpret_cast<float4 *>(out + (size_t)row * cols);
  const int cols4 = cols >> 2;
  __shared__ float smem[BLOCK];

  float local_sum = 0.0f, local_sq = 0.0f;
  for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
    float4 v = xr[c];
    local_sum += v.x + v.y + v.z + v.w;
    local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  const float sum = block_sum(local_sum, smem);
  __syncthreads();
  const float sumsq = block_sum(local_sq, smem);

  const float mean = sum / cols;
  const float var = sumsq / cols - mean * mean;
  const float inv_std = rsqrtf(var + EPS);

  for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
    float4 v = xr[c];
    v.x = (v.x - mean) * inv_std;
    v.y = (v.y - mean) * inv_std;
    v.z = (v.z - mean) * inv_std;
    v.w = (v.w - mean) * inv_std;
    outr[c] = v;
  }
}

static std::vector<float> read_raw(const char *path, size_t n) {
  std::vector<float> v(n);
  FILE *f = fopen(path, "rb");
  if (!f || fread(v.data(), sizeof(float), n, f) != n) {
    fprintf(stderr, "read error on %s\n", path);
    exit(EXIT_FAILURE);
  }
  fclose(f);
  return v;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <rows> <cols> <in.f32> <out.f32>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int rows = atoi(argv[1]), cols = atoi(argv[2]);
  const size_t n = (size_t)rows * cols;
  std::vector<float> hx = read_raw(argv[3], n), hout(n);

  float *dx, *dout;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  const bool vec4 = (cols % 4 == 0);
  auto launch = [&] {
    if (vec4)
      layernorm_kernel_vec4<<<rows, BLOCK>>>(dx, dout, rows, cols);
    else
      layernorm_kernel<<<rows, BLOCK>>>(dx, dout, rows, cols);
  };

  for (int i = 0; i < WARMUP_ITERS; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  double total = 0.0, best = 1e30;
  for (int i = 0; i < TIMED_ITERS; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total += ms;
    if (ms < best) best = ms;
  }
  const double mean_ms = total / TIMED_ITERS;

  CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  FILE *f = fopen(argv[4], "wb");
  fwrite(hout.data(), sizeof(float), n, f);
  fclose(f);

  const double bytes = 2.0 * n * sizeof(float);
  const double gbps_mean = bytes / (mean_ms * 1e-3) / 1e9;
  const double gbps_best = bytes / (best * 1e-3) / 1e9;
  printf(
      "{\"op\":\"layernorm\",\"rows\":%d,\"cols\":%d,\"mean_ms\":%.6f,"
      "\"best_ms\":%.6f,\"gbps_mean\":%.2f,\"gbps_best\":%.2f}\n",
      rows, cols, mean_ms, best, gbps_mean, gbps_best);

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dout));
  return EXIT_SUCCESS;
}
