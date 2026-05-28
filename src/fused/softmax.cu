// ============================================================================
// cuda-kernel-lab : fused row-wise softmax
//
// Numerically stable softmax over the last dimension of a (rows x cols)
// row-major float32 tensor: out[r,c] = exp(x[r,c] - max_r) / sum_r exp(...).
// One thread block per row; the row max and the normalization sum are each
// found with a single block-wide reduction, so the row is read from global
// memory a small constant number of times (the "fusion" — no separate max /
// exp / sum / divide kernels round-tripping through global memory).
//
// This is a memory-bound op, so the figure of merit is achieved HBM bandwidth,
// not FLOPS. Validated against torch.softmax in bench/bench_fused.py.
//
// Usage (file handoff, so torch can supply inputs and check outputs):
//   ./softmax <rows> <cols> <in.f32> <out.f32>   -> prints timing JSON
// ============================================================================

#include <cuda_runtime.h>

#include <cfloat>
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

constexpr int BLOCK = 256;       // threads per row
constexpr int WARMUP_ITERS = 10;
constexpr int TIMED_ITERS = 50;

// Block-wide reduction into shared memory. op: 0 = max, 1 = sum.
template <int OP>
__device__ float block_reduce(float val, float *smem) {
  smem[threadIdx.x] = val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float other = smem[threadIdx.x + stride];
      smem[threadIdx.x] = (OP == 0) ? fmaxf(smem[threadIdx.x], other)
                                    : smem[threadIdx.x] + other;
    }
    __syncthreads();
  }
  return smem[0];
}

__global__ void softmax_kernel(const float *x, float *out, int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *xr = x + (size_t)row * cols;
  float *outr = out + (size_t)row * cols;
  __shared__ float smem[BLOCK];

  // Pass 1: row max (stable softmax shift).
  float local_max = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    local_max = fmaxf(local_max, xr[c]);
  const float row_max = block_reduce<0>(local_max, smem);
  __syncthreads();

  // Pass 2: exp(x - max) into output, accumulate the row sum.
  float local_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = __expf(xr[c] - row_max);
    outr[c] = e;
    local_sum += e;
  }
  const float row_sum = block_reduce<1>(local_sum, smem);
  const float inv = 1.0f / row_sum;

  // Pass 3: normalize.
  for (int c = threadIdx.x; c < cols; c += blockDim.x) outr[c] *= inv;
}

// float4-vectorized variant: same three passes, but each thread loads/stores
// 16 bytes at a time, which raises memory throughput on this bandwidth-bound
// op. Requires cols % 4 == 0 (row offsets are then 16-byte aligned).
__global__ void softmax_kernel_vec4(const float *x, float *out, int rows,
                                    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float4 *xr = reinterpret_cast<const float4 *>(x + (size_t)row * cols);
  float4 *outr = reinterpret_cast<float4 *>(out + (size_t)row * cols);
  const int cols4 = cols >> 2;
  __shared__ float smem[BLOCK];

  float local_max = -FLT_MAX;
  for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
    float4 v = xr[c];
    local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  const float row_max = block_reduce<0>(local_max, smem);
  __syncthreads();

  float local_sum = 0.0f;
  for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
    float4 v = xr[c];
    v.x = __expf(v.x - row_max);
    v.y = __expf(v.y - row_max);
    v.z = __expf(v.z - row_max);
    v.w = __expf(v.w - row_max);
    outr[c] = v;
    local_sum += v.x + v.y + v.z + v.w;
  }
  const float inv = 1.0f / block_reduce<1>(local_sum, smem);

  for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
    float4 v = outr[c];
    v.x *= inv;
    v.y *= inv;
    v.z *= inv;
    v.w *= inv;
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
      softmax_kernel_vec4<<<rows, BLOCK>>>(dx, dout, rows, cols);
    else
      softmax_kernel<<<rows, BLOCK>>>(dx, dout, rows, cols);
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

  // Bandwidth: one read + one write of the tensor.
  const double bytes = 2.0 * n * sizeof(float);
  const double gbps_mean = bytes / (mean_ms * 1e-3) / 1e9;
  const double gbps_best = bytes / (best * 1e-3) / 1e9;
  printf(
      "{\"op\":\"softmax\",\"rows\":%d,\"cols\":%d,\"mean_ms\":%.6f,"
      "\"best_ms\":%.6f,\"gbps_mean\":%.2f,\"gbps_best\":%.2f}\n",
      rows, cols, mean_ms, best, gbps_mean, gbps_best);

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dout));
  return EXIT_SUCCESS;
}
