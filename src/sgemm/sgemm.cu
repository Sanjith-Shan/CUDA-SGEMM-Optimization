// ============================================================================
// cuda-kernel-lab : SGEMM host driver
//
// Runs a single SGEMM kernel (or a cuBLAS baseline) at a given size, validates
// it against a pedantic-FP32 cuBLAS reference, times it with CUDA events, and
// prints one machine-readable JSON line for the Python harness to collect.
//
// Baselines (per project decision): the headline / percent-of-cuBLAS basis is
// pedantic FP32 cuBLAS (CUBLAS_PEDANTIC_MATH, no TF32). A TF32 tensor-core
// column (CUBLAS_TF32_TENSOR_OP_MATH) is reported for context only.
//
// Usage:
//   ./sgemm --device-info
//   ./sgemm <kernel> <size>            (square: M = N = K = size)
//   ./sgemm <kernel> <M> <N> <K>
//   kernels: naive coalesced smem_tiled 1d_blocktile 2d_blocktile
//            cublas cublas_tf32
// ============================================================================

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels.cuh"

// ---- Error-checking macros (print file and line on failure) ----------------
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st__ = (call);                                             \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__,  \
              (int)st__);                                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// Benchmark methodology constants.
constexpr int WARMUP_ITERS = 10;
constexpr int TIMED_ITERS = 50;
constexpr unsigned RNG_SEED = 1234u;

using LaunchFn = void (*)(int, int, int, float, const float *, const float *,
                          float, float *);

// ---------------------------------------------------------------------------
// Kernel launchers: one wrapper per rung that picks grid/block from cfg::.
// ---------------------------------------------------------------------------
static void launch_naive(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
  dim3 block(cfg::NAIVE_BLOCK, cfg::NAIVE_BLOCK);
  dim3 grid((M + cfg::NAIVE_BLOCK - 1) / cfg::NAIVE_BLOCK,
            (N + cfg::NAIVE_BLOCK - 1) / cfg::NAIVE_BLOCK);
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

static void launch_coalesced(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  constexpr uint32_t BS = cfg::COALESCE_BLOCK;
  dim3 block(BS * BS);
  dim3 grid((M + BS - 1) / BS, (N + BS - 1) / BS);
  sgemm_coalesced<BS><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

static void launch_smem(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  constexpr uint32_t T = cfg::SMEM_TILE;
  dim3 block(T * T);
  dim3 grid((M + T - 1) / T, (N + T - 1) / T);
  sgemm_smem_tiled<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

static void launch_1d(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  using namespace cfg;
  dim3 block((BT1_BM * BT1_BN) / BT1_TM);
  dim3 grid((N + BT1_BN - 1) / BT1_BN, (M + BT1_BM - 1) / BT1_BM);
  sgemm_1d_blocktile<BT1_BM, BT1_BN, BT1_BK, BT1_TM>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

static void launch_2d(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  using namespace cfg;
  dim3 block((BT2_BM * BT2_BN) / (BT2_TM * BT2_TN));
  dim3 grid((N + BT2_BN - 1) / BT2_BN, (M + BT2_BM - 1) / BT2_BM);
  sgemm_2d_blocktile<BT2_BM, BT2_BN, BT2_BK, BT2_TM, BT2_TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// ---------------------------------------------------------------------------
// cuBLAS row-major C = alpha*A*B + beta*C via the standard column-major trick:
// computing C^T = B^T * A^T with swapped operands and leading dims.
// math_mode selects pedantic FP32 vs TF32 tensor-op.
// ---------------------------------------------------------------------------
static void cublas_sgemm(cublasHandle_t handle, cublasMath_t math_mode, int M,
                         int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSetMathMode(handle, math_mode));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                           N, A, K, &beta, C, N));
}

// ---------------------------------------------------------------------------
struct DeviceInfo {
  std::string name;
  int major, minor;
  int driver, runtime;
};

static DeviceInfo get_device_info() {
  DeviceInfo d;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  d.name = prop.name;
  d.major = prop.major;
  d.minor = prop.minor;
  CUDA_CHECK(cudaDriverGetVersion(&d.driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime));
  return d;
}

static void print_device_info() {
  DeviceInfo d = get_device_info();
  // Driver/runtime version encoded as 1000*major + 10*minor.
  printf(
      "{\"gpu\":\"%s\",\"cc\":\"%d.%d\",\"driver\":\"%d.%d\",\"cuda_runtime\":"
      "\"%d.%d\"}\n",
      d.name.c_str(), d.major, d.minor, d.driver / 1000, (d.driver % 100) / 10,
      d.runtime / 1000, (d.runtime % 100) / 10);
}

// Map a kernel name to its launcher (or a cuBLAS math mode). Returns false on
// an unknown name. Shared by the timing path and the --check path.
static bool resolve(const std::string &kernel, LaunchFn *fn, bool *is_cublas,
                    cublasMath_t *mode) {
  *fn = nullptr;
  *is_cublas = false;
  *mode = CUBLAS_DEFAULT_MATH;
  if (kernel == "naive")
    *fn = launch_naive;
  else if (kernel == "coalesced")
    *fn = launch_coalesced;
  else if (kernel == "smem_tiled")
    *fn = launch_smem;
  else if (kernel == "1d_blocktile")
    *fn = launch_1d;
  else if (kernel == "2d_blocktile")
    *fn = launch_2d;
  else if (kernel == "cublas") {
    *is_cublas = true;
    *mode = CUBLAS_PEDANTIC_MATH;
  } else if (kernel == "cublas_tf32") {
    *is_cublas = true;
    *mode = CUBLAS_TF32_TENSOR_OP_MATH;
  } else {
    return false;
  }
  return true;
}

static std::vector<float> read_raw(const char *path, size_t n) {
  std::vector<float> v(n);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(EXIT_FAILURE);
  }
  if (fread(v.data(), sizeof(float), n, f) != n) {
    fprintf(stderr, "short read on %s\n", path);
    exit(EXIT_FAILURE);
  }
  fclose(f);
  return v;
}

// Run one kernel on caller-supplied A,B (from files) and write C. Used by
// verify.py to compare against an independent NumPy float64 reference.
static int run_check(const std::string &kernel, int M, int N, int K,
                     const char *aPath, const char *bPath, const char *cPath) {
  LaunchFn fn;
  bool is_cublas;
  cublasMath_t mode;
  if (!resolve(kernel, &fn, &is_cublas, &mode)) {
    fprintf(stderr, "unknown kernel: %s\n", kernel.c_str());
    return EXIT_FAILURE;
  }
  const float alpha = 1.0f, beta = 0.0f;
  const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
  std::vector<float> hA = read_raw(aPath, szA);
  std::vector<float> hB = read_raw(bPath, szB);
  std::vector<float> hC(szC);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));

  if (is_cublas) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublas_sgemm(handle, mode, M, N, K, alpha, dA, dB, beta, dC);
    CUBLAS_CHECK(cublasDestroy(handle));
  } else {
    fn(M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, szC * sizeof(float),
                        cudaMemcpyDeviceToHost));

  FILE *f = fopen(cPath, "wb");
  if (!f) {
    fprintf(stderr, "cannot open %s for write\n", cPath);
    return EXIT_FAILURE;
  }
  fwrite(hC.data(), sizeof(float), szC, f);
  fclose(f);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return EXIT_SUCCESS;
}

// Report theoretical occupancy for each rung from the CUDA occupancy API.
// This needs no performance counters, so it works even where Nsight's hardware
// counters are restricted. Occupancy = active warps / max warps per SM.
static void print_occupancy() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
  const int warpsPerSM = maxThreadsPerSM / prop.warpSize;

  struct Entry {
    const char *name;
    const void *func;
    int block;
  };
  const Entry entries[] = {
      {"naive", (const void *)sgemm_naive,
       cfg::NAIVE_BLOCK * cfg::NAIVE_BLOCK},
      {"coalesced", (const void *)sgemm_coalesced<cfg::COALESCE_BLOCK>,
       cfg::COALESCE_BLOCK * cfg::COALESCE_BLOCK},
      {"smem_tiled", (const void *)sgemm_smem_tiled<cfg::SMEM_TILE>,
       cfg::SMEM_TILE * cfg::SMEM_TILE},
      {"1d_blocktile",
       (const void *)sgemm_1d_blocktile<cfg::BT1_BM, cfg::BT1_BN, cfg::BT1_BK,
                                        cfg::BT1_TM>,
       (cfg::BT1_BM * cfg::BT1_BN) / cfg::BT1_TM},
      {"2d_blocktile",
       (const void *)sgemm_2d_blocktile<cfg::BT2_BM, cfg::BT2_BN, cfg::BT2_BK,
                                        cfg::BT2_TM, cfg::BT2_TN>,
       (cfg::BT2_BM * cfg::BT2_BN) / (cfg::BT2_TM * cfg::BT2_TN)},
  };
  for (const auto &e : entries) {
    int maxBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, e.func, e.block, 0));
    const double occ = (double)(maxBlocks * e.block) / maxThreadsPerSM;
    printf(
        "{\"kernel\":\"%s\",\"block\":%d,\"max_blocks_per_sm\":%d,"
        "\"active_warps\":%d,\"max_warps\":%d,\"theoretical_occupancy\":%.3f}\n",
        e.name, e.block, maxBlocks, maxBlocks * e.block / prop.warpSize,
        warpsPerSM, occ);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s --device-info | <kernel> <size> | <kernel> <M> <N> "
            "<K>\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  if (strcmp(argv[1], "--device-info") == 0) {
    print_device_info();
    return EXIT_SUCCESS;
  }

  if (strcmp(argv[1], "--occupancy") == 0) {
    print_occupancy();
    return EXIT_SUCCESS;
  }

  // File-based verification mode for an independent (NumPy) reference:
  //   --check <kernel> <M> <N> <K> <Afile> <Bfile> <Cout>
  // Reads raw little-endian float32 A and B, runs the kernel once, writes C.
  if (strcmp(argv[1], "--check") == 0) {
    if (argc != 9) {
      fprintf(stderr,
              "usage: %s --check <kernel> <M> <N> <K> <Afile> <Bfile> "
              "<Cout>\n",
              argv[0]);
      return EXIT_FAILURE;
    }
    return run_check(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                     argv[6], argv[7], argv[8]);
  }

  const std::string kernel = argv[1];
  int M, N, K;
  if (argc == 3) {
    M = N = K = atoi(argv[2]);
  } else if (argc == 6) {
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
  } else {
    fprintf(stderr, "bad arguments\n");
    return EXIT_FAILURE;
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    fprintf(stderr, "sizes must be positive\n");
    return EXIT_FAILURE;
  }

  const float alpha = 1.0f, beta = 0.0f;
  const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

  // ---- Host init with a fixed seed for reproducibility --------------------
  std::vector<float> hA(szA), hB(szB), hC(szC, 0.0f), hRef(szC), hOut(szC);
  std::mt19937 rng(RNG_SEED);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < szA; ++i) hA[i] = dist(rng);
  for (size_t i = 0; i < szB; ++i) hB[i] = dist(rng);

  // ---- Device buffers -----------------------------------------------------
  float *dA, *dB, *dC, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dRef, szC * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float),
                        cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ---- Trusted reference: pedantic FP32 cuBLAS ----------------------------
  cublas_sgemm(handle, CUBLAS_PEDANTIC_MATH, M, N, K, alpha, dA, dB, beta,
               dRef);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, szC * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ---- Resolve which thing we are timing ----------------------------------
  LaunchFn fn = nullptr;
  cublasMath_t cublas_mode = CUBLAS_DEFAULT_MATH;
  bool is_cublas = false;
  if (!resolve(kernel, &fn, &is_cublas, &cublas_mode)) {
    fprintf(stderr, "unknown kernel: %s\n", kernel.c_str());
    return EXIT_FAILURE;
  }

  auto run_once = [&](float *out) {
    if (is_cublas)
      cublas_sgemm(handle, cublas_mode, M, N, K, alpha, dA, dB, beta, out);
    else
      fn(M, N, K, alpha, dA, dB, beta, out);
  };

  // ---- Correctness: run once, compare to reference ------------------------
  CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
  run_once(dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hOut.data(), dC, szC * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Normwise relative error: max|C-Cref| / max|Cref| (infinity-norm relative).
  // Per-element relative error is ill-posed for a random GEMM because output
  // entries can be ~0 from cancellation; normalizing by the largest reference
  // magnitude gives a well-defined error at the scale of the result.
  double max_abs = 0.0, max_ref = 0.0;
  for (size_t i = 0; i < szC; ++i) {
    double abs_err = fabs((double)hOut[i] - (double)hRef[i]);
    if (abs_err > max_abs) max_abs = abs_err;
    double a = fabs((double)hRef[i]);
    if (a > max_ref) max_ref = a;
  }
  const double max_rel = max_abs / (max_ref + 1e-30);
  // cublas_tf32 is a lower-precision baseline, not a kernel under test; its
  // error vs pedantic FP32 is expected and reported for context, not gated.
  const bool gated = (kernel != "cublas_tf32");
  const bool correct = (max_abs < 1e-2 && max_rel < 1e-2);

  // ---- Timing: CUDA events, warmup then timed repeats ---------------------
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < WARMUP_ITERS; ++i) run_once(dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0.0, best_ms = 1e30;
  for (int i = 0; i < TIMED_ITERS; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    run_once(dC);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
    if (ms < best_ms) best_ms = ms;
  }
  const double mean_ms = total_ms / TIMED_ITERS;

  const double flops = 2.0 * (double)M * (double)N * (double)K;
  const double gflops_mean = flops / (mean_ms * 1e-3) / 1e9;
  const double gflops_best = flops / (best_ms * 1e-3) / 1e9;

  printf(
      "{\"kernel\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,"
      "\"mean_ms\":%.6f,\"best_ms\":%.6f,"
      "\"gflops_mean\":%.3f,\"gflops_best\":%.3f,"
      "\"max_abs_err\":%.3e,\"max_rel_err\":%.3e,"
      "\"correct\":%s,\"gated\":%s}\n",
      kernel.c_str(), M, N, K, mean_ms, best_ms, gflops_mean, gflops_best,
      max_abs, max_rel, correct ? "true" : "false", gated ? "true" : "false");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dRef));

  // Exit non-zero if a gated kernel failed correctness, so callers can detect.
  return (gated && !correct) ? EXIT_FAILURE : EXIT_SUCCESS;
}
