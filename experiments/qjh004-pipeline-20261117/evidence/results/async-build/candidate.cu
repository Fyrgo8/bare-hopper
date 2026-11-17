#include "candidate.hpp"
#include "hopper_ptx.cuh"
#include <cuda.h>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace bare::candidate {
namespace {
constexpr int BM = 64, BN = 128, BK = 64;
constexpr int StableK = 16384, SegmentK = 256;

// The WGMMA partial is bounded in length. Merge completed partials using
// ordinary FP32 additions, before the single final BF16 conversion.
template <bool Stable> struct Accumulator {
  float part[64] = {};
  float total[Stable ? 64 : 1] = {};

  __device__ void finish_tile(int tile, int tiles) {
    if constexpr (Stable) {
      if ((tile + 1) % (SegmentK / BK) == 0 || tile + 1 == tiles) {
#pragma unroll
        for (int i = 0; i < 64; ++i) {
          total[i] += part[i];
          part[i] = 0;
        }
        // All WGMMA operations have been waited for before this call.
        // Publish the ordinary register writes to the next WGMMA group.
        hopper::fence();
      }
    }
  }
  __device__ float const (&result() const)[64] {
    if constexpr (Stable) return total;
    else return part;
  }
};

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("candidate: ") + cudaGetErrorString(status));
}

// Vectorize each contiguous K row; partial K vectors use scalar loads.
template <int Rows>
__device__ void load(__nv_bfloat16 *dst, __nv_bfloat16 const *src,
                     int base_row, int rows, int k, int base_k) {
  for (int v = threadIdx.x; v < Rows * BK / 8; v += 128) {
    int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
    int offset = (row * BK + col) ^ ((row & 7) * 8);
    uint4 data{};
    if (base_row + row < rows && base_k + col + 7 < k) {
      data = *reinterpret_cast<uint4 const *>(src + size_t(base_row + row) * k +
                                              base_k + col);
    } else if (base_row + row < rows) {
      auto *elements = reinterpret_cast<__nv_bfloat16 *>(&data);
      for (int j = 0; j < 8 && base_k + col + j < k; ++j)
        elements[j] = src[size_t(base_row + row) * k + base_k + col + j];
    }
    *reinterpret_cast<uint4 *>(dst + offset) = data;
  }
}

__device__ void store(float const (&acc)[64], Problem p, Buffers b) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int i = 0; i < 64; i += 2) {
    int row = int(blockIdx.y) * BM + warp * 16 + lane / 4 + (i % 4) / 2 * 8;
    int col = int(blockIdx.x) * BN + (i / 4) * 8 + (lane % 4) * 2 + i % 2;
    if (row < p.m && col < p.n)
      *reinterpret_cast<__nv_bfloat162 *>(b.d + size_t(row) * p.n + col) =
          __floats2bfloat162_rn(acc[i], acc[i + 1]);
  }
}

// Split-K partials remain FP32. Each CTA owns a disjoint output tile and
// split, so no atomics or inter-CTA synchronization are needed here.
__device__ void store_partial(float const (&acc)[64], Problem p, float *partial) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  size_t base = size_t(blockIdx.z) * p.m * p.n;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    int row = int(blockIdx.y) * BM + warp * 16 + lane / 4 + (i % 4) / 2 * 8;
    int col = int(blockIdx.x) * BN + (i / 4) * 8 + (lane % 4) * 2 + i % 2;
    if (row < p.m && col < p.n)
      partial[base + size_t(row) * p.n + col] = acc[i];
  }
}

__global__ void reduce_splits(Problem p, Buffers b, float const *partial, int splits) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t elements = size_t(p.m) * p.n;
  if (index < elements) {
    float sum = 0;
#pragma unroll 1
    for (int split = 0; split < splits; ++split)
      sum += partial[size_t(split) * elements + index];
    b.d[index] = __float2bfloat16_rn(sum);
  }
}

template <bool Stable = false>
__global__ __launch_bounds__(128) void vector_gemm(Problem p, Buffers b) {
  __shared__ __align__(1024) __nv_bfloat16 a[BM * BK];
  __shared__ __align__(1024) __nv_bfloat16 bb[BN * BK];
  Accumulator<Stable> acc;
  hopper::fence();
  for (int k = 0; k < p.k; k += BK) {
    load<BM>(a, b.a, int(blockIdx.y) * BM, p.m, p.k, k);
    load<BN>(bb, b.b, int(blockIdx.x) * BN, p.n, p.k, k);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int sub = 0; sub < BK; sub += 16)
      hopper::mma(acc.part, hopper::descriptor(a + sub), hopper::descriptor(bb + sub));
    hopper::commit();
    hopper::wait();
    acc.finish_tile(k / BK, (p.k + BK - 1) / BK);
    __syncthreads(); // WGMMA has finished reading operands before reuse.
  }
  store(acc.result(), p, b);
}

struct Context {
  Problem problem;
  Buffers buffers;
  CUtensorMap map_a{}, map_b{};
  int stages = 0;
  bool specialized = false;
  bool stable = false;
  bool async_pipeline = false;
  int splits = 1;
  float *partial = nullptr;
  ~Context() { if (partial) cudaFree(partial); }
};

__device__ void barrier_wait(uint64_t *barrier, int phase) {
  asm volatile(
      "{ .reg .pred done;\n"
      "retry: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra retry; }\n"
      :: "r"(hopper::smem_address(barrier)), "r"(phase) : "memory");
}

__device__ void tma_load(CUtensorMap const *map, void *dst, uint64_t *barrier,
                         int k, int row) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3, %4}], [%2];"
      :: "r"(hopper::smem_address(dst)), "l"(map),
         "r"(hopper::smem_address(barrier)), "r"(k), "r"(row) : "memory");
}

template <int Stages, bool Specialized = false, bool Stable = false, bool Split = false,
          bool Async = false>
__global__ __launch_bounds__(Specialized ? 256 : 128) void tma_gemm(
    Problem p, Buffers b, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b, float *partial) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *data = reinterpret_cast<__nv_bfloat16 (*)[(BM + BN) * BK]>(storage);
  __shared__ __align__(8) uint64_t full[Stages];
  __shared__ __align__(8) uint64_t empty[Stages];
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < Stages; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                   :: "r"(hopper::smem_address(full + s)) : "memory");
      if constexpr (Specialized)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(hopper::smem_address(empty + s)) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  Accumulator<Stable> acc;
  hopper::fence();
  int k_start = Split ? int(blockIdx.z) * SegmentK : 0;
  int remaining = p.k - k_start;
  int extent = Split && remaining > SegmentK ? SegmentK : remaining;
  int tiles = (extent + BK - 1) / BK;
  if constexpr (Specialized) {
    if (threadIdx.x >= 128) {
      if (threadIdx.x == 128) {
        for (int tile = 0; tile < tiles; ++tile) {
          int s = tile % Stages;
          if (tile >= Stages)
            barrier_wait(empty + s, ((tile / Stages) - 1) & 1);
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                       :: "r"(hopper::smem_address(full + s)),
                          "r"((BM + BN) * BK * 2) : "memory");
          tma_load(&map_a, data[s], full + s, k_start + tile * BK, int(blockIdx.y) * BM);
          tma_load(&map_b, data[s] + BM * BK, full + s, k_start + tile * BK,
                   int(blockIdx.x) * BN);
        }
      }
      return;
    }
  }
  // Prime the ring. Only thread zero issues TMA; all threads compute.
  if (!Specialized && threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < Stages; ++s) {
      if (s < tiles) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(hopper::smem_address(full + s)),
                        "r"((BM + BN) * BK * 2) : "memory");
        tma_load(&map_a, data[s], full + s, k_start + s * BK, int(blockIdx.y) * BM);
        tma_load(&map_b, data[s] + BM * BK, full + s, k_start + s * BK,
                 int(blockIdx.x) * BN);
      }
    }
  }
  for (int tile = 0; tile < tiles; ++tile) {
    int s = tile % Stages;
    barrier_wait(full + s, (tile / Stages) & 1);
    auto *a = data[s];
    auto *bb = a + BM * BK;
#pragma unroll
    for (int sub = 0; sub < BK; sub += 16)
      hopper::mma(acc.part, hopper::descriptor(a + sub), hopper::descriptor(bb + sub));
    hopper::commit();
    if constexpr (Async) {
      // A single committed group may remain pending. Only the previous
      // operand slot is released; the current slot remains owned by WGMMA.
      if constexpr (Stable) {
        if ((tile + 1) % (SegmentK / BK) == 0 || tile + 1 == tiles) {
          hopper::wait();
          acc.finish_tile(tile, tiles);
        } else {
          hopper::wait_one();
        }
      } else {
        hopper::wait_one();
      }
      if (tile > 0) {
        __syncthreads();
        int previous = (tile - 1) % Stages;
        int next = tile - 1 + Stages;
        if (threadIdx.x == 0 && next < tiles) {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                       :: "r"(hopper::smem_address(full + previous)),
                          "r"((BM + BN) * BK * 2) : "memory");
          tma_load(&map_a, data[previous], full + previous,
                   k_start + next * BK, int(blockIdx.y) * BM);
          tma_load(&map_b, data[previous] + BM * BK, full + previous,
                   k_start + next * BK, int(blockIdx.x) * BN);
        }
      }
      continue;
    }
    hopper::wait();
    acc.finish_tile(tile, tiles);
    if constexpr (Specialized) {
      // Only the consumer warpgroup participates in this named barrier.
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (threadIdx.x == 0)
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(hopper::smem_address(empty + s)) : "memory");
    } else {
      __syncthreads();
    }
    // No reuse until every warp's WGMMA has completed. The next ring slot
    // can compute while this slot's next TMA transfer is in flight.
    if (!Specialized && threadIdx.x == 0 && tile + Stages < tiles) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   :: "r"(hopper::smem_address(full + s)),
                      "r"((BM + BN) * BK * 2) : "memory");
      tma_load(&map_a, a, full + s, k_start + (tile + Stages) * BK, int(blockIdx.y) * BM);
      tma_load(&map_b, bb, full + s, k_start + (tile + Stages) * BK, int(blockIdx.x) * BN);
    }
  }
  if constexpr (Async) hopper::wait();
  if constexpr (Split)
    store_partial(acc.result(), p, partial);
  else
    store(acc.result(), p, b);
}

void make_map(CUtensorMap &map, void const *ptr, int rows, int k, int tile_rows) {
  uint64_t dims[] = {uint64_t(k), uint64_t(rows)};
  uint64_t strides[] = {uint64_t(k) * 2};
  uint32_t box[] = {BK, uint32_t(tile_rows)}, element_strides[] = {1, 1};
  auto status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void *>(ptr),
      dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS)
    throw std::runtime_error("candidate: tensor map error " + std::to_string(status));
}

template <int Stages, bool Specialized, bool Stable, bool Split = false, bool Async = false>
void dispatch(Context const &c, dim3 grid, cudaStream_t stream, bool configure) {
  constexpr int bytes = Stages * (BM + BN) * BK * 2;
  if (configure)
    check(cudaFuncSetAttribute(tma_gemm<Stages, Specialized, Stable, Split, Async>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  else
    tma_gemm<Stages, Specialized, Stable, Split, Async><<<grid, Specialized ? 256 : 128, bytes, stream>>>(
        c.problem, c.buffers, c.map_a, c.map_b, c.partial);
}

template <int Stages, bool Specialized>
void dispatch(Context const &c, dim3 grid, cudaStream_t stream, bool configure) {
  if constexpr (Stages >= 2 && !Specialized) {
    if (c.async_pipeline) {
      if (c.splits > 1)
        dispatch<Stages, false, false, true, true>(c, grid, stream, configure);
      else if (c.stable)
        dispatch<Stages, false, true, false, true>(c, grid, stream, configure);
      else
        dispatch<Stages, false, false, false, true>(c, grid, stream, configure);
      return;
    }
  }
  if (c.splits > 1)
    dispatch<Stages, Specialized, false, true>(c, grid, stream, configure);
  else if (c.stable)
    dispatch<Stages, Specialized, true>(c, grid, stream, configure);
  else
    dispatch<Stages, Specialized, false>(c, grid, stream, configure);
}

void dispatch(Context const &c, dim3 grid, cudaStream_t stream, bool configure) {
#define BARE_DISPATCH(S) \
  case S: \
    if (c.specialized) dispatch<S, true>(c, grid, stream, configure); \
    else dispatch<S, false>(c, grid, stream, configure); \
    break
  switch (c.stages) {
    BARE_DISPATCH(1);
    BARE_DISPATCH(2);
    BARE_DISPATCH(3);
    BARE_DISPATCH(4);
  }
#undef BARE_DISPATCH
}
} // namespace

bool available() { return true; }
char const *implementation_name() {
  if (auto *ws = std::getenv("BARE_GEMM_WARP_SPECIALIZED");
      ws && std::string(ws) == "1") {
    static std::string specialized_name = std::string("bare-sm90a-bf16-64x128x64-ws-stages") +
        (std::getenv("BARE_GEMM_STAGES") ? std::getenv("BARE_GEMM_STAGES") : "1");
    return specialized_name.c_str();
  }
  char const *mode = std::getenv("BARE_GEMM_STAGES");
  if (!mode) return "bare-sm90a-bf16-64x128x64-auto-tma-s1-s3";
  int stages = std::stoi(mode);
  switch (stages) {
  case 0: return "bare-sm90a-bf16-64x128x64-vector-single-stage";
  case 1: return "bare-sm90a-bf16-64x128x64-tma-stages1";
  case 2: return "bare-sm90a-bf16-64x128x64-tma-stages2";
  case 3: return "bare-sm90a-bf16-64x128x64-tma-stages3";
  case 4: return "bare-sm90a-bf16-64x128x64-tma-stages4";
  default: throw std::runtime_error("BARE_GEMM_STAGES must be in 0..4");
  }
}
char const *name() {
  static std::string result = std::string(implementation_name()) +
      ((!std::getenv("BARE_GEMM_STABLE_LONG_K") ||
        std::string(std::getenv("BARE_GEMM_STABLE_LONG_K")) == "1")
           ? "-segment256-k16384" : "") +
      ((!std::getenv("BARE_GEMM_SPLIT_K") ||
        std::string(std::getenv("BARE_GEMM_SPLIT_K")) == "1")
           ? "-split256-small-output" : "");
  return result.c_str();
}
void *prepare(Problem p, Buffers b, cudaStream_t) {
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.k % 8 || p.n % 8 ||
      !b.a || !b.b || !b.d)
    throw std::runtime_error("candidate: invalid dimensions or buffers");
  auto c = std::make_unique<Context>();
  c->problem = p;
  c->buffers = b;
  char const *mode = std::getenv("BARE_GEMM_STAGES");
  c->stages = mode ? std::stoi(mode) : (p.k >= StableK ? 3 : 1);
  char const *specialized = std::getenv("BARE_GEMM_WARP_SPECIALIZED");
  c->specialized = specialized && std::string(specialized) == "1";
  char const *stable = std::getenv("BARE_GEMM_STABLE_LONG_K");
  c->stable = (!stable || std::string(stable) == "1") && p.k >= StableK;
  char const *async = std::getenv("BARE_GEMM_ASYNC_PIPELINE");
  c->async_pipeline = async && std::string(async) == "1";
  if (c->stages < 0 || c->stages > 4)
    throw std::runtime_error("BARE_GEMM_STAGES must be 0 (vector), or 1..4 (TMA)");
  if (c->stages) {
    // An opt-in path for long reductions with too few output CTAs to fill
    // the device. Bound scratch storage independently of the input size.
    char const *split = std::getenv("BARE_GEMM_SPLIT_K");
    size_t output_tiles = size_t((p.m + BM - 1) / BM) * ((p.n + BN - 1) / BN);
    int splits = (p.k + SegmentK - 1) / SegmentK;
    size_t bytes = size_t(splits) * p.m * p.n * sizeof(float);
    if (c->stable && (!split || std::string(split) == "1") &&
        output_tiles <= 32 && splits <= 65535 && bytes <= 64ULL * 1024 * 1024) {
      c->splits = splits;
      check(cudaMalloc(&c->partial, bytes));
    }
    make_map(c->map_a, b.a, p.m, p.k, BM);
    make_map(c->map_b, b.b, p.n, p.k, BN);
    dispatch(*c, dim3{}, nullptr, true);
  }
  return c.release();
}
void launch(void *context, cudaStream_t stream) {
  auto const &c = *static_cast<Context *>(context);
  dim3 grid((c.problem.n + BN - 1) / BN, (c.problem.m + BM - 1) / BM, c.splits);
  if (c.stages == 0) {
    if (c.stable)
      vector_gemm<true><<<grid, 128, 0, stream>>>(c.problem, c.buffers);
    else
      vector_gemm<false><<<grid, 128, 0, stream>>>(c.problem, c.buffers);
  }
  else
    dispatch(c, grid, stream, false);
  if (c.splits > 1) {
    check(cudaGetLastError());
    reduce_splits<<<(size_t(c.problem.m) * c.problem.n + 255) / 256, 256, 0, stream>>>(
        c.problem, c.buffers, c.partial, c.splits);
  }
  check(cudaGetLastError());
}
void cleanup(void *context) { delete static_cast<Context *>(context); }
} // namespace bare::candidate
