#pragma once
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <memory>

namespace bare {
// D(row-major MxN) = A(row-major MxK) * B(column-major KxN).
// BF16 input/output; FP32 accumulation; alpha=1, beta=0. No C read.
struct Problem {
  int m, n, k;
};
struct Buffers {
  __nv_bfloat16 const *a;
  __nv_bfloat16 const *b;
  __nv_bfloat16 *d;
};
class Reference {
public:
  Reference(Problem, Buffers, cudaStream_t);
  ~Reference();
  Reference(Reference const &) = delete;
  Reference &operator=(Reference const &) = delete;
  void launch(cudaStream_t);
  static char const *name();
  static int stages();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace bare
