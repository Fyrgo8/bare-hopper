#include "gemm.hpp"
#include <cublas_v2.h>
#include <stdexcept>

namespace bare {
static void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Audit cuBLAS GEMM failed: " + std::to_string(status));
}
struct Reference::Impl {
  cublasHandle_t handle{};
  Problem p;
  Buffers b;
  Impl(Problem problem, Buffers buffers, cudaStream_t stream) : p(problem), b(buffers) {
    check(cublasCreate(&handle));
    check(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    check(cublasSetStream(handle, stream));
  }
  ~Impl() { cublasDestroy(handle); }
};
Reference::Reference(Problem p, Buffers b, cudaStream_t stream)
    : impl_(new Impl(p, b, stream)) {}
Reference::~Reference() = default;
void Reference::launch(cudaStream_t stream) {
  auto const &x = *impl_;
  check(cublasSetStream(x.handle, stream));
  float alpha = 1, beta = 0;
  // Column-major D^T(N,M) = B^T(N,K) * A^T(K,M), matching the original layout.
  check(cublasGemmEx(x.handle, CUBLAS_OP_T, CUBLAS_OP_N, x.p.n, x.p.m, x.p.k,
                    &alpha, x.b.b, CUDA_R_16BF, x.p.k,
                    x.b.a, CUDA_R_16BF, x.p.k, &beta,
                    x.b.d, CUDA_R_16BF, x.p.n,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
char const *Reference::name() {
  return "audit-cublasGemmEx-bf16-f32-bf16-default-no-reduced-precision-reduction";
}
int Reference::stages() { return -1; }
} // namespace bare
