#include "gemm.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
#include <string>

namespace bare {
using namespace cute;
using Element = cutlass::bfloat16_t;
using Tile = Shape<_128, _128, _64>;
using Cluster = Shape<_1, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, Element,
    cutlass::layout::RowMajor, 8, Element, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element,
    cutlass::layout::RowMajor, 8, Element, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
        typename Epilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                    Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

static void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS: ") +
                             cutlassGetStatusString(status));
}
struct Reference::Impl {
  Gemm gemm;
  void *workspace = nullptr;
  Impl(Problem p, Buffers b, cudaStream_t stream) {
    auto sa = cutlass::make_cute_packed_stride(Kernel::StrideA{},
                                               make_shape(p.m, p.k, 1));
    auto sb = cutlass::make_cute_packed_stride(Kernel::StrideB{},
                                               make_shape(p.n, p.k, 1));
    auto sd = cutlass::make_cute_packed_stride(Kernel::StrideD{},
                                               make_shape(p.m, p.n, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {p.m, p.n, p.k, 1},
        {reinterpret_cast<Element const *>(b.a), sa,
         reinterpret_cast<Element const *>(b.b), sb},
        {{1.0f, 0.0f},
         reinterpret_cast<Element const *>(b.d),
         sd,
         reinterpret_cast<Element *>(b.d),
         sd},
        cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(0)};
    check(gemm.can_implement(args));
    auto bytes = Gemm::get_workspace_size(args);
    if (bytes && cudaMalloc(&workspace, bytes) != cudaSuccess)
      throw std::runtime_error("CUTLASS workspace allocation failed");
    auto status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
      cudaFree(workspace);
      workspace = nullptr;
    }
    check(status);
  }
  ~Impl() { cudaFree(workspace); }
};
Reference::Reference(Problem p, Buffers b, cudaStream_t s)
    : impl_(new Impl(p, b, s)) {}
Reference::~Reference() = default;
void Reference::launch(cudaStream_t stream) { check(impl_->gemm.run(stream)); }
char const *Reference::name() {
  return "cutlass-3.9.2-sm90a-bf16-f32-bf16-TN-rowD-128x128x64-cluster1x1x1-"
         "cooperative-autoStages-tmaEpilogue";
}
int Reference::stages() { return Mainloop::DispatchPolicy::Stages; }
} // namespace bare
