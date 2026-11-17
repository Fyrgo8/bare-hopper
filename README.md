# bare-hopper

用 CUDA C++ 和 PTX 手写 Hopper GPU 算子，不依赖 CUTLASS/CuTe，通过复现高性能实现锻炼算子开发基本功。

从 BF16 GEMM 开始，逐步复现 FlashAttention-3 的前向和反向。在 H20 上保证数值正确，并以每个约定测试用例的耗时不超过参考实现的 103% 为目标。

## BF16 GEMM 优化轮次

第 1 轮：候选已实现 64×128×64 tile 的 BF16 GEMM，使用向量加载、128-byte shared-memory swizzle 和手写 WGMMA PTX，FP32 累加后转换为 BF16。支持正维度、N/K 为 8 的倍数，并对 M/N/K 尾块处理边界。候选单独编译，不获得 CUTLASS/CuTe include 路径。

第 2 轮：默认使用单级 TMA；BARE_GEMM_STAGES=0 选择普通向量加载，1–4 选择 TMA 环形缓冲，BARE_GEMM_WARP_SPECIALIZED=1 启用独立搬运和计算 warpgroup。tensor map 和资源设置在 prepare 阶段完成，launch 不分配内存。每个 stage 复用前等待引用它的 WGMMA 完成。

