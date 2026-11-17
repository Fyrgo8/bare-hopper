# bare-hopper

用 CUDA C++ 和 PTX 手写 Hopper GPU 算子，不依赖 CUTLASS/CuTe，通过复现高性能实现锻炼算子开发基本功。

从 BF16 GEMM 开始，逐步复现 FlashAttention-3 的前向和反向。在 H20 上保证数值正确，并以每个约定测试用例的耗时不超过参考实现的 103% 为目标。

## BF16 GEMM 优化轮次

第 1 轮：候选已实现 64×128×64 tile 的 BF16 GEMM，使用向量加载、128-byte shared-memory swizzle 和手写 WGMMA PTX，FP32 累加后转换为 BF16。支持正维度、N/K 为 8 的倍数，并对 M/N/K 尾块处理边界。候选单独编译，不获得 CUTLASS/CuTe include 路径。

第 2 轮：默认使用单级 TMA；BARE_GEMM_STAGES=0 选择普通向量加载，1–4 选择 TMA 环形缓冲，BARE_GEMM_WARP_SPECIALIZED=1 启用独立搬运和计算 warpgroup。tensor map 和资源设置在 prepare 阶段完成，launch 不分配内存。每个 stage 复用前等待引用它的 WGMMA 完成。

第 3 轮：支持 --validate-only、8 种输入分布、刷新输入后的重复检查和独立 FP64 DGEMM 基准。python3 scripts/run_coverage.py --gpu 0 对单级 TMA 扫描 159 个形状，其余路径使用代表形状；bash scripts/check_gemm_sanitizers.sh 0 运行内存、竞态与同步诊断。长 K 混合幅度输入存在候选与固定参考的数值失败，不能由五个约定用例外推通用正确性。见 [覆盖报告](experiments/h20-20261116/COVERAGE.md)。

第 4 轮：BARE_GEMM_STABLE_LONG_K=1 显式启用长 K 分段累加，K≥16384 时每 256 项形成 WGMMA 部分和，再用普通 FP32 加法合并，总和最终只转换一次 BF16；默认关闭。全部 9 条路径的 3024 项长 K 组合候选通过，固定参考仍有 126 项失败。见 [验证报告](experiments/qjh004-20261116/REPORT.md) 和 [诊断计时](experiments/qjh004-20261116/SPEED.md)。

