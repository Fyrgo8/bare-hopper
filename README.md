# bare-hopper

用 CUDA C++ 和 PTX 手写 Hopper GPU 算子，不依赖 CUTLASS/CuTe，通过复现高性能实现锻炼算子开发基本功。

从 BF16 GEMM 开始，逐步复现 FlashAttention-3 的前向和反向。在 H20 上保证数值正确，并以每个约定测试用例的耗时不超过参考实现的 103% 为目标。
