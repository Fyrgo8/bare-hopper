# bare-hopper

A hands-on project to learn GPU kernel fundamentals by reimplementing Hopper kernels in CUDA C++ and PTX, without CUTLASS/CuTe dependencies.

Starting with BF16 GEMM, then FlashAttention-3 forward and backward. On H20, the goal is numerical correctness and latency no more than 3% above the reference implementation for each agreed benchmark case.
