#pragma once
#include "gemm.hpp"

namespace bare::candidate {
// Implement these functions in your CUDA/PTX source. Setup is outside timing;
// launch must enqueue ALL computation on stream, without allocating or syncing.
// Inputs are immutable; output must be completely overwritten on every launch.
// Candidate gets no reference/oracle output, and no CUTLASS include directory.
bool available();
char const *name();
void *prepare(Problem, Buffers, cudaStream_t);
void launch(void *context, cudaStream_t);
void cleanup(void *context);
} // namespace bare::candidate
