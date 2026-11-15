#include "candidate.hpp"
#include <stdexcept>

namespace bare::candidate {
bool available() { return false; }
char const *name() { return "unimplemented"; }
void *prepare(Problem, Buffers, cudaStream_t) {
  throw std::runtime_error("Implement src/candidate.cu before using --compare");
}
void launch(void *, cudaStream_t) {
  throw std::runtime_error("Candidate is not implemented");
}
void cleanup(void *) {}
} // namespace bare::candidate
