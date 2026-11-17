#include "candidate.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Compare the two reduction schedules on identical inputs and prepared
// contexts. This complements FP64 tolerance checks with exact BF16 equality.
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  std::ifstream cases(argv[1]);
  if (!cases) return 2;
  std::string line;
  std::getline(cases, line);
  size_t calls = 0, elements = 0;
  std::puts("case,stage,ws,pattern,repeat,elements,mismatches");
  while (std::getline(cases, line)) {
    std::stringstream row(line);
    std::string id, field;
    std::getline(row, id, ',');
    int dims[3];
    for (int &dim : dims) {
      std::getline(row, field, ',');
      dim = std::stoi(field);
    }
    bare::Problem p{dims[0], dims[1], dims[2]};
    size_t ac = size_t(p.m) * p.k, bc = size_t(p.n) * p.k;
    size_t dc = size_t(p.m) * p.n;
    __nv_bfloat16 *a, *b, *d1, *d16;
    check(cudaMalloc(&a, ac * 2));
    check(cudaMalloc(&b, bc * 2));
    check(cudaMalloc(&d1, dc * 2));
    check(cudaMalloc(&d16, dc * 2));
    std::vector<__nv_bfloat16> input(ac + bc), out1(dc), out16(dc);
    for (int stage = 1; stage <= 4; ++stage) {
      for (int ws = 0; ws <= 1; ++ws) {
        setenv("BARE_GEMM_STAGES", std::to_string(stage).c_str(), 1);
        setenv("BARE_GEMM_WARP_SPECIALIZED", std::to_string(ws).c_str(), 1);
        setenv("BARE_GEMM_REDUCE_UNROLL", "1", 1);
        void *c1 = bare::candidate::prepare(p, {a, b, d1}, nullptr);
        setenv("BARE_GEMM_REDUCE_UNROLL", "16", 1);
        void *c16 = bare::candidate::prepare(p, {a, b, d16}, nullptr);
        for (int pattern = 0; pattern < 3; ++pattern) {
          for (int repeat = 0; repeat < 3; ++repeat) {
            uint32_t state = 17 + repeat * 99173;
            for (auto &value : input) {
              state ^= state << 13;
              state ^= state >> 17;
              state ^= state << 5;
              float x = (float(state & 65535) / 32768.0f - 1.0f);
              if (pattern == 1) x *= std::ldexp(1.0f, int(state >> 27) % 13 - 6);
              if (pattern == 2) x = float(int(state & 15) - 8) / 8.0f;
              value = __float2bfloat16_rn(x);
            }
            check(cudaMemcpy(a, input.data(), ac * 2, cudaMemcpyHostToDevice));
            check(cudaMemcpy(b, input.data() + ac, bc * 2, cudaMemcpyHostToDevice));
            check(cudaMemset(d1, 0x7f, dc * 2));
            check(cudaMemset(d16, 0x7f, dc * 2));
            bare::candidate::launch(c1, nullptr);
            bare::candidate::launch(c16, nullptr);
            check(cudaMemcpy(out1.data(), d1, dc * 2, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(out16.data(), d16, dc * 2, cudaMemcpyDeviceToHost));
            size_t mismatch = 0;
            for (size_t i = 0; i < dc; ++i)
              mismatch += std::memcmp(&out1[i], &out16[i], 2) != 0;
            std::printf("%s,%d,%d,%d,%d,%zu,%zu\n", id.c_str(), stage, ws,
                        pattern, repeat, dc, mismatch);
            if (mismatch) return 1;
            ++calls;
            elements += dc;
          }
        }
        bare::candidate::cleanup(c1);
        bare::candidate::cleanup(c16);
      }
    }
    check(cudaFree(a));
    check(cudaFree(b));
    check(cudaFree(d1));
    check(cudaFree(d16));
  }
  std::fprintf(stderr, "PASS: %zu output pairs, %zu BF16 elements identical\n",
               calls, elements);
}
