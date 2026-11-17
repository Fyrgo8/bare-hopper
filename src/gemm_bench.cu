#include "candidate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t s) {
  if (s != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(s));
}
static void cb(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(s));
}
template <class T> struct Device {
  T *ptr = nullptr;
  explicit Device(size_t n) {
    ck(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
  }
  ~Device() { cudaFree(ptr); }
  Device(Device const &) = delete;
  Device &operator=(Device const &) = delete;
};
struct Stream {
  cudaStream_t value;
  Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value;
  Event() { ck(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
struct Blas {
  cublasHandle_t value;
  explicit Blas(cudaStream_t s) {
    cb(cublasCreate(&value));
    cb(cublasSetStream(value, s));
    cb(cublasSetMathMode(value, CUBLAS_PEDANTIC_MATH));
  }
  ~Blas() { cublasDestroy(value); }
};
struct Options {
  std::string cases = "configs/gemm.csv", output = "samples.csv",
              cache = "rotating", pattern = "dyadic", oracle = "fp32";
  int rounds = 30, iterations = 64, warmup = 32;
  uint32_t seed = 20261115;
  int validation_repeats = 3;
  bool compare = false, describe = false, validate_only = false;
};
static Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help") {
      std::cout
          << "gemm_bench [--compare] [--describe] [--validate-only] [--cases FILE] [--output "
             "FILE]\n"
             "  [--rounds 30] [--iterations 64] [--warmup 32] [--seed "
             "20261115]\n"
             "  [--cache rotating|hot]\n"
             "  [--pattern dyadic|uniform|normal|logscale|zero|ones|identity|cancellation]\n"
             "  [--validation-repeats 3] (validate-only; refresh inputs each time)\n"
             "  [--oracle fp32|fp64] (fp64 requires validate-only)\n"
             "BF16 A=row(M,K), B=column(K,N), D=row(M,N); FP32 accumulation.\n"
             "Reference-only is the default; --compare requires a real "
             "candidate.\n"
             "CUDA events, ordinary launches, alternating AB/BA batches, no "
             "graph.\n"
             "Iterations is a lower bound: batches are calibrated to >=250ms "
             "on reference.\n"
             "Rotating input ring exceeds 4x L2 (at least two slots); hot uses "
             "one.\n"
             "Every output element is checked against independent pedantic "
             "FP32 SGEMM\n"
             "on exactly BF16-rounded inputs: abs_error <= 0.02 + "
             "0.01*abs(expected).\n"
             "Use scripts/run_bench.py for clock evidence, telemetry and "
             "verdicts.\n";
      std::exit(0);
    }
    if (a == "--compare") {
      o.compare = true;
      continue;
    }
    if (a == "--describe") {
      o.describe = true;
      continue;
    }
    if (a == "--validate-only") {
      o.validate_only = true;
      continue;
    }
    if (i + 1 >= argc)
      throw std::runtime_error("Missing value for " + a);
    std::string v = argv[++i];
    if (a == "--cases")
      o.cases = v;
    else if (a == "--output")
      o.output = v;
    else if (a == "--cache")
      o.cache = v;
    else if (a == "--pattern")
      o.pattern = v;
    else if (a == "--oracle")
      o.oracle = v;
    else if (a == "--validation-repeats")
      o.validation_repeats = std::stoi(v);
    else if (a == "--rounds")
      o.rounds = std::stoi(v);
    else if (a == "--iterations")
      o.iterations = std::stoi(v);
    else if (a == "--warmup")
      o.warmup = std::stoi(v);
    else if (a == "--seed")
      o.seed = static_cast<uint32_t>(std::stoul(v));
    else
      throw std::runtime_error("Unknown option " + a);
  }
  if (o.rounds < 10 || o.iterations < 1 || o.warmup < 1 ||
      o.validation_repeats < 1 ||
      (o.cache != "rotating" && o.cache != "hot"))
    throw std::runtime_error("Invalid timing options");
  if (o.pattern != "dyadic" && o.pattern != "uniform" && o.pattern != "normal" &&
      o.pattern != "logscale" && o.pattern != "zero" && o.pattern != "ones" &&
      o.pattern != "identity" && o.pattern != "cancellation")
    throw std::runtime_error("Unknown input pattern: " + o.pattern);
  if (o.pattern != "dyadic" && !o.validate_only)
    throw std::runtime_error("Extra input patterns require --validate-only");
  if (o.oracle != "fp32" && o.oracle != "fp64")
    throw std::runtime_error("Unknown oracle: " + o.oracle);
  if (o.oracle == "fp64" && !o.validate_only)
    throw std::runtime_error("FP64 oracle requires --validate-only");
  return o;
}
struct Case {
  std::string id;
  bare::Problem p;
};
static std::vector<Case> read_cases(std::string const &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot open cases: " + path);
  std::string line;
  std::getline(file, line);
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  if (line != "id,m,n,k")
    throw std::runtime_error("Expected CSV header id,m,n,k");
  std::vector<Case> result;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    Case c;
    std::string extra;
    if (!(row >> c.id >> c.p.m >> c.p.n >> c.p.k) || row >> extra)
      throw std::runtime_error("Invalid case row: " + line);
    if (c.p.m <= 0 || c.p.n <= 0 || c.p.k <= 0 || c.p.n % 8 || c.p.k % 8)
      throw std::runtime_error(
          "Positive sizes and N/K divisible by 8 required");
    for (auto const &old : result)
      if (old.id == c.id)
        throw std::runtime_error("Duplicate case id");
    result.push_back(c);
  }
  if (result.empty())
    throw std::runtime_error("Empty case list");
  return result;
}
struct Slot {
  Device<__nv_bfloat16> a, b, ref, out;
  std::unique_ptr<bare::Reference> reference;
  void *candidate = nullptr;
  bool prepared = false;
  Slot(bare::Problem p)
      : a(size_t(p.m) * p.k), b(size_t(p.k) * p.n), ref(size_t(p.m) * p.n),
        out(size_t(p.m) * p.n) {}
  ~Slot() {
    if (prepared)
      bare::candidate::cleanup(candidate);
  }
};
static void fill(Device<__nv_bfloat16> &device, size_t n, std::mt19937 &rng,
                 Options const &o, bare::Problem p, bool is_a) {
  std::vector<__nv_bfloat16> host(n);
  // Integer distribution makes all initial values exactly representable in
  // BF16.
  std::uniform_int_distribution<int> dist(-64, 64);
  std::uniform_real_distribution<float> uniform(-1.f, 1.f);
  std::normal_distribution<float> normal(0.f, 0.5f);
  std::uniform_int_distribution<int> exponent(-3, 3);
  for (size_t i = 0; i < n; ++i) {
    float x = float(dist(rng)) / 128.f;
    int row = int(i / p.k), k = int(i % p.k);
    if (o.pattern == "uniform") x = uniform(rng);
    else if (o.pattern == "normal") x = normal(rng);
    else if (o.pattern == "logscale") x = std::ldexp(uniform(rng), exponent(rng));
    else if (o.pattern == "zero") x = 0;
    else if (o.pattern == "ones") x = 1;
    else if (o.pattern == "identity" && is_a) x = row == k ? 1.f : 0.f;
    else if (o.pattern == "cancellation") {
      // Adjacent A values are identical; adjacent B values are exact opposites.
      // K is even, so the exact matrix product is zero.
      int code = (row * 17 + (k / 2) * 13 + int(o.seed % 127)) % 65 - 32;
      x = float(code) / 64.f;
      if (!is_a && k % 2) x = -x;
    }
    host[i] = __float2bfloat16_rn(x);
  }
  ck(cudaMemcpy(device.ptr, host.data(), n * sizeof(host[0]),
                cudaMemcpyHostToDevice));
}
__global__ void to_float(__nv_bfloat16 const *src, float *dst, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    dst[i] = __bfloat162float(src[i]);
}
__global__ void to_double(__nv_bfloat16 const *src, double *dst, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    dst[i] = double(__bfloat162float(src[i]));
}
struct Error {
  bool ok = true;
  double max_scaled = 0;
};
static Error validate(__nv_bfloat16 const *output,
                      std::vector<float> const &expected) {
  std::vector<__nv_bfloat16> actual(expected.size());
  ck(cudaMemcpy(actual.data(), output, actual.size() * sizeof(actual[0]),
                cudaMemcpyDeviceToHost));
  Error e;
  size_t worst = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double x = __bfloat162float(actual[i]), y = expected[i];
    if (!std::isfinite(x) || !std::isfinite(y)) {
      e.ok = false;
      e.max_scaled = 1e30;
      continue;
    }
    double scaled = std::abs(x - y) / (0.02 + 0.01 * std::abs(y));
    if (scaled > e.max_scaled) {
      e.max_scaled = scaled;
      worst = i;
    }
    if (scaled > 1)
      e.ok = false;
  }
  if (!e.ok)
    std::cerr << "validation mismatch: index=" << worst
              << ", actual=" << std::setprecision(10) << __bfloat162float(actual[worst])
              << ", expected=" << expected[worst]
              << ", scaled_error=" << e.max_scaled << '\n';
  return e;
}
static float time_batch(std::function<void(int)> const &launch, int count,
                        cudaStream_t s) {
  Event start, end;
  ck(cudaEventRecord(start.value, s));
  for (int i = 0; i < count; ++i)
    launch(i);
  ck(cudaEventRecord(end.value, s));
  ck(cudaEventSynchronize(end.value));
  float ms = 0;
  ck(cudaEventElapsedTime(&ms, start.value, end.value));
  if (!(ms > 0) || !std::isfinite(ms))
    throw std::runtime_error("Invalid event timing");
  return ms / count;
}
static int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
static bool run_case(Case const &c, Options const &o,
                     cudaDeviceProp const &prop, std::ofstream &csv) {
  auto p = c.p;
  Stream stream;
  Blas blas(stream.value);
  size_t inputs =
      (size_t(p.m) * p.k + size_t(p.k) * p.n) * sizeof(__nv_bfloat16);
  size_t ring =
      o.cache == "hot"
          ? 1
          : std::max<size_t>(2, (4ULL * prop.l2CacheSize) / inputs + 1);
  if (ring > 128)
    throw std::runtime_error(
        "Tiny problem requires excessive cache ring; use --cache hot");
  int count = static_cast<int>(((o.iterations + ring - 1) / ring) * ring);
  std::cerr << c.id << ": ring=" << ring << ", iterations=" << count << "\n";
  std::vector<std::unique_ptr<Slot>> slots;
  std::mt19937 rng(o.seed);
  for (size_t i = 0; i < ring; ++i) {
    auto s = std::make_unique<Slot>(p);
    fill(s->a, size_t(p.m) * p.k, rng, o, p, true);
    fill(s->b, size_t(p.k) * p.n, rng, o, p, false);
    s->reference = std::make_unique<bare::Reference>(
        p, bare::Buffers{s->a.ptr, s->b.ptr, s->ref.ptr}, stream.value);
    if (o.compare) {
      s->candidate = bare::candidate::prepare(
          p, {s->a.ptr, s->b.ptr, s->out.ptr}, stream.value);
      s->prepared = true;
    }
    slots.push_back(std::move(s));
  }
  auto ref = [&](int i) { slots[i % ring]->reference->launch(stream.value); };
  auto own = [&](int i) {
    auto &s = *slots[i % ring];
    bare::candidate::launch(s.candidate, stream.value);
  };
  Error re, ce;
  auto verify = [&](bool launch_fresh) {
    Device<float> af(size_t(p.m) * p.k), bf(size_t(p.k) * p.n),
        gold(size_t(p.m) * p.n);
    std::vector<float> expected(size_t(p.m) * p.n);
    for (auto const &s : slots) {
      to_float<<<256, 256, 0, stream.value>>>(s->a.ptr, af.ptr,
                                              size_t(p.m) * p.k);
      to_float<<<256, 256, 0, stream.value>>>(s->b.ptr, bf.ptr,
                                              size_t(p.k) * p.n);
      ck(cudaGetLastError());
      float alpha = 1, beta = 0;
      // Column-major D^T(N,M) = B^T(N,K) * A^T(K,M).
      cb(cublasSgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k,
                     &alpha, bf.ptr, p.k, af.ptr, p.k, &beta, gold.ptr, p.n));
      ck(cudaMemcpyAsync(expected.data(), gold.ptr,
                         expected.size() * sizeof(float),
                         cudaMemcpyDeviceToHost, stream.value));
      ck(cudaStreamSynchronize(stream.value));
      if (o.oracle == "fp64") {
        // FP32 SGEMM can accumulate enough rounding error on long K to
        // misclassify near-zero outputs. Use independent DGEMM for coverage;
        // the timed protocol continues to use its original FP32 oracle.
        Device<double> ad(size_t(p.m) * p.k), bd(size_t(p.k) * p.n),
            gd(expected.size());
        to_double<<<256, 256, 0, stream.value>>>(s->a.ptr, ad.ptr, size_t(p.m) * p.k);
        to_double<<<256, 256, 0, stream.value>>>(s->b.ptr, bd.ptr, size_t(p.k) * p.n);
        ck(cudaGetLastError());
        double da = 1, db = 0;
        cb(cublasDgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k,
                       &da, bd.ptr, p.k, ad.ptr, p.k, &db, gd.ptr, p.n));
        std::vector<double> exact(expected.size());
        ck(cudaMemcpyAsync(exact.data(), gd.ptr, exact.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream.value));
        ck(cudaStreamSynchronize(stream.value));
        double fp32_max_abs = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
          fp32_max_abs = std::max(fp32_max_abs, std::abs(double(expected[i]) - exact[i]));
          expected[i] = float(exact[i]);
        }
        std::cerr << c.id << ": fp32_oracle_max_abs_error=" << fp32_max_abs << '\n';
      }
      // Exact structured oracles also verify the independent SGEMM result.
      if (o.pattern == "zero" || o.pattern == "cancellation")
        for (float y : expected)
          if (y != 0) throw std::runtime_error("Exact zero oracle failed: " + c.id);
      if (o.pattern == "ones")
        for (float y : expected)
          if (y != float(p.k)) throw std::runtime_error("Exact ones oracle failed: " + c.id);
      if (o.pattern == "identity") {
        std::vector<__nv_bfloat16> host_b(size_t(p.n) * p.k);
        ck(cudaMemcpy(host_b.data(), s->b.ptr, host_b.size() * sizeof(host_b[0]),
                      cudaMemcpyDeviceToHost));
        for (int row = 0; row < p.m; ++row)
          for (int col = 0; col < p.n; ++col) {
            float exact = row < p.k ? __bfloat162float(host_b[size_t(col) * p.k + row]) : 0;
            if (expected[size_t(row) * p.n + col] != exact)
              throw std::runtime_error("Exact identity oracle failed: " + c.id);
          }
      }
      if (launch_fresh) {
        ck(cudaMemsetAsync(s->ref.ptr, 0xff,
                           expected.size() * sizeof(__nv_bfloat16),
                           stream.value));
        s->reference->launch(stream.value);
        if (o.compare) {
          ck(cudaMemsetAsync(s->out.ptr, 0xff,
                             expected.size() * sizeof(__nv_bfloat16),
                             stream.value));
          bare::candidate::launch(s->candidate, stream.value);
        }
      }
      ck(cudaStreamSynchronize(stream.value));
      auto r = validate(s->ref.ptr, expected);
      re.ok &= r.ok;
      re.max_scaled = std::max(re.max_scaled, r.max_scaled);
      if (o.compare) {
        auto e = validate(s->out.ptr, expected);
        ce.ok &= e.ok;
        ce.max_scaled = std::max(ce.max_scaled, e.max_scaled);
      }
    }
    if ((!re.ok || (o.compare && !ce.ok)) && !o.validate_only)
      throw std::runtime_error("Numerical validation failed: " + c.id);
  };
  verify(true);
  if (o.validate_only) {
    for (int repeat = 1; repeat < o.validation_repeats; ++repeat) {
      for (auto const &s : slots) {
        fill(s->a, size_t(p.m) * p.k, rng, o, p, true);
        fill(s->b, size_t(p.k) * p.n, rng, o, p, false);
      }
      verify(true);
    }
    csv << c.id << ',' << p.m << ',' << p.n << ',' << p.k << ',' << o.pattern
        << ',' << o.seed << ',' << o.validation_repeats << ',' << o.oracle << ',' << re.ok << ','
        << (o.compare ? (ce.ok ? "1" : "0") : "") << ','
        << re.max_scaled << ',' << (o.compare ? ce.max_scaled : 0) << '\n';
    csv.flush();
    bool ok = re.ok && (!o.compare || ce.ok);
    std::cerr << c.id << (ok ? ": validation passed" : ": validation FAILED")
              << ", reference_max_scaled_error="
              << re.max_scaled << ", candidate_max_scaled_error="
              << ce.max_scaled << "\n";
    return ok;
  }
  for (int i = 0; i < std::max<int>(o.warmup, ring); ++i) {
    ref(i);
    if (o.compare)
      own(i);
  }
  ck(cudaStreamSynchronize(stream.value));
  // Calibration is part of this run's warmup, outside recorded paired samples.
  float estimate = time_batch(ref, count, stream.value);
  size_t target = static_cast<size_t>(std::ceil(250.0 / estimate));
  count = static_cast<int>(
      ((std::max<size_t>(count, target) + ring - 1) / ring) * ring);
  std::cerr << c.id << ": calibrated iterations=" << count << "\n";
  for (int r = 0; r < o.rounds; ++r) {
    int64_t start = now_ms();
    float rt = 0, ct = 0;
    if (o.compare && r % 2) {
      ct = time_batch(own, count, stream.value);
      rt = time_batch(ref, count, stream.value);
    } else {
      rt = time_batch(ref, count, stream.value);
      if (o.compare)
        ct = time_batch(own, count, stream.value);
    }
    csv << c.id << ',' << r << ',' << (o.compare ? (r % 2 ? "BA" : "AB") : "A")
        << ',' << rt << ',';
    if (o.compare)
      csv << ct;
    csv << ',' << re.ok << ',' << (o.compare ? (ce.ok ? "1" : "0") : "") << ','
        << re.max_scaled << ',';
    if (o.compare)
      csv << ce.max_scaled;
    csv << ',' << ring << ',' << count << ',' << start << ',' << now_ms()
        << '\n';
    csv.flush();
  }
  ck(cudaStreamSynchronize(stream.value));
  verify(false); // Check actual timed outputs, without repairing them by
                 // relaunching.
  return true;
}
int main(int argc, char **argv) {
  try {
    auto o = parse(argc, argv);
    ck(cudaSetDevice(0));
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 9 || prop.minor != 0)
      throw std::runtime_error("SM90 Hopper GPU required");
    if (o.describe) {
      int driver = 0, runtime = 0;
      ck(cudaDriverGetVersion(&driver));
      ck(cudaRuntimeGetVersion(&runtime));
      std::cout << "{\"gpu\":\"" << prop.name
                << "\",\"sm_count\":" << prop.multiProcessorCount
                << ",\"l2_bytes\":" << prop.l2CacheSize
                << ",\"driver\":" << driver << ",\"runtime\":" << runtime
                << ",\"nvcc\":\"" << __CUDACC_VER_MAJOR__ << '.'
                << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
                << "\",\"reference\":\"" << bare::Reference::name()
                << "\",\"stages\":" << bare::Reference::stages()
                << ",\"cutlass_revision\":\"" << BARE_CUTLASS_REVISION
                << "\",\"candidate\":\"" << bare::candidate::name()
                << "\",\"candidate_available\":"
                << (bare::candidate::available() ? "true" : "false") << "}\n";
      return 0;
    }
    if (o.compare && !bare::candidate::available())
      throw std::runtime_error(
          "--compare requires a real candidate; stub cannot pass");
    auto cases = read_cases(o.cases);
    std::ofstream csv(o.output);
    if (!csv)
      throw std::runtime_error("Cannot open output file");
    csv << std::setprecision(10);
    if (o.validate_only)
      csv << "case,m,n,k,pattern,seed,repeats,oracle,reference_ok,candidate_ok,"
             "reference_max_scaled_error,candidate_max_scaled_error\n";
    else csv << "case,round,order,reference_ms,candidate_ms,reference_ok,candidate_"
           "ok,reference_max_scaled_error,candidate_max_scaled_error,ring,"
           "iterations,start_unix_ms,end_unix_ms\n";
    bool all_ok = true;
    for (auto const &c : cases)
      all_ok &= run_case(c, o, prop, csv);
    if (!csv)
      throw std::runtime_error("Writing samples failed");
    return all_ok ? 0 : 1;
  } catch (std::exception const &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
