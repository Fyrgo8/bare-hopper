#!/usr/bin/env bash
# Run inside the dedicated experiment checkout on qjh000, as an authorized admin.
# Only GPU 0 is exposed; host PID namespace keeps nvidia-smi PID checks valid.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=GPU-d1b595ce-0b57-cdee-c1c0-9b1051aaa07e
image=2cffe13200ba6a5db0f1984b46a0a176fd8d6c6824d1de1edf67f1cd05c37b9e
stamp=$(date -u +%Y%m%dT%H%M%SZ)
evidence="results/validation-$stamp"
mkdir -p "$evidence"
docker image inspect "$image" > "$evidence/image.json"
cp build/gemm_bench "$evidence/gemm_bench"
tar -czf "$evidence/source.tgz" src include scripts configs CMakeLists.txt
sha256sum "$evidence/gemm_bench" "$evidence/source.tgz" > "$evidence/SHA256SUMS"
base=(docker run --rm --pid host --gpus "device=$gpu" --cap-add SYS_ADMIN
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/work
  -e LD_LIBRARY_PATH=/opt/cuda128/lib64:/usr/local/nvidia/lib64
  -v /usr/local/cuda-12.8:/opt/cuda128:ro
  -v "$PWD:/work" -w /work --entrypoint /bin/bash)
binary="/work/$evidence/gemm_bench"
run_timing() {
  local label=$1 stages=$2 ws=$3
  local rc=0
  "${base[@]}" -e BARE_GEMM_STAGES="$stages" -e BARE_GEMM_WARP_SPECIALIZED="$ws" \
    "$image" -c "python scripts/run_bench.py --binary '$binary' --sm-clock 1800 --memory-clock 2619 --compare" \
    > "$evidence/$label.log" 2>&1 || rc=$?
  echo "$label exit_code=$rc" | tee -a "$evidence/exits.txt"
  # A performance failure is a measured result; an execution error stops the run.
  if [[ $rc != 0 && $rc != 2 ]]; then return "$rc"; fi
}
run_timing vector 0 0
run_timing tma1 1 0
run_timing ws2 2 1
{ cat configs/gemm.csv; tail -n +2 configs/gemm_edges.csv; } > "$evidence/correctness-cases.csv"
for mode in vector tma1 tma2 ws2; do
  stages=2; ws=0
  [[ $mode != vector ]] || stages=0
  [[ $mode != tma1 ]] || stages=1
  [[ $mode != ws2 ]] || ws=1
  "${base[@]}" -e BARE_GEMM_STAGES="$stages" -e BARE_GEMM_WARP_SPECIALIZED="$ws" \
    "$image" -c "/opt/cuda128/bin/compute-sanitizer --tool memcheck --error-exitcode 3 '$binary' --compare --validate-only --cache hot --cases /work/$evidence/correctness-cases.csv --output /work/$evidence/$mode-validation.csv" \
    > "$evidence/$mode-memcheck.log" 2>&1
done
for mode in tma2 ws2; do
  ws=0
  [[ $mode != ws2 ]] || ws=1
  "${base[@]}" -e BARE_GEMM_STAGES=2 -e BARE_GEMM_WARP_SPECIALIZED="$ws" \
    "$image" -c "/opt/cuda128/bin/compute-sanitizer --tool racecheck --error-exitcode 3 '$binary' --compare --validate-only --cache hot --cases /work/configs/gemm_edges.csv --output /work/$evidence/$mode-racecheck.csv" \
    > "$evidence/$mode-racecheck.log" 2>&1
done
for mode in vector tma1 tma2 ws2; do
  stages=2; ws=0
  [[ $mode != vector ]] || stages=0
  [[ $mode != tma1 ]] || stages=1
  [[ $mode != ws2 ]] || ws=1
  "${base[@]}" -e BARE_GEMM_STAGES="$stages" -e BARE_GEMM_WARP_SPECIALIZED="$ws" \
    "$image" -c "/opt/cuda128/bin/ncu --kernel-name 'regex:vector_gemm|tma_gemm' --launch-count 1 --section SpeedOfLight --csv '$binary' --compare --validate-only --cache hot --output /work/$evidence/$mode-profile-validation.csv" \
    > "$evidence/$mode-profile.csv" 2>&1
done
echo "Completed: $evidence"
