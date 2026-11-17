#!/usr/bin/env bash
set -euo pipefail
root=/home/fengyunhao/bare-hopper-qjh004-20261116
cd "$root"
base=results/tile128-20261117
gpu=${1:-0}
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
mkdir "$base/sanitizers"
printf 'tool,mode,exit_code\n' > "$base/sanitizers/status.csv"
for tool in memcheck racecheck synccheck; do
  for mode in 1:0 2:0 3:0 4:0 1:1 2:1 3:1 4:1; do
    stages=${mode%:*}
    ws=${mode#*:}
    label="$tool-s$stages-ws$ws"
    status=0
    CUDA_VISIBLE_DEVICES=$gpu BARE_GEMM_STAGES=$stages \
      BARE_GEMM_WARP_SPECIALIZED=$ws BARE_GEMM_REDUCE_UNROLL=16 \
      timeout 600 /usr/local/cuda-12.8/bin/compute-sanitizer \
      --tool "$tool" --error-exitcode 86 "$base/final-bench" \
      --compare --validate-only --cache hot --cases "$base/sanitizer-cases.csv" \
      --pattern uniform --seed 17 --validation-repeats 2 --oracle fp64 \
      --output "$base/sanitizers/$label.csv" \
      > "$base/sanitizers/$label.log" 2>&1 || status=$?
    printf '%s,%s,%s\n' "$tool" "$mode" "$status" \
      >> "$base/sanitizers/status.csv"
    if [[ $status != 0 ]]; then exit "$status"; fi
  done
done
