#!/usr/bin/env bash
set -euo pipefail
cd /home/fengyunhao/bare-hopper-qjh004-20261116
gpu=$1
tool=$2
out=results/group-$tool
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
mkdir "$out"
printf 'mode,exit_code\n' > "$out/status.csv"
for mode in 1:0 2:0 3:0 4:0 1:1 2:1 3:1 4:1; do
    stages=${mode%:*}
    ws=${mode#*:}
    status=0
    CUDA_VISIBLE_DEVICES=$gpu BARE_GEMM_STAGES=$stages BARE_GEMM_WARP_SPECIALIZED=$ws \
        BARE_GEMM_GROUP_M=8 timeout 300 /usr/local/cuda-12.8/bin/compute-sanitizer \
        --tool "$tool" --error-exitcode 86 results/group-final-build/gemm_bench \
        --compare --validate-only --cache hot --cases group-cases.csv \
        --pattern uniform --seed 17 --validation-repeats 1 --oracle fp64 \
        --output "$out/s$stages-ws$ws.csv" > "$out/s$stages-ws$ws.log" 2>&1 || status=$?
    printf '%s,%s\n' "$mode" "$status" >> "$out/status.csv"
done
