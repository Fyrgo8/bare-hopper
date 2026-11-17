#!/usr/bin/env bash
set -euo pipefail
cd /home/fengyunhao/bare-hopper-qjh004-20261116
gpu=$1
shift
out=results/group-validation-$gpu
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
mkdir "$out"
printf 'mode,pattern,seed,exit_code\n' > "$out/status.csv"
for mode in "$@"; do
    stages=${mode%:*}
    ws=${mode#*:}
    for pattern in dyadic uniform normal logscale zero ones identity cancellation; do
        for seed in 17 99173; do
            label="s$stages-ws$ws-$pattern-$seed"
            status=0
            CUDA_VISIBLE_DEVICES=$gpu BARE_GEMM_STAGES=$stages \
                BARE_GEMM_WARP_SPECIALIZED=$ws BARE_GEMM_GROUP_M=8 \
                timeout 300 results/group-final-build/gemm_bench \
                --compare --validate-only --cache hot --cases group-cases.csv \
                --pattern "$pattern" --seed "$seed" --validation-repeats 3 \
                --oracle fp64 --output "$out/$label.csv" > "$out/$label.log" 2>&1 || status=$?
            printf '%s,%s,%s,%s\n' "$mode" "$pattern" "$seed" "$status" >> "$out/status.csv"
        done
    done
done
