#!/usr/bin/env bash
set -euo pipefail
cd /home/fengyunhao/bare-hopper-qjh004-20261116
tool=$1
gpu=$2
out="results/opt-final-$tool"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
mkdir "$out"
cp results/opt-delivery-build/gemm_bench "$out/gemm_bench"
cat > "$out/cases.csv" <<'CASES'
id,m,n,k
split_boundary,65,136,16384
split_remainder,65,136,16648
split_tiny,1,8,65536
split_mn_tail,129,264,16392
CASES
printf 'mode,exit_code\n' > "$out/status.csv"
for mode in '1 0' '2 0' '3 0' '4 0' '1 1' '2 1' '3 1' '4 1'; do
    read -r stages ws <<< "$mode"
    label="s$stages-ws$ws"
    status=0
    CUDA_VISIBLE_DEVICES="$gpu" BARE_GEMM_STAGES="$stages" \
        BARE_GEMM_WARP_SPECIALIZED="$ws" BARE_GEMM_STABLE_LONG_K=1 \
        BARE_GEMM_SPLIT_K=1 timeout 300 \
        /usr/local/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 86 \
        "$out/gemm_bench" --compare --validate-only --cache hot \
        --cases "$out/cases.csv" --pattern uniform --seed 99173 \
        --validation-repeats 3 --oracle fp64 --output "$out/$label.csv" \
        > "$out/$label.log" 2>&1 || status=$?
    printf '%s,%s\n' "$label" "$status" >> "$out/status.csv"
done
test "$(awk -F, 'NR>1 && $2!=0 {n++} END {print n+0}' "$out/status.csv")" -eq 0
