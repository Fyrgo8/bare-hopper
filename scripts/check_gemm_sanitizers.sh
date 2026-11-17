#!/usr/bin/env bash
# Correctness diagnostics only. Never changes clocks, services, or GPU state.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
gpu=${1:-0}
sanitizer=${COMPUTE_SANITIZER:-/usr/local/cuda-12.8/bin/compute-sanitizer}
processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$processes" ]]; then
    echo "Selected GPU has compute processes; no kernels launched" >&2
    exit 1
fi
out="$root/results/sanitizers-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$out"
cp build/gemm_bench "$out/gemm_bench"
sha256sum "$out/gemm_bench" > "$out/SHA256SUMS"
cat > "$out/cases.csv" <<'CASES'
id,m,n,k
tiny,1,8,8
before_tile,63,120,56
tile,64,128,64
after_tile,65,136,72
wrap3,127,248,200
wrap4,129,264,264
wrap8,65,136,520
CASES
nvidia-smi -i "$gpu" -q > "$out/gpu-before.txt"
printf 'tool,stages,warp_specialized,exit_code\n' > "$out/status.csv"
failures=0
for tool in memcheck racecheck synccheck; do
    for mode in '0 0' '1 0' '2 0' '3 0' '4 0' '1 1' '2 1' '3 1' '4 1'; do
        read -r stages ws <<< "$mode"
        label="$tool-s$stages-ws$ws"
        status=0
        CUDA_VISIBLE_DEVICES="$gpu" BARE_GEMM_STAGES="$stages" \
            BARE_GEMM_WARP_SPECIALIZED="$ws" \
            timeout 180 "$sanitizer" --tool "$tool" --error-exitcode 86 \
            "$out/gemm_bench" --compare --validate-only --cache hot \
            --cases "$out/cases.csv" --pattern uniform --seed 99173 \
            --validation-repeats 3 --output "$out/$label.csv" \
            > "$out/$label.log" 2>&1 || status=$?
        printf '%s,%s,%s,%s\n' "$tool" "$stages" "$ws" "$status" >> "$out/status.csv"
        if [[ "$status" -ne 0 ]]; then failures=$((failures + 1)); fi
        echo "$label: exit_code=$status"
    done
done
nvidia-smi -i "$gpu" -q > "$out/gpu-after.txt"
echo "directory=$out failed_jobs=$failures"
[[ "$failures" -eq 0 ]]
