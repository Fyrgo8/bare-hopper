#!/usr/bin/env bash
set -euo pipefail
cd /home/fengyunhao/bare-hopper-qjh004-20261116
base=results/tile128-20261117
while python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(x["status"]!="running")' \
  "$base/final-validation/manifest.json"; do
  sleep 10
done
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="pass"' \
  "$base/final-validation/manifest.json"
# All correctness workers must have exited before performance collection.
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
BARE_GEMM_REDUCE_UNROLL=1 python3 "$base/measure-final.py" \
  --binary "$base/final-bench" --out "$base/final-control" \
  --rounds 30 --modes auto:0 --split 1
env -u BARE_GEMM_REDUCE_UNROLL python3 "$base/measure-final.py" \
  --binary "$base/final-bench" --out "$base/final-speed" \
  --rounds 30 --modes auto:0 --split 1
BARE_GEMM_REDUCE_UNROLL=1 python3 "$base/measure-final.py" \
  --binary "$base/final-bench" --out "$base/final-control-repeat" \
  --rounds 30 --modes auto:0 --split 1 --cases "$base/long-cases.csv"
