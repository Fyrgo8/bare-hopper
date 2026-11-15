#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e
if [[ ! -d .deps/cutlass ]]; then
  mkdir -p .deps
  git clone --depth 1 --branch v3.9.2 git@github.com:NVIDIA/cutlass.git .deps/cutlass
fi
test "$(git -C .deps/cutlass rev-parse HEAD)" = "$revision"
test -z "$(git -C .deps/cutlass status --porcelain --untracked-files=no)"
echo "CUTLASS v3.9.2 pinned at $revision"
echo 'Build: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 4'
