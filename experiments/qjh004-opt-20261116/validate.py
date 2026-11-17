"""Run immutable correctness suites on dedicated GPU workers (Python 3.10+)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import queue
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict


class Job(TypedDict):
    label: str
    gpu: str
    argv: list[str]
    exit_code: int
    rows: int
    expected_rows: int
    complete: bool
    candidate_failures: int
    reference_failures: int
    candidate_max_scaled_error: float


PATTERNS = (
    "dyadic",
    "uniform",
    "normal",
    "logscale",
    "zero",
    "ones",
    "identity",
    "cancellation",
)
MODES = ((1, 0), (0, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1), (4, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gpus", default="3,4,5,6,7")
    parser.add_argument("--full-source", type=Path, required=True)
    parser.add_argument("--long-source", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.out / "gemm_bench"
    shutil.copy2(args.binary, binary)
    shutil.copy2(args.root / "src/candidate.cu", args.out / "candidate.cu")
    for suite, source in (("full", args.full_source), ("long", args.long_source)):
        for kind in ("broad", "alternative"):
            shutil.copy2(source / f"{kind}.csv", args.out / f"{suite}-{kind}.csv")
    extra = args.out / "dispatch.csv"
    extra.write_text(
        "id,m,n,k\nlarge,1024,1024,16384\nlarge_cap_fallback,1024,1024,16392\n"
        "small_cap,512,512,16384\nsmall_cap_fallback,512,512,32776\n"
        "tile_switch,513,512,16392\nn_tail,512,520,16384\n"
        "large_n_cap_fallback,1024,1032,16384\n"
    )
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu in args.gpus.split(","):
        if subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip():
            raise RuntimeError(f"GPU {gpu} busy")
        gpu_pool.put(gpu)
    jobs: list[Job] = []
    manifest: dict[str, object] = {
        "kind": "correctness_only",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "stable_long_k": "1",
        "split_k": "1",
        "gpus": args.gpus,
        "jobs": jobs,
        "status": "running",
        "repeats": 3,
    }
    tasks: list[tuple[str, int, int, str, int, Path]] = []
    for suite in ("full", "long", "dispatch"):
        for stages, ws in MODES:
            broad = suite != "full" or (stages, ws) == (1, 0)
            patterns = (
                PATTERNS
                if broad
                else ("uniform", "logscale", "identity", "cancellation")
            )
            seeds = (17, 20261115, 99173) if broad else (17, 99173)
            cases = (
                extra
                if suite == "dispatch"
                else args.out / f"{suite}-{'broad' if broad else 'alternative'}.csv"
            )
            for pattern, seed in itertools.product(patterns, seeds):
                tasks.append((suite, stages, ws, pattern, seed, cases))

    def work(task: tuple[str, int, int, str, int, Path]) -> Job:
        suite, stages, ws, pattern, seed, cases = task
        label = f"{suite}-s{stages}-ws{ws}-{pattern}-{seed}"
        gpu = gpu_pool.get()
        try:
            argv = [
                str(binary),
                "--compare",
                "--validate-only",
                "--cache",
                "hot",
                "--cases",
                str(cases),
                "--pattern",
                pattern,
                "--seed",
                str(seed),
                "--validation-repeats",
                "3",
                "--oracle",
                "fp64",
                "--output",
                str(args.out / f"{label}.csv"),
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                BARE_GEMM_STAGES=str(stages),
                BARE_GEMM_WARP_SPECIALIZED=str(ws),
                BARE_GEMM_STABLE_LONG_K="1",
                BARE_GEMM_SPLIT_K="1",
            )
            with (args.out / f"{label}.log").open("w") as log:
                try:
                    code = subprocess.run(
                        argv, env=env, stdout=log, stderr=log, timeout=600
                    ).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            with cases.open() as handle:
                expected = list(csv.DictReader(handle))
            with (args.out / f"{label}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            complete = code in (0, 1) and [row["case"] for row in rows] == [
                row["id"] for row in expected
            ]
            return {
                "label": label,
                "gpu": gpu,
                "argv": argv,
                "exit_code": code,
                "rows": len(rows),
                "expected_rows": len(expected),
                "complete": complete,
                "candidate_failures": sum(row["candidate_ok"] != "1" for row in rows),
                "reference_failures": sum(row["reference_ok"] != "1" for row in rows),
                "candidate_max_scaled_error": max(
                    float(row["candidate_max_scaled_error"]) for row in rows
                ),
            }
        finally:
            gpu_pool.put(gpu)

    with ThreadPoolExecutor(max_workers=gpu_pool.qsize()) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            job = future.result()
            jobs.append(job)
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["status"] = (
        "complete" if all(job["complete"] for job in jobs) else "incomplete"
    )
    manifest["expected_jobs"] = len(tasks)
    manifest["candidate_failures"] = sum(int(job["candidate_failures"]) for job in jobs)
    manifest["reference_failures"] = sum(int(job["reference_failures"]) for job in jobs)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
