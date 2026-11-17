"""FP64 checks of Split-K reduction tails and dispatch boundaries (Python 3.10+)."""

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


class Job(TypedDict):
    label: str
    gpu: str
    argv: list[str]
    env: dict[str, str]
    exit_code: int
    complete: bool
    checks: int
    candidate_failures: int
    reference_failures: int
    max_scaled_error: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--optimized-unroll", default="16")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.out.resolve() / "gemm_bench"
    shutil.copy2(args.binary, binary)
    cases = args.out.resolve() / "cases.csv"
    shutil.copy2(args.cases, cases)
    expected = [row["id"] for row in read_rows(cases)]
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu in args.gpus.split(","):
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if pids.strip():
            raise RuntimeError(f"GPU {gpu} busy; no work launched")
        gpu_pool.put(gpu)
    # Every TMA path can invoke the changed reduction. Include the unchanged
    # vector path and automatic dispatch as controls on the final binary.
    modes = [("auto", "0"), ("0", "0")]
    modes += [(str(stage), str(ws)) for stage in range(1, 5) for ws in (0, 1)]
    tasks = list(
        itertools.product(
            ("1", args.optimized_unroll),
            modes,
            PATTERNS,
            (17, 99173),
        )
    )
    jobs: list[Job] = []
    manifest: dict[str, object] = {
        "kind": "correctness_only",
        "oracle": "fp64",
        "repeats": 3,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "gpus": args.gpus,
        "expected_jobs": len(tasks),
        "jobs": jobs,
        "expected_cases": expected,
        "status": "running",
    }

    def work(task: tuple[str, tuple[str, str], str, int]) -> Job:
        unroll, (stage, ws), pattern, seed = task
        label = f"u{unroll}-s{stage}-ws{ws}-{pattern}-{seed}"
        gpu = gpu_pool.get()
        try:
            env = {
                k: v for k, v in os.environ.items() if not k.startswith("BARE_GEMM_")
            }
            settings = {
                "CUDA_VISIBLE_DEVICES": gpu,
                "BARE_GEMM_REDUCE_UNROLL": unroll,
                "BARE_GEMM_WARP_SPECIALIZED": ws,
            }
            if stage != "auto":
                settings["BARE_GEMM_STAGES"] = stage
            env.update(settings)
            output = args.out / f"{label}.csv"
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
                str(output),
            ]
            with (args.out / f"{label}.log").open("w", encoding="utf-8") as log:
                try:
                    code = subprocess.run(
                        argv,
                        env=env,
                        stdout=log,
                        stderr=log,
                        timeout=600,
                        check=False,
                    ).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            rows = read_rows(output) if output.exists() else []
            complete = code in (0, 1) and [r["case"] for r in rows] == expected
            complete = complete and all(
                r["oracle"] == "fp64"
                and r["pattern"] == pattern
                and int(r["seed"]) == seed
                and int(r["repeats"]) == 3
                for r in rows
            )
            return {
                "label": label,
                "gpu": gpu,
                "argv": argv,
                "env": settings,
                "exit_code": code,
                "complete": complete,
                "checks": len(rows),
                "candidate_failures": sum(r["candidate_ok"] != "1" for r in rows),
                "reference_failures": sum(r["reference_ok"] != "1" for r in rows),
                "max_scaled_error": max(
                    (float(r["candidate_max_scaled_error"]) for r in rows),
                    default=0,
                ),
            }
        finally:
            gpu_pool.put(gpu)

    with ThreadPoolExecutor(max_workers=gpu_pool.qsize()) as pool:
        for future in as_completed([pool.submit(work, task) for task in tasks]):
            jobs.append(future.result())
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["status"] = (
        "pass"
        if len(jobs) == len(tasks)
        and all(job["complete"] and not job["candidate_failures"] for job in jobs)
        else "fail"
    )
    manifest["candidate_failures"] = sum(job["candidate_failures"] for job in jobs)
    manifest["reference_failures"] = sum(job["reference_failures"] for job in jobs)
    manifest["checks"] = sum(job["checks"] for job in jobs)
    manifest["max_scaled_error"] = max(job["max_scaled_error"] for job in jobs)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if manifest["status"] != "pass":
        raise RuntimeError(
            "Incomplete or failed candidate validation; inspect manifest"
        )


if __name__ == "__main__":
    main()
