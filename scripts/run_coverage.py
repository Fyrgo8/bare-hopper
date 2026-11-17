"""Correctness coverage only: no latency measurements or clock changes.

Runs on the H20 host with its installed Python (3.10+) and compiled binary.
Each launch is checked against independent FP64 DGEMM; structured
patterns additionally check exact oracle values inside the binary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def coverage_shapes() -> list[tuple[int, int, int]]:
    """Cartesian boundaries, pipeline wrap lengths, scales and aspect ratios."""
    shapes: set[tuple[int, int, int]] = set(
        itertools.product(
            (63, 64, 65, 127, 129),
            (120, 128, 136, 248, 264),
            (56, 64, 72, 128, 264),
        )
    )
    shapes.update(
        (65, 136, k) for k in (8, 16, 192, 200, 256, 320, 512, 520, 1024, 4104)
    )
    shapes.update((m, 4096, 4096) for m in (1, 8, 32, 128, 1024))
    shapes.update((4096, n, 1024) for n in (8, 32, 128))
    shapes.update((m, m, m) for m in (128, 256, 512, 1024, 2048, 4096))
    shapes.update((65, 136, k) for k in (8192, 32768, 65536))
    shapes.update(
        {
            (1, 8, 8),
            (64, 128, 64),
            (4096, 4096, 1024),
            (3968, 4096, 4096),
            (4104, 4088, 4096),
            (4096, 64, 4096),
            (256, 8192, 512),
            (8192, 256, 512),
        }
    )
    return sorted(shapes)


def write_cases(path: Path, shapes: list[tuple[int, int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "m", "n", "k"])
        for m, n, k in shapes:
            writer.writerow([f"m{m}_n{n}_k{k}", m, n, k])


def long_k_shapes() -> list[tuple[int, int, int]]:
    # Switch boundary, segment remainder, former failures, and output tails.
    return sorted(
        {
            (65, 136, k)
            for k in (
                8192,
                16376,
                16384,
                16392,
                16632,
                16640,
                16648,
                32760,
                32768,
                32776,
                65536,
            )
        }
        | {(1, 8, 16392), (63, 120, 32776), (129, 264, 16384)}
    )


def run(args: argparse.Namespace) -> int:
    gpu = str(args.gpu)
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if query.stdout.strip() and not args.allow_shared_gpu:
        raise RuntimeError("Selected GPU has compute processes; no kernels launched")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017 (host Python 3.10)
    out = ROOT / "results" / f"coverage-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    binary = out / "gemm_bench"
    shutil.copy2(Path(args.binary).resolve(), binary)
    broad = long_k_shapes() if args.suite == "long-k" else coverage_shapes()
    # Cover each ring wrap length and representative aspect ratios in every
    # alternative implementation without repeating the entire large sweep.
    alternative = sorted(
        set(broad[::13])
        | {
            (1, 8, 8),
            (63, 120, 56),
            (64, 128, 64),
            (65, 136, 72),
            *((65, 136, k) for k in (192, 200, 256, 264, 320, 512, 520, 4104, 65536)),
        }
    )
    if args.suite == "long-k":
        alternative = broad
    write_cases(out / "broad.csv", broad)
    write_cases(out / "alternative.csv", alternative)
    manifest: dict[str, object] = {
        "kind": "correctness_only",
        "suite": args.suite,
        "stable_long_k": os.environ.get("BARE_GEMM_STABLE_LONG_K", "0"),
        "oracle": "fp64",
        "gpu": gpu,
        "allow_shared_gpu": args.allow_shared_gpu,
        "initial_compute_pids": query.stdout.split(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src", "include", "scripts")
            for p in (ROOT / folder).glob("*")
            if p.is_file()
        },
        "broad_shapes": len(broad),
        "alternative_shapes": len(alternative),
        "patterns": PATTERNS,
        "seeds": [17, 20261115, 99173],
        "repeats": args.repeats,
        "jobs": [],
        "status": "running",
    }
    jobs: list[dict[str, object]] = []
    manifest["jobs"] = jobs
    failures = 0
    checks = 0
    candidate_failures = 0
    reference_failures = 0
    # Include all compiled paths: vector, cooperative TMA 1..4, specialized 1..4.
    modes = [(1, 0), (0, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1), (4, 1)]
    for stages, ws in modes:
        is_default = (stages, ws) == (1, 0) or args.suite == "long-k"
        cases = out / ("broad.csv" if is_default else "alternative.csv")
        expected = broad if is_default else alternative
        patterns = (
            PATTERNS
            if is_default
            else (
                "uniform",
                "logscale",
                "identity",
                "cancellation",
            )
        )
        seeds = (17, 20261115, 99173) if is_default else (17, 99173)
        for pattern, seed in itertools.product(patterns, seeds):
            label = f"s{stages}-ws{ws}-{pattern}-{seed}"
            result_path = out / f"{label}.csv"
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
                str(args.repeats),
                "--oracle",
                "fp64",
                "--output",
                str(result_path),
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                BARE_GEMM_STAGES=str(stages),
                BARE_GEMM_WARP_SPECIALIZED=str(ws),
            )
            with (out / f"{label}.log").open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(
                        argv,
                        env=env,
                        stdout=log,
                        stderr=log,
                        timeout=300,
                        check=False,
                    )
                    exit_code = result.returncode
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    log.write("\nTIMEOUT after 300 seconds\n")
            rows: list[dict[str, str]] = []
            if result_path.exists():
                with result_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            complete = (
                exit_code in (0, 1)
                and [(int(r["m"]), int(r["n"]), int(r["k"])) for r in rows] == expected
                and all(
                    r["pattern"] == pattern
                    and int(r["seed"]) == seed
                    and int(r["repeats"]) == args.repeats
                    and r["oracle"] == "fp64"
                    for r in rows
                )
            )
            candidate_ok = complete and all(r["candidate_ok"] == "1" for r in rows)
            reference_ok = complete and all(r["reference_ok"] == "1" for r in rows)
            valid = exit_code == 0 and candidate_ok and reference_ok
            candidate_failures += not candidate_ok
            reference_failures += not reference_ok
            failures += not valid
            checks += len(rows)
            jobs.append(
                {
                    "label": label,
                    "argv": argv,
                    "exit_code": exit_code,
                    "rows": len(rows),
                    "expected_rows": len(expected),
                    "passed": valid,
                    "candidate_passed": candidate_ok,
                    "reference_passed": reference_ok,
                }
            )
            (out / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"{label}: {'PASS' if valid else 'FAIL'} {len(rows)}/{len(expected)}",
                flush=True,
            )
    manifest["status"] = "pass" if failures == 0 else "fail"
    manifest["candidate_status"] = "pass" if candidate_failures == 0 else "fail"
    manifest["reference_status"] = "pass" if reference_failures == 0 else "fail"
    manifest["shape_pattern_seed_checks"] = checks
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "directory": str(out),
                "status": manifest["status"],
                "jobs": len(jobs),
                "failed_jobs": failures,
            }
        ),
        flush=True,
    )
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/gemm_bench")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--suite", choices=("full", "long-k"), default="full")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Explicitly allow correctness checks alongside existing GPU processes",
    )
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two repeated launches with refreshed inputs required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
