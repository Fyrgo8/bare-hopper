"""Run one H20 GEMM experiment, lock clocks and archive evidence.

Linux / Python >=3.12 / CUDA >=12.3; invoke from the repository root.

Preparation (no GPU kernels):
  bash scripts/setup_cutlass.sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j 4
  python scripts/run_bench.py --list-clocks --gpu 0

Reference collection (replace clocks with supported values):
  python scripts/run_bench.py --gpu 0 --sm-clock SM_MHZ --memory-clock MEM_MHZ
Comparison after implementing candidate.cu:
  python scripts/run_bench.py --gpu 0 --sm-clock SM_MHZ --memory-clock MEM_MHZ \
    --compare

Clock settings require administrator privileges; --sudo uses sudo -n only for
clock changes. Locks remain set afterwards; this script never resets another
job's settings. Memory may be omitted ONLY with --memory-unlocked-reason when
memory locking is unsupported; this is archived as a qualified result.
No jobs are killed. An occupied GPU is rejected. Each invocation creates a new
results/<UTC timestamp>-<random suffix> directory and never overwrites a run.
The initial reference is one explicit kernel, not a claim of best CUTLASS config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from analyze import summarize

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,"
    "power.draw,power.limit,temperature.gpu,utilization.gpu,"
    "clocks_event_reasons.active"
)


def command(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{argv}: {result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def active_pids(gpu: str) -> list[int]:
    output = command(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not line.isdigit() for line in rows):
        raise RuntimeError("GPU process enumeration unavailable: " + output)
    return [int(line) for line in rows]


def clock_issues(
    samples: list[dict[str, str]], sm: int, memory: int | None
) -> list[str]:
    if not samples:
        return ["No telemetry samples overlap the measured windows"]
    issues: set[str] = set()
    for row in samples:
        try:
            if int(row["clocks.current.sm"]) != sm:
                issues.add("Actual SM frequency deviated from requested lock")
            if memory is not None and int(row["clocks.current.memory"]) != memory:
                issues.add("Actual memory frequency deviated from requested lock")
            # GPU-idle and application-clock-setting bits alone are harmless.
            mask = int(row["clocks_event_reasons.active"], 16)
            if mask & ~0x3:
                issues.add("Power/thermal/hardware clock limiting observed")
        except (ValueError, KeyError):
            issues.add("Telemetry field unavailable or malformed")
    return sorted(issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--list-clocks", action="store_true")
    parser.add_argument("--sm-clock", type=int)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--memory-clock", type=int)
    memory.add_argument("--memory-unlocked-reason")
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/gemm_bench")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/gemm.csv")
    parser.add_argument("--cache", choices=["hot", "rotating"], default="rotating")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if not args.list_clocks and (
        not args.sm_clock
        or not (args.memory_clock or args.memory_unlocked_reason)
        or args.rounds < 20
        or args.iterations < 1
        or args.warmup < 1
    ):
        parser.error("Need supported clock values (or memory exemption), rounds>=20")
    return args


def main() -> int:
    args = parse_args()
    gpu = command(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"]
    )
    if "\n" in gpu or not gpu.startswith("GPU-"):
        raise RuntimeError("Select exactly one physical GPU")
    supported = command(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-supported-clocks=memory,graphics",
            "--format=csv,noheader,nounits",
        ]
    )
    if args.list_clocks:
        print("memory_MHz,graphics_MHz\n" + supported)
        return 0
    pairs = {
        tuple(int(x.strip()) for x in line.split(","))
        for line in supported.splitlines()
    }
    if not any(
        s == args.sm_clock and (args.memory_clock is None or m == args.memory_clock)
        for m, s in pairs
    ):
        raise RuntimeError("Requested SM/memory pair is not in supported clocks")
    if active_pids(gpu):
        raise RuntimeError("GPU has compute jobs; no settings changed")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    binary: Path = args.binary.resolve()
    cases: Path = args.cases.resolve()
    with cases.open(newline="", encoding="utf-8") as handle:
        case_ids = [row["id"] for row in csv.DictReader(handle)]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Case ids must be nonempty and unique")
    manifest: dict[str, object] = {
        "run_id": out.name,
        "gpu_uuid": gpu,
        "platform": platform.platform(),
        "python": sys.version,
        "options": {k: str(v) for k, v in vars(args).items()},
        "git_revision": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--porcelain"]),
        "supported_clocks": supported,
        "status": "preparing",
        "environment": {
            k: v
            for k, v in env.items()
            if k.startswith(("CUDA_", "NVIDIA_", "CUBLAS_"))
        },
    }
    with binary.open("rb") as handle:
        manifest["binary_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    source_hashes: dict[str, str] = {}
    for folder in ("src", "include", "scripts", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                with path.open("rb") as handle:
                    source_hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(
                        handle, "sha256"
                    ).hexdigest()
    manifest["source_hashes"] = source_hashes
    (out / "cases.csv").write_bytes(cases.read_bytes())
    (out / "source.patch").write_text(
        command(["git", "diff", "HEAD"]), encoding="utf-8"
    )
    manifest["before"] = command(["nvidia-smi", "-i", gpu, "-q"])
    binary_info: dict[str, object] = json.loads(
        command([str(binary), "--describe"], env=env)
    )
    manifest["binary_info"] = binary_info
    write_json(out / "manifest.json", manifest)
    try:
        if args.compare and binary_info.get("candidate_available") is not True:
            raise RuntimeError("Candidate is unimplemented; no clock settings changed")
        prefix = ["sudo", "-n"] if args.sudo else []
        locks: list[dict[str, object]] = []
        settings = [["-lgc", f"{args.sm_clock},{args.sm_clock}"]]
        if args.memory_clock is not None:
            settings.append(["-lmc", f"{args.memory_clock},{args.memory_clock}"])
        manifest["clock_commands"] = locks
        for setting in settings:
            argv = prefix + ["nvidia-smi", "-i", gpu] + setting
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
            locks.append(
                {
                    "argv": argv,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            write_json(out / "manifest.json", manifest)
            if result.returncode:
                raise RuntimeError(
                    "Clock lock failed; inspect manifest; no GPU benchmark ran"
                )
        # Catch unsupported telemetry before launching the actual experiment.
        command(
            [
                "nvidia-smi",
                "-i",
                gpu,
                f"--query-gpu={FIELDS}",
                "--format=csv,noheader,nounits",
            ]
        )
        argv = [
            str(binary),
            "--cases",
            str(cases),
            "--output",
            str(out / "samples.csv"),
            "--cache",
            args.cache,
            "--rounds",
            str(args.rounds),
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
        ]
        if args.compare:
            argv.append("--compare")
        manifest["argv"] = argv
        telemetry: list[dict[str, str]] = []
        problems: list[str] = []
        with (
            (out / "benchmark.log").open("w", encoding="utf-8") as log,
            subprocess.Popen(argv, env=env, stdout=log, stderr=log) as process,
        ):
            while process.poll() is None:
                start = time.time_ns() // 1_000_000
                row = command(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        f"--query-gpu={FIELDS}",
                        "--format=csv,noheader,nounits",
                    ]
                )
                values = next(csv.reader([row], skipinitialspace=True))
                sample: dict[str, str] = dict(
                    zip(FIELDS.split(","), values, strict=True)
                )
                sample["poll_start_unix_ms"] = str(start)
                sample["poll_end_unix_ms"] = str(time.time_ns() // 1_000_000)
                telemetry.append(sample)
                with (out / "telemetry.jsonl").open("a", encoding="utf-8") as trace:
                    trace.write(json.dumps(sample) + "\n")
                if any(pid != process.pid for pid in active_pids(gpu)):
                    problems.append(
                        "Other compute processes appeared during measurement"
                    )
                time.sleep(0.02)
            manifest["exit_code"] = process.returncode
        if manifest["exit_code"] != 0:
            raise RuntimeError("Benchmark failed; see benchmark.log")
        samples = out / "samples.csv"
        timing = summarize(samples, case_ids, expected_rounds=args.rounds)
        with samples.open(newline="", encoding="utf-8") as handle:
            windows = [
                (int(row["start_unix_ms"]), int(row["end_unix_ms"]))
                for row in csv.DictReader(handle)
            ]
        measured = [
            row
            for row in telemetry
            if any(
                int(row["poll_start_unix_ms"]) <= end
                and int(row["poll_end_unix_ms"]) >= start
                for start, end in windows
            )
        ]
        problems += clock_issues(measured, args.sm_clock, args.memory_clock)
        if len({row["power.limit"] for row in measured}) != 1:
            problems.append("Power limit unavailable or changed during measurement")
        if not all(
            any(
                int(row["poll_start_unix_ms"]) <= end
                and int(row["poll_end_unix_ms"]) >= start
                for row in measured
            )
            for start, end in windows
        ):
            problems.append("Telemetry did not cover every paired timing round")
        if args.memory_unlocked_reason:
            problems.append(
                "Memory clock lock exempted: " + args.memory_unlocked_reason
            )
        timing["environment_issues"] = sorted(set(problems))
        timing["telemetry_measured_samples"] = len(measured)
        timing["acceptance_status"] = (
            "inconclusive" if problems else timing["timing_status"]
        )
        write_json(out / "summary.json", timing)
        manifest["status"] = timing["acceptance_status"]
        print(json.dumps({"directory": str(out), **timing}, indent=2))
        return 0 if manifest["status"] in {"pass", "reference_only"} else 2
    except Exception as error:
        manifest["status"] = "error"
        manifest["error"] = str(error)
        raise
    finally:
        # No automatic clock reset: preserve explicit settings and record final state.
        result = subprocess.run(
            ["nvidia-smi", "-i", gpu, "-q"], capture_output=True, text=True, check=False
        )
        manifest["after"] = result.stdout
        write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
