"""Diagnostic mode sweep; memory is observed, not hard locked (host Python 3.10)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--modes", default="1:0,2:0,3:0,4:0,3:1")
    parser.add_argument("--split", default="0")
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.out / "gemm_bench"
    shutil.copy2(args.binary, binary)
    cases = args.out / "cases.csv"
    cases.write_text(
        "id,m,n,k\nsquare,4096,4096,4096\nsmall_m,1024,4096,4096\n"
        "short_k,4096,4096,1024\nm_tail,3968,4096,4096\n"
        "mn_tail,4104,4088,4096\nlong32768,65,136,32768\n"
        "long65536,65,136,65536\nlarge_long,1024,1024,16384\n"
    )
    if args.cases:
        shutil.copy2(args.cases, cases)
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    jobs: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "acceptance_status": "inconclusive_memory_unlocked",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "split": args.split,
        "jobs": jobs,
    }
    for mode in args.modes.split(","):
        stages, specialized = mode.split(":")
        label = f"s{stages}-ws{specialized}"
        if subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip():
            raise RuntimeError("GPU busy; no benchmark launched")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            BARE_GEMM_STAGES=stages,
            BARE_GEMM_WARP_SPECIALIZED=specialized,
            BARE_GEMM_STABLE_LONG_K="1",
            BARE_GEMM_SPLIT_K=args.split,
        )
        if stages == "auto":
            env.pop("BARE_GEMM_STAGES", None)
        argv = [
            str(binary),
            "--compare",
            "--cache",
            "rotating",
            "--cases",
            str(cases),
            "--rounds",
            str(args.rounds),
            "--warmup",
            "32",
            "--output",
            str(args.out / f"{label}.csv"),
        ]
        telemetry: list[dict[str, object]] = []
        external: set[str] = set()
        with (args.out / f"{label}.log").open("w") as log:
            proc = subprocess.Popen(argv, env=env, stdout=log, stderr=log)
            while proc.poll() is None:
                start = int(time.time() * 1000)
                values = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        "0",
                        "--query-gpu=clocks.sm,clocks.mem,"
                        "temperature.gpu,power.draw,clocks_event_reasons.active",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                telemetry.append(
                    {
                        "start_ms": start,
                        "end_ms": int(time.time() * 1000),
                        "values": values,
                    }
                )
                process_rows = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        "0",
                        "--query-compute-apps=gpu_uuid,pid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).splitlines()
                for row in csv.reader(process_rows):
                    if len(row) == 2 and row[0].strip() == gpu_uuid:
                        pid = row[1].strip()
                        if pid != str(proc.pid):
                            external.add(pid)
                time.sleep(0.15)
            code = proc.wait()
        (args.out / f"{label}-telemetry.json").write_text(
            json.dumps(telemetry, indent=2)
        )
        with (args.out / f"{label}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        summary = {}
        for case in dict.fromkeys(row["case"] for row in rows):
            group = [row for row in rows if row["case"] == case]
            summary[case] = {
                "candidate_ms": statistics.median(
                    float(row["candidate_ms"]) for row in group
                ),
                "reference_ms": statistics.median(
                    float(row["reference_ms"]) for row in group
                ),
                "speedup": 1
                / statistics.median(
                    float(row["candidate_ms"]) / float(row["reference_ms"])
                    for row in group
                ),
                "rounds": len(group),
                "valid": all(
                    row["candidate_ok"] == row["reference_ok"] == "1" for row in group
                ),
            }
        jobs.append(
            {
                "mode": label,
                "argv": argv,
                "exit_code": code,
                "external_pids": sorted(external),
                "cases": summary,
            }
        )
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if code or external:
            raise RuntimeError(f"Invalid run {label}: exit={code}, external={external}")


if __name__ == "__main__":
    main()
