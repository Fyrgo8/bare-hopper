"""Summarize timing windows and numerical checks without changing their verdicts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, cast


def main() -> None:
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root / "scripts"))
    from analyze import summarize

    out = root / "results/opt-final-speed"
    manifest = json.loads((out / "manifest.json").read_text())
    path = out / "sauto-ws0.csv"
    with (out / "cases.csv").open() as handle:
        ids = [row["id"] for row in csv.DictReader(handle)]
    summary = cast(dict[str, Any], summarize(path, ids, expected_rounds=30))
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    windows = [(int(row["start_unix_ms"]), int(row["end_unix_ms"])) for row in rows]
    telemetry = json.loads((out / "sauto-ws0-telemetry.json").read_text())
    samples = [
        row
        for row in telemetry
        if any(
            row["end_ms"] >= start and row["start_ms"] <= end for start, end in windows
        )
    ]
    summary["acceptance_status"] = "inconclusive_memory_unlocked"
    summary["binary_sha256"] = manifest["binary_sha256"]
    summary["sm_mhz"] = sorted({row["values"].split(",")[0].strip() for row in samples})
    summary["memory_mhz"] = sorted(
        {row["values"].split(",")[1].strip() for row in samples}
    )
    summary["clock_reason_masks"] = sorted(
        {row["values"].split(",")[-1].strip() for row in samples}
    )
    summary["external_pids"] = manifest["jobs"][0]["external_pids"]
    for result in summary["cases"].values():
        result["speedup"] = 1 / result["paired_median_ratio"]
        lo, hi = result["bootstrap_95pct_interval"]
        result["speedup_95pct_interval"] = [1 / hi, 1 / lo]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    validation = root / "results/opt-final-validation"
    vm = json.loads((validation / "manifest.json").read_text())
    assert vm["status"] == "complete" and len(vm["jobs"]) == vm["expected_jobs"]
    result: dict[str, Any] = {
        "binary_sha256": vm["binary_sha256"],
        "jobs": len(vm["jobs"]),
        "suites": {},
    }
    for suite in ("full", "long", "dispatch"):
        jobs = [job for job in vm["jobs"] if job["label"].startswith(suite + "-")]
        result["suites"][suite] = {
            "jobs": len(jobs),
            "checks": sum(job["rows"] for job in jobs),
            "candidate_failures": sum(job["candidate_failures"] for job in jobs),
            "reference_failures": sum(job["reference_failures"] for job in jobs),
            "candidate_max_scaled_error": max(
                job["candidate_max_scaled_error"] for job in jobs
            ),
        }
    (validation / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
