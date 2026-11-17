"""Audit immutable validation, timing, telemetry and sanitizer evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    base = root / "results/tile128-20261117"
    sys.path.insert(0, str(root / "scripts"))
    from analyze import summarize

    binary_hash = hashlib.sha256((base / "final-bench").read_bytes()).hexdigest()
    runs: dict[str, Any] = {}
    for label in ("final-control", "final-speed", "final-control-repeat"):
        folder = base / label
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["binary_sha256"] == binary_hash
        assert len(manifest["jobs"]) == 1
        assert not manifest["jobs"][0]["external_pids"]
        assert manifest["jobs"][0]["exit_code"] == 0
        rows = read_rows(folder / "sauto-ws0.csv")
        cases = [row["id"] for row in read_rows(folder / "cases.csv")]
        result = cast(
            dict[str, Any],
            summarize(
                folder / "sauto-ws0.csv",
                cases,
                expected_rounds=30,
            ),
        )
        windows = [(int(r["start_unix_ms"]), int(r["end_unix_ms"])) for r in rows]
        telemetry = json.loads((folder / "sauto-ws0-telemetry.json").read_text())
        measured = [
            r
            for r in telemetry
            if any(
                r["end_ms"] >= start and r["start_ms"] <= end for start, end in windows
            )
        ]
        assert measured
        values = [r["values"].split(",") for r in measured]
        result["sm_mhz"] = sorted({int(r[0].strip()) for r in values})
        result["memory_mhz"] = sorted({int(r[1].strip()) for r in values})
        result["clock_reason_masks"] = sorted({r[-1].strip() for r in values})
        result["clock_samples"] = len(measured)
        result["frequency_evidence_valid"] = (
            result["sm_mhz"] == [1800]
            and result["memory_mhz"] == [2619]
            and all(int(r[-1].strip(), 16) & ~3 == 0 for r in values)
        )
        result["binary_sha256"] = binary_hash
        result["candidate_env"] = manifest["candidate_env"]
        result["acceptance_status"] = "inconclusive_memory_unlocked"
        for case in result["cases"].values():
            case["speedup"] = 1 / case["paired_median_ratio"]
        (folder / "summary.json").write_text(json.dumps(result, indent=2))
        runs[label] = result
    validation = json.loads((base / "final-validation/manifest.json").read_text())
    assert validation["binary_sha256"] == binary_hash
    assert validation["status"] == "pass"
    assert len(validation["jobs"]) == validation["expected_jobs"] == 320
    assert all(job["complete"] for job in validation["jobs"])
    assert validation["candidate_failures"] == 0
    sanitizer = read_rows(base / "sanitizers/status.csv")
    assert len(sanitizer) == 24 and all(r["exit_code"] == "0" for r in sanitizer)
    for row in sanitizer:
        stage, ws = row["mode"].split(":")
        label = f"{row['tool']}-s{stage}-ws{ws}"
        text = (base / "sanitizers" / f"{label}.log").read_text()
        marker = (
            "RACECHECK SUMMARY: 0 hazards"
            if row["tool"] == "racecheck"
            else "ERROR SUMMARY: 0 errors"
        )
        assert marker in text, label
    bitwise = read_rows(base / "bitwise.csv")
    assert len(bitwise) == 1224 and all(r["mismatches"] == "0" for r in bitwise)
    output = {
        "binary_sha256": binary_hash,
        "runs": runs,
        "validation": {k: v for k, v in validation.items() if k != "jobs"},
        "sanitizers": sanitizer,
        "bitwise_pairs": len(bitwise),
        "bitwise_elements": sum(int(r["elements"]) for r in bitwise),
        "source_hashes": {
            name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in ("final.cu", "reduce-cases.csv", "bitwise.cu")
        },
    }
    (base / "summary.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
