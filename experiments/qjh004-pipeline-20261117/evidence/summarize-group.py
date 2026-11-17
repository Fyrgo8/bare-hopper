"""Check coverage and timing evidence for the grouped CTA experiment."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, cast


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root / "scripts"))
    from analyze import summarize

    runs: dict[str, Any] = {}
    for label in ("group-control", "group-final-speed", "group-disabled"):
        out = root / "results" / label
        cases = [row["id"] for row in read_rows(out / "cases.csv")]
        result = cast(
            dict[str, Any], summarize(out / "sauto-ws0.csv", cases, expected_rounds=30)
        )
        rows = read_rows(out / "sauto-ws0.csv")
        windows = [(int(row["start_unix_ms"]), int(row["end_unix_ms"])) for row in rows]
        telemetry = json.loads((out / "sauto-ws0-telemetry.json").read_text())
        measured = [
            row
            for row in telemetry
            if any(
                row["end_ms"] >= start and row["start_ms"] <= end
                for start, end in windows
            )
        ]
        result["sm_mhz"] = sorted(
            {row["values"].split(",")[0].strip() for row in measured}
        )
        result["memory_mhz"] = sorted(
            {row["values"].split(",")[1].strip() for row in measured}
        )
        result["clock_reason_masks"] = sorted(
            {row["values"].split(",")[-1].strip() for row in measured}
        )
        manifest = json.loads((out / "manifest.json").read_text())
        result["external_pids"] = manifest["jobs"][0]["external_pids"]
        result["binary_sha256"] = manifest["binary_sha256"]
        result["acceptance_status"] = "inconclusive_memory_unlocked"
        for case in result["cases"].values():
            case["speedup"] = 1 / case["paired_median_ratio"]
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        runs[label] = result
    expected = [row["id"] for row in read_rows(root / "group-cases.csv")]
    validation: list[dict[str, Any]] = []
    for folder in sorted((root / "results").glob("group-validation-*")):
        for status in read_rows(folder / "status.csv"):
            stages, ws = status["mode"].split(":")
            path = folder / f"s{stages}-ws{ws}-{status['pattern']}-{status['seed']}.csv"
            rows = read_rows(path)
            assert [row["case"] for row in rows] == expected
            assert int(status["exit_code"]) in (0, 1)
            validation.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": len(rows),
                    "candidate_failures": sum(
                        row["candidate_ok"] != "1" for row in rows
                    ),
                    "reference_failures": sum(
                        row["reference_ok"] != "1" for row in rows
                    ),
                    "max_scaled_error": max(
                        float(row["candidate_max_scaled_error"]) for row in rows
                    ),
                }
            )
    assert len(validation) == 128
    assert len({job["path"] for job in validation}) == 128
    sanitizers: dict[str, list[dict[str, str]]] = {}
    for tool in ("memcheck", "racecheck", "synccheck"):
        statuses = read_rows(root / "results" / f"group-{tool}" / "status.csv")
        assert len(statuses) == 8 and all(row["exit_code"] == "0" for row in statuses)
        sanitizers[tool] = statuses
    default = [
        row
        for path in (root / "results/group-default").glob("*.csv")
        for row in read_rows(path)
    ]
    assert len(default) == 16 and all(row["candidate_ok"] == "1" for row in default)
    output: dict[str, Any] = {
        "runs": runs,
        "validation": {
            "jobs": len(validation),
            "checks": sum(job["rows"] for job in validation),
            "candidate_failures": sum(job["candidate_failures"] for job in validation),
            "reference_failures": sum(job["reference_failures"] for job in validation),
            "max_scaled_error": max(job["max_scaled_error"] for job in validation),
            "default_checks": len(default),
            "details": validation,
        },
        "sanitizers": sanitizers,
    }
    (root / "results/group-summary.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
