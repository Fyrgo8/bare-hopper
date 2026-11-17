"""Audit stronger references without changing the original acceptance baseline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, cast


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    base = root / "results/reference-audit-20261117"
    sys.path.insert(0, str(root / "scripts"))
    from analyze import summarize

    output: dict[str, Any] = {}
    for name in ("cublas-speed", "cublas-strict-speed", "streamk-formal16"):
        folder = base / name
        manifest = json.loads((folder / "manifest.json").read_text())
        job = manifest["jobs"][0]
        assert job["exit_code"] == 0 and not job["external_pids"]
        result = cast(
            dict[str, Any],
            summarize(
                folder / "sauto-ws0.csv",
                [r["id"] for r in rows(folder / "cases.csv")],
                expected_rounds=30,
            ),
        )
        windows = [
            (int(r["start_unix_ms"]), int(r["end_unix_ms"]))
            for r in rows(folder / "sauto-ws0.csv")
        ]
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
        result["binary_sha256"] = manifest["binary_sha256"]
        result["external_pids"] = job["external_pids"]
        result["acceptance_status"] = "inconclusive_memory_unlocked"
        assert result["sm_mhz"] == [1800] and result["memory_mhz"] == [2619]
        assert all(int(r[-1].strip(), 16) & ~3 == 0 for r in values)
        for case in result["cases"].values():
            case["reference_over_candidate"] = 1 / case["paired_median_ratio"]
            case["candidate_over_reference"] = case["paired_median_ratio"]
        output[name] = result
    statuses = rows(base / "validation/status.csv")
    statuses += rows(base / "strict-validation/status.csv")
    assert len(statuses) == 24
    details: list[dict[str, str | int]] = []
    for status in statuses:
        folder = (
            "strict-validation"
            if status["backend"] == "cublas_strict"
            else "validation"
        )
        result = rows(base / folder / f"{status['backend']}-{status['pattern']}.csv")
        assert status["exit_code"] in ("0", "1") and len(result) == 2
        assert [r["case"] for r in result] == ["long32768", "long65536"]
        assert all(r["oracle"] == "fp64" and r["repeats"] == "3" for r in result)
        assert all(r["pattern"] == status["pattern"] for r in result)
        details.append(
            {
                **status,
                "rows": len(result),
                "candidate_failures": sum(r["candidate_ok"] != "1" for r in result),
                "reference_failures": sum(r["reference_ok"] != "1" for r in result),
            }
        )
    output["validation"] = details
    (base / "summary.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
