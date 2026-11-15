"""Paired, per-case latency verdicts; reference-only data never passes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def assess(
    reference: list[float], candidate: list[float], *, limit: float = 1.03
) -> dict[str, object]:
    if len(reference) != len(candidate) or len(reference) < 20:
        raise ValueError("At least 20 paired rounds are required")
    if any(not math.isfinite(x) or x <= 0 for x in reference + candidate):
        raise ValueError("Timings must be finite and positive")
    ratios = [c / r for r, c in zip(reference, candidate, strict=True)]
    rng = random.Random(20261115)
    boot = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(4000)
    )
    lo, hi = boot[100], boot[3899]

    def relative_mad(values: list[float]) -> float:
        median = statistics.median(values)
        return statistics.median(abs(x - median) for x in values) / median

    noise = max(relative_mad(reference), relative_mad(candidate))
    # Noise near the 3% budget cannot justify a precise pass/fail verdict.
    status = "inconclusive"
    if noise <= 0.01:
        if hi <= limit:
            status = "pass"
        elif lo > limit:
            status = "fail"
    return {
        "status": status,
        "rounds": len(ratios),
        "reference_median_ms": statistics.median(reference),
        "candidate_median_ms": statistics.median(candidate),
        "paired_median_ratio": statistics.median(ratios),
        "bootstrap_95pct_interval": [lo, hi],
        "max_relative_mad": noise,
        "limit": limit,
    }


def summarize(
    path: Path, expected_ids: list[str], *, expected_rounds: int | None = None
) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in csv.DictReader(handle):
            groups[row["case"]].append(row)
    if set(groups) != set(expected_ids):
        raise ValueError("Samples do not match the complete case list")
    results: dict[str, object] = {}
    statuses: list[str] = []
    for case_id in expected_ids:
        rows = groups[case_id]
        if expected_rounds is not None and len(rows) != expected_rounds:
            raise ValueError(f"{case_id}: incomplete round count")
        rounds = [int(row["round"]) for row in rows]
        if rounds != list(range(len(rows))) or len(rows) < 20:
            raise ValueError(f"{case_id}: missing/duplicate rounds or fewer than 20")
        ref = [float(row["reference_ms"]) for row in rows]
        if any(not math.isfinite(x) or x <= 0 for x in ref):
            raise ValueError("Invalid reference timing")
        if any(row["reference_ok"] != "1" for row in rows):
            raise ValueError("Reference numerical validation failed")
        if all(row["candidate_ms"] == "" for row in rows):
            results[case_id] = {
                "status": "reference_only",
                "reference_median_ms": statistics.median(ref),
            }
            statuses.append("reference_only")
            continue
        if any(row["candidate_ok"] != "1" for row in rows):
            raise ValueError("Candidate numerical validation missing or failed")
        if any(
            row["order"] != ("AB" if i % 2 == 0 else "BA") for i, row in enumerate(rows)
        ):
            raise ValueError("Expected alternating AB/BA order")
        result = assess(ref, [float(row["candidate_ms"]) for row in rows])
        results[case_id] = result
        statuses.append(str(result["status"]))
    overall = "inconclusive"
    if all(s == "reference_only" for s in statuses):
        overall = "reference_only"
    elif "fail" in statuses:
        overall = "fail"
    elif all(s == "pass" for s in statuses):
        overall = "pass"
    return {"timing_status": overall, "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--cases", type=Path, default=Path("configs/gemm.csv"))
    args = parser.parse_args()
    with args.cases.open(newline="", encoding="utf-8") as handle:
        ids = [row["id"] for row in csv.DictReader(handle)]
    result = summarize(args.samples, ids)
    # Standalone timing analysis lacks clock evidence and is not an acceptance.
    result["acceptance_status"] = "unverified_environment"
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
