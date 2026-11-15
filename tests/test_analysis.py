from __future__ import annotations

import csv
from pathlib import Path

import pytest
from analyze import assess, summarize
from run_bench import clock_issues


@pytest.mark.parametrize(
    ("ratio", "expected"), [(1.02, "pass"), (1.04, "fail"), (0.8, "pass")]
)
def test_stable_ratios(ratio: float, expected: str) -> None:
    assert assess([1.0] * 30, [ratio] * 30)["status"] == expected


def test_boundary_crossing_is_inconclusive() -> None:
    assert assess([1.0] * 30, [1.029, 1.031] * 15)["status"] == "inconclusive"


def test_noise_cannot_pass() -> None:
    assert assess([0.9, 1.1] * 15, [0.9, 1.1] * 15)["status"] == "inconclusive"


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timings_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError):
        assess([1.0] * 30, [invalid] * 30)


def test_unpaired_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        assess([1.0] * 30, [1.0] * 29)


def make_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "round",
                "order",
                "reference_ms",
                "candidate_ms",
                "reference_ok",
                "candidate_ok",
            ],
        )
        writer.writeheader()
        for i in range(30):
            writer.writerow(
                {
                    "case": "square",
                    "round": i,
                    "order": "A",
                    "reference_ms": 1.0,
                    "candidate_ms": "",
                    "reference_ok": "1",
                    "candidate_ok": "",
                }
            )


def test_reference_only_never_passes(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    make_csv(path)
    assert summarize(path, ["square"])["timing_status"] == "reference_only"


def test_missing_case_cannot_pass(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    make_csv(path)
    with pytest.raises(ValueError, match="complete case list"):
        summarize(path, ["square", "missing"])


def test_truncated_rounds_cannot_pass(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    make_csv(path)
    with pytest.raises(ValueError, match="incomplete round count"):
        summarize(path, ["square"], expected_rounds=40)


def test_duplicate_round_cannot_pass(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    make_csv(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing/duplicate rounds"):
        summarize(path, ["square"])


def test_frequency_and_throttling_must_be_valid() -> None:
    good = {
        "clocks.current.sm": "1410",
        "clocks.current.memory": "1593",
        "clocks_event_reasons.active": "0x0000000000000002",
    }
    assert not clock_issues([good], 1410, 1593)
    assert clock_issues([{**good, "clocks.current.sm": "1200"}], 1410, 1593)
    assert clock_issues([{**good, "clocks_event_reasons.active": "0x4"}], 1410, 1593)
    assert clock_issues([{**good, "clocks.current.memory": "N/A"}], 1410, 1593)
    assert clock_issues([], 1410, 1593)
