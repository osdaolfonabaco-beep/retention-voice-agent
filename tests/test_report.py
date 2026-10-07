"""Offline tests for retention/report.py."""

import json
from pathlib import Path

import pytest

from retention import report
from retention.audit import CATEGORIES


def audit_row(cid, fail=None):
    verdicts = {c: "pass" for c in CATEGORIES}
    findings = []
    for cat, rules, turns in (fail or []):
        verdicts[cat] = "fail"
        findings.append({"category": cat, "rules": rules, "turns": turns,
                         "evidence": f"evidence for {cid}", "why": f"why {cat}"})
    return {"id": cid, "verdicts": verdicts, "findings": findings, "warnings": [],
            "model": "m", "input_tokens": 1, "output_tokens": 1}


def write_run(tmp_path, rows, name="r1"):
    run_dir = tmp_path / "data" / "audits" / name
    run_dir.mkdir(parents=True)
    (run_dir / "audits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                          encoding="utf-8")
    return run_dir


ROWS = [
    audit_row("ag-01"),
    audit_row("ag-02", [("privacy_breach", ["VR-3"], [4])]),
    audit_row("ag-03", [("false_resolution", ["CL-1", "FL-4"], [6, 8]),
                        ("premature_closure", ["CL-8"], [9])]),
    audit_row("ag-04"),
]


def test_counts_percent_and_findings(tmp_path):
    write_run(tmp_path, ROWS)
    lines = []
    assert report.main(["--run", "r1"], root=tmp_path, say=lines.append) == 0
    text = "\n".join(lines)
    assert "conversations audited: 4" in lines
    assert "with no failed category: 2 (50.0%)" in lines
    assert "  privacy_breach: 1" in lines and "  false_resolution: 1" in lines
    assert "  premature_closure: 1" in lines and "  missed_escalation: 0" in lines
    assert "findings: 3" in lines
    assert "ag-03  false_resolution  rules: CL-1, FL-4  turns: 6, 8" in lines
    assert "  evidence: evidence for ag-02" in lines and "  why: why premature_closure" in lines
    assert text.index("ag-02  privacy_breach") < text.index("ag-03  false_resolution")


def test_categories_are_listed_in_the_auditor_order(tmp_path):
    write_run(tmp_path, ROWS)
    lines = []
    report.main(["--run", "r1"], root=tmp_path, say=lines.append)
    listed = [line.strip().split(":")[0] for line in lines if line.startswith("  ")
              and line.strip().split(":")[0] in CATEGORIES]
    assert listed == list(CATEGORIES)


def test_summary_json(tmp_path):
    run_dir = write_run(tmp_path, ROWS)
    report.main(["--run", "r1"], root=tmp_path, say=lambda *_: None)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "run": "r1", "audited": 4, "clean": 2, "clean_pct": 50.0, "findings": 3,
        "failed_by_category": {"false_resolution": 1, "missed_escalation": 0,
                               "unfulfillable_promise": 0, "aggravating_tone": 0,
                               "premature_closure": 1, "privacy_breach": 1}}


def test_percent_has_one_decimal(tmp_path):
    rows = [audit_row("a1"), audit_row("a2"), audit_row("a3", [("aggravating_tone", [], [])])]
    write_run(tmp_path, rows)
    summary = report.summarize("r1", report.load_audits(
        tmp_path / "data" / "audits" / "r1" / "audits.jsonl"))
    assert summary["clean"] == 2 and summary["clean_pct"] == 66.7


def test_a_conversation_with_two_failed_categories_counts_once_per_category(tmp_path):
    write_run(tmp_path, [ROWS[2]])
    summary = report.summarize("r1", [ROWS[2]])
    assert summary["audited"] == 1 and summary["clean"] == 0 and summary["clean_pct"] == 0.0
    assert summary["failed_by_category"]["false_resolution"] == 1
    assert summary["failed_by_category"]["premature_closure"] == 1


def test_all_clean_run_has_no_findings(tmp_path):
    write_run(tmp_path, [audit_row("a1"), audit_row("a2")])
    lines = []
    report.main(["--run", "r1"], root=tmp_path, say=lines.append)
    assert "with no failed category: 2 (100.0%)" in lines and "findings: 0" in lines
    assert not any("evidence:" in line for line in lines)


def test_errors(tmp_path):
    with pytest.raises(SystemExit) as err:
        report.main(["--run", "nope"], root=tmp_path, say=lambda *_: None)
    assert str(err.value).startswith("error: required file not found:")
    run_dir = write_run(tmp_path, [], name="empty")
    with pytest.raises(SystemExit, match="no audits"):
        report.main(["--run", "empty"], root=tmp_path, say=lambda *_: None)
    (run_dir / "audits.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="bad audit line"):
        report.main(["--run", "empty"], root=tmp_path, say=lambda *_: None)
    assert not (run_dir / "summary.json").exists()


@pytest.mark.parametrize("argv", [[], ["--run", "../x"]])
def test_bad_arguments_exit_2(tmp_path, argv):
    with pytest.raises(SystemExit) as err:
        report.main(argv, root=tmp_path)
    assert err.value.code == 2


def test_report_never_names_label_or_sealed_files():
    source = Path(report.__file__).read_text(encoding="utf-8")
    for name in ("labels_key", "second_reader", "scenarios.sealed", "sealed"):
        assert name not in source
