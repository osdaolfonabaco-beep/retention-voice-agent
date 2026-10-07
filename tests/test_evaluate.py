"""Offline tests for the evaluator, on a small hand-made example.

Key and auditor verdicts (only failures listed; everything else is "pass"):

    id  dataset  key fails                                    auditor fails
    c1  batch2   false_resolution (FL-1, FL-2),                false_resolution (fl-2),
                 privacy_breach (PR-1)                         privacy_breach (PP-9)
    c2  batch2   false_resolution (FL-3), disputed             -
    c3  batch3   -                                             aggravating_tone (TN-2)
    c4  batch3   -                                             -
    c5           in the "check" split, never audited

Cells (4 conversations x 6 categories = 24):
    false_resolution   TP 1  FN 1  FP 0  TN 2
    privacy_breach     TP 1  FN 0  FP 0  TN 3
    aggravating_tone   TP 0  FN 0  FP 1  TN 3
    other three        TP 0  FN 0  FP 0  TN 4 each
    total              TP 2  FN 1  FP 1  TN 20
Without the disputed cell (c2, false_resolution): total TP 2  FN 0  FP 1  TN 20.
Exact matches of all six verdicts: c1, c4 = 2. Key's clean conversations: c3, c4;
the auditor passed c4 only = 1 of 2. Rule agreement over TP cells: c1/FR yes, c1/PR no = 1 of 2.
"""

import json

import pytest

from retention import evaluate
from retention.audit import CATEGORIES

KEY = {
    "c1": ({"false_resolution": ["FL-1", "FL-2"], "privacy_breach": ["PR-1"]}, []),
    "c2": ({"false_resolution": ["FL-3"]}, ["false_resolution"]),
    "c3": ({}, []),
    "c4": ({}, []),
}
AUDITOR = {
    "c1": {"false_resolution": (["fl-2"], [2], "q1", "w1"), "privacy_breach": (["PP-9"], [4], "q2", "w2")},
    "c2": {},
    "c3": {"aggravating_tone": (["TN-2"], [3, 1], "rude", "snapped")},
    "c4": {},
}
DATASET = {"c1": "batch2", "c2": "batch2", "c3": "batch3", "c4": "batch3"}


def verdicts(fails):
    return {c: ("fail" if c in fails else "pass") for c in CATEGORIES}


def make_root(tmp_path, with_audits=True):
    root = tmp_path / "proj"
    for ds in ("batch2", "batch3"):
        (root / "data" / ds).mkdir(parents=True)
    (root / "data" / "dev_split.json").write_text(
        json.dumps({"tune": ["c1", "c2", "c3", "c4"], "check": ["c5"]}), encoding="utf-8")
    for ds in ("batch2", "batch3"):
        rows = []
        for cid, (fails, disputed) in KEY.items():
            if DATASET[cid] != ds:
                continue
            rows.append({"id": cid, "verdicts": verdicts(fails),
                         "findings": [{"category": c, "rules": r} for c, r in fails.items()],
                         **({"disputed": [{"category": c} for c in disputed]} if disputed else {})})
        (root / "data" / ds / "labels_key.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = root / "data" / "audits" / "r1"
    out.mkdir(parents=True)
    if with_audits:
        rows = []
        for cid, fails in AUDITOR.items():
            rows.append({"id": cid, "verdicts": verdicts(fails),
                         "findings": [{"category": c, "rules": r, "turns": t, "evidence": e, "why": w}
                                      for c, (r, t, e, w) in fails.items()]})
        (out / "audits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                          encoding="utf-8")
    return root


def run(root, split="tune", *extra):
    lines = []
    code = evaluate.main(["--run", "r1", "--split", split, *extra], root=root, say=lines.append)
    return code, "\n".join(lines)


def metrics(root, split="tune"):
    return json.loads((root / "data" / "audits" / "r1" / f"metrics_{split}.json").read_text())


def counts(m):
    return (m["tp"], m["fn"], m["fp"], m["tn"])


# ------------------------------------------------------------------- statistics

def test_wilson_known_values():
    assert evaluate.wilson(0, 0) is None
    assert evaluate.wilson(1, 2) == pytest.approx((0.0945, 0.9055), abs=1e-3)
    assert evaluate.wilson(2, 3) == pytest.approx((0.2077, 0.9385), abs=1e-3)
    assert evaluate.wilson(20, 21) == pytest.approx((0.7733, 0.9923), abs=1e-3)
    low, high = evaluate.wilson(0, 5)
    assert low == 0.0 and high == pytest.approx(0.4345, abs=1e-3)
    low, high = evaluate.wilson(5, 5)
    assert high == 1.0 and low == pytest.approx(0.5655, abs=1e-3)


def test_rate_with_empty_denominator():
    assert evaluate.rate(0, 0) == {"k": 0, "n": 0, "rate": None, "low": None, "high": None}


# ------------------------------------------------------------------- counts

def test_counts_per_category_and_total(tmp_path):
    root = make_root(tmp_path)
    assert run(root)[0] == 0
    t = metrics(root)["all_cells"]
    assert counts(t["per_category"]["false_resolution"]) == (1, 1, 0, 2)
    assert counts(t["per_category"]["privacy_breach"]) == (1, 0, 0, 3)
    assert counts(t["per_category"]["aggravating_tone"]) == (0, 0, 1, 3)
    for cat in ("missed_escalation", "unfulfillable_promise", "premature_closure"):
        assert counts(t["per_category"][cat]) == (0, 0, 0, 4)
    assert counts(t["total"]) == (2, 1, 1, 20)


def test_rates_and_intervals(tmp_path):
    root = make_root(tmp_path)
    run(root)
    total = metrics(root)["all_cells"]["total"]
    assert total["tpr"]["k"] == 2 and total["tpr"]["n"] == 3
    assert total["tpr"]["rate"] == pytest.approx(2 / 3)
    assert (total["tpr"]["low"], total["tpr"]["high"]) == pytest.approx((0.2077, 0.9385), abs=1e-3)
    assert total["tnr"]["k"] == 20 and total["tnr"]["n"] == 21
    assert total["tnr"]["rate"] == pytest.approx(20 / 21)
    assert (total["tnr"]["low"], total["tnr"]["high"]) == pytest.approx((0.7733, 0.9923), abs=1e-3)
    # no positives in the key for this category: TPR undefined
    none = metrics(root)["all_cells"]["per_category"]["aggravating_tone"]["tpr"]
    assert none["rate"] is None and none["low"] is None


def test_conversation_level_and_rule_agreement(tmp_path):
    root = make_root(tmp_path)
    run(root)
    m = metrics(root)
    conv = m["conversations"]
    assert (conv["audited"], conv["missing"], conv["exact_matches"]) == (4, 0, 2)
    assert (conv["key_clean"], conv["key_clean_passed_all"]) == (2, 1)
    agree = m["all_cells"]["total"]["rule_agreement"]
    assert (agree["k"], agree["n"], agree["rate"]) == (1, 2, 0.5)
    assert m["all_cells"]["per_category"]["false_resolution"]["rule_agreement"]["k"] == 1
    assert m["all_cells"]["per_category"]["privacy_breach"]["rule_agreement"]["k"] == 0


def test_second_table_leaves_out_disputed_cells(tmp_path):
    root = make_root(tmp_path)
    run(root)
    m = metrics(root)
    t = m["excluding_disputed"]
    assert m["disputed_cells"] == 1
    assert counts(t["per_category"]["false_resolution"]) == (1, 0, 0, 2)
    assert counts(t["total"]) == (2, 0, 1, 20)
    assert t["total"]["tpr"]["k"] == 2 and t["total"]["tpr"]["n"] == 2
    assert t["total"]["tpr"]["low"] == pytest.approx(0.3424, abs=1e-3)
    assert counts(m["all_cells"]["total"]) == (2, 1, 1, 20)


def test_printed_tables_show_counts_next_to_rates(tmp_path):
    root = make_root(tmp_path)
    _, out = run(root)
    assert "All cells" in out and "Without disputed cells (1 left out)" in out
    total_lines = [l for l in out.splitlines() if l.startswith("TOTAL")]
    assert len(total_lines) == 2
    first, second = (l.split() for l in total_lines)
    assert first[1:5] == ["2", "1", "1", "20"]
    assert second[1:5] == ["2", "0", "1", "20"]
    assert "0.67" in total_lines[0] and "2/3" in total_lines[0] and "20/21" in total_lines[0]
    assert "audited: 4  missing: 0  exact matches (all six verdicts): 2/4" in out
    assert "passed in all six: 1/2" in out


# ------------------------------------------------------------------- missing

def test_missing_conversations_warn_and_are_left_out(tmp_path):
    root = make_root(tmp_path)
    code, out = run(root, "all")
    assert code == 0
    assert "warning: 1 of 5 conversations in the split are not audited" in out
    m = metrics(root, "all")
    assert m["conversations"]["audited"] == 4 and m["conversations"]["missing"] == 1
    assert m["conversations"]["missing_ids"] == ["c5"]
    assert counts(m["all_cells"]["total"]) == (2, 1, 1, 20)


def test_partial_audits_are_scored_over_audited_ones(tmp_path):
    root = make_root(tmp_path)
    path = root / "data" / "audits" / "r1" / "audits.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + "\n")  # c1, c2 only
    code, out = run(root)
    assert "warning: 2 of 4" in out
    m = metrics(root)
    assert counts(m["all_cells"]["total"]) == (2, 1, 0, 9)
    assert m["conversations"]["key_clean"] == 0


def test_nothing_audited_in_split_is_an_error(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(SystemExit, match="no audited"):
        run(root, "check")


def test_audited_id_without_key_row_is_an_error(tmp_path):
    root = make_root(tmp_path)
    path = root / "data" / "batch3" / "labels_key.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")  # drop c4
    with pytest.raises(SystemExit, match="no key row"):
        run(root)


def test_unknown_run_is_an_error(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(SystemExit, match="run folder not found"):
        evaluate.main(["--run", "nope", "--split", "tune"], root=root, say=print)


# ------------------------------------------------------------------- disagreements

def read_disagreements(root, split="tune"):
    path = root / "data" / "audits" / "r1" / f"disagreements_{split}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_tune_prints_and_writes_disagreements(tmp_path):
    root = make_root(tmp_path)
    _, out = run(root)
    assert read_disagreements(root) == [
        {"id": "c2", "category": "false_resolution", "key_verdict": "fail",
         "auditor_verdict": "pass", "key_rules": ["FL-3"], "auditor_rules": [],
         "auditor_turns": [], "auditor_evidence": [], "auditor_why": []},
        {"id": "c3", "category": "aggravating_tone", "key_verdict": "pass",
         "auditor_verdict": "fail", "key_rules": [], "auditor_rules": ["TN-2"],
         "auditor_turns": [1, 3], "auditor_evidence": ["rude"], "auditor_why": ["snapped"]},
    ]
    assert "disagreements: 2" in out
    assert "c3  aggravating_tone  key=pass  auditor=fail" in out
    assert "auditor evidence: rude" in out


def test_check_and_all_hide_disagreements_by_default(tmp_path):
    root = make_root(tmp_path)
    audits = root / "data" / "audits" / "r1" / "audits.jsonl"
    _, out = run(root, "all")
    assert "disagreements" not in out
    assert not list(audits.parent.glob("disagreements_*"))
    assert (audits.parent / "metrics_all.json").is_file()


def test_show_disagreements_flag_prints_and_writes_for_all(tmp_path):
    root = make_root(tmp_path)
    _, out = run(root, "all", "--show-disagreements")
    assert "disagreements: 2" in out
    assert [r["id"] for r in read_disagreements(root, "all")] == ["c2", "c3"]


def test_metrics_file_holds_both_tables_for_each_split(tmp_path):
    root = make_root(tmp_path)
    run(root)
    run(root, "all")
    for split in ("tune", "all"):
        m = metrics(root, split)
        assert m["run"] == "r1" and m["split"] == split
        assert set(m) >= {"conversations", "all_cells", "excluding_disputed", "disputed_cells"}
