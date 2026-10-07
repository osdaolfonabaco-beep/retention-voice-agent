"""Score an audit run against the planted-failure key.

Run: python -m retention.evaluate --run NAME --split {tune,check,all}

This is the only module that reads labels. retention/audit.py never does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from retention.audit import CATEGORIES, DATASETS

ROOT = Path(__file__).resolve().parent.parent
Z = 1.96  # 95% Wilson interval


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = Z) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None,
            "low": ci[0] if ci else None, "high": ci[1] if ci else None}


# ---------------------------------------------------------------- loading

def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise SystemExit(f"error: bad JSON line in {path}") from None
    return rows


def load_split_ids(data_dir: Path, split: str) -> list[str]:
    path = data_dir / "dev_split.json"
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    parts = json.loads(path.read_text(encoding="utf-8"))
    return parts["tune"] + parts["check"] if split == "all" else parts[split]


def load_key(data_dir: Path) -> dict[str, dict]:
    key: dict[str, dict] = {}
    for name in DATASETS:
        for row in _read_jsonl(data_dir / name / "labels_key.jsonl"):
            key[row["id"]] = row
    return key


def load_audits(path: Path) -> dict[str, dict]:
    audits: dict[str, dict] = {}
    for row in _read_jsonl(path):
        audits.setdefault(row["id"], row)
    return audits


# ---------------------------------------------------------------- cells

def _rules_for(findings: list[dict], category: str) -> list[str]:
    """Rule IDs of the findings in one category, upper-cased, first-seen order."""
    seen: list[str] = []
    for f in findings:
        if f.get("category") != category:
            continue
        for rule in f.get("rules", []):
            rule = rule.strip().upper()
            if rule not in seen:
                seen.append(rule)
    return seen


def build_cells(ids: list[str], audits: dict, key: dict) -> list[dict]:
    cells = []
    for cid in ids:
        k, a = key[cid], audits[cid]
        disputed = {d["category"] for d in k.get("disputed", [])}
        for cat in CATEGORIES:
            cells.append({
                "id": cid, "category": cat,
                "key_fail": k["verdicts"][cat] == "fail",
                "auditor_fail": a["verdicts"][cat] == "fail",
                "disputed": cat in disputed,
                "key_rules": _rules_for(k.get("findings", []), cat),
                "auditor_rules": _rules_for(a.get("findings", []), cat),
            })
    return cells


def _count(cells: list[dict]) -> dict:
    tp = sum(c["key_fail"] and c["auditor_fail"] for c in cells)
    fn = sum(c["key_fail"] and not c["auditor_fail"] for c in cells)
    fp = sum(not c["key_fail"] and c["auditor_fail"] for c in cells)
    tn = sum(not c["key_fail"] and not c["auditor_fail"] for c in cells)
    agree = sum(bool(set(c["key_rules"]) & set(c["auditor_rules"]))
                for c in cells if c["key_fail"] and c["auditor_fail"])
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": rate(tp, tp + fn), "tnr": rate(tn, tn + fp),
            "rule_agreement": rate(agree, tp)}


def table(cells: list[dict]) -> dict:
    return {"per_category": {cat: _count([c for c in cells if c["category"] == cat])
                             for cat in CATEGORIES},
            "total": _count(cells)}


def conversation_level(ids: list[str], audits: dict, key: dict, missing: list[str]) -> dict:
    exact = clean = clean_passed = 0
    for cid in ids:
        k, a = key[cid]["verdicts"], audits[cid]["verdicts"]
        exact += all(k[c] == a[c] for c in CATEGORIES)
        if all(k[c] == "pass" for c in CATEGORIES):
            clean += 1
            clean_passed += all(a[c] == "pass" for c in CATEGORIES)
    return {"audited": len(ids), "missing": len(missing), "missing_ids": missing,
            "exact_matches": exact, "key_clean": clean, "key_clean_passed_all": clean_passed}


def disagreements(cells: list[dict], audits: dict) -> list[dict]:
    rows = []
    for c in cells:
        if c["key_fail"] == c["auditor_fail"]:
            continue
        mine = [f for f in audits[c["id"]].get("findings", []) if f.get("category") == c["category"]]
        rows.append({
            "id": c["id"], "category": c["category"],
            "key_verdict": "fail" if c["key_fail"] else "pass",
            "auditor_verdict": "fail" if c["auditor_fail"] else "pass",
            "key_rules": c["key_rules"], "auditor_rules": c["auditor_rules"],
            "auditor_turns": sorted({t for f in mine for t in f.get("turns", [])}),
            "auditor_evidence": [f.get("evidence", "") for f in mine],
            "auditor_why": [f.get("why", "") for f in mine],
        })
    return rows


# ---------------------------------------------------------------- output

def _fmt(r: dict) -> str:
    if r["rate"] is None:
        return f"{'n/a':>22}"
    return f"{r['rate']:.2f} [{r['low']:.2f}-{r['high']:.2f}] {r['k']:>2}/{r['n']:<3}"


def print_table(title: str, t: dict, say=print) -> None:
    say(f"\n{title}")
    say(f"{'category':<22}{'TP':>4}{'FN':>4}{'FP':>4}{'TN':>4}   "
        f"{'TPR':<22}  {'TNR':<22}  {'rule agreement':<18}")
    rows = [(cat, t["per_category"][cat]) for cat in CATEGORIES] + [("TOTAL", t["total"])]
    for name, m in rows:
        say(f"{name:<22}{m['tp']:>4}{m['fn']:>4}{m['fp']:>4}{m['tn']:>4}   "
            f"{_fmt(m['tpr'])}  {_fmt(m['tnr'])}  {_fmt(m['rule_agreement'])}")


def print_disagreements(rows: list[dict], say=print) -> None:
    say(f"\ndisagreements: {len(rows)}")
    for r in rows:
        say(f"\n{r['id']}  {r['category']}  key={r['key_verdict']}  auditor={r['auditor_verdict']}")
        say(f"  key rules: {', '.join(r['key_rules']) or '-'}")
        say(f"  auditor rules: {', '.join(r['auditor_rules']) or '-'}")
        say(f"  auditor turns: {', '.join(map(str, r['auditor_turns'])) or '-'}")
        say(f"  auditor evidence: {' | '.join(r['auditor_evidence']) or '-'}")
        say(f"  auditor why: {' | '.join(r['auditor_why']) or '-'}")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- CLI

def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m retention.evaluate")
    p.add_argument("--run", required=True, metavar="NAME")
    p.add_argument("--split", required=True, choices=("tune", "check", "all"))
    p.add_argument("--show-disagreements", action="store_true",
                   help="print and write disagreements (always done for tune)")
    return p.parse_args(argv)


def main(argv=None, root: Path = ROOT, say=print) -> int:
    args = _parse_args(argv)
    data_dir = Path(root) / "data"
    out_dir = data_dir / "audits" / args.run
    if not out_dir.is_dir():
        raise SystemExit(f"error: run folder not found: {out_dir}")

    split_ids = load_split_ids(data_dir, args.split)
    audits = load_audits(out_dir / "audits.jsonl")
    key = load_key(data_dir)
    ids = [i for i in split_ids if i in audits]
    missing = [i for i in split_ids if i not in audits]
    no_key = [i for i in ids if i not in key]
    if no_key:
        raise SystemExit(f"error: no key row for {len(no_key)} audited id(s), e.g. {no_key[0]}")
    if missing:
        say(f"warning: {len(missing)} of {len(split_ids)} conversations in the split "
            "are not audited; computed over the audited ones")
    if not ids:
        raise SystemExit("error: no audited conversation in this split")

    cells = build_cells(ids, audits, key)
    conv = conversation_level(ids, audits, key, missing)
    metrics = {
        "run": args.run, "split": args.split,
        "conversations": conv,
        "all_cells": table(cells),
        "excluding_disputed": table([c for c in cells if not c["disputed"]]),
        "disputed_cells": sum(c["disputed"] for c in cells),
    }

    say(f"run {args.run}  split {args.split}")
    say(f"audited: {conv['audited']}  missing: {conv['missing']}  "
        f"exact matches (all six verdicts): {conv['exact_matches']}/{conv['audited']}")
    say(f"key's no-failure conversations passed in all six: "
        f"{conv['key_clean_passed_all']}/{conv['key_clean']}")
    print_table("All cells", metrics["all_cells"], say)
    print_table(f"Without disputed cells ({metrics['disputed_cells']} left out)",
                metrics["excluding_disputed"], say)

    _write_json(out_dir / f"metrics_{args.split}.json", metrics)
    if args.split == "tune" or args.show_disagreements:
        rows = disagreements(cells, audits)
        (out_dir / f"disagreements_{args.split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print_disagreements(rows, say)
    return 0


if __name__ == "__main__":
    sys.exit(main())
