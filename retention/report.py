"""Summarize an audit run: counts and every finding. Reads audits.jsonl only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retention.audit import CATEGORIES, RUN_NAME

ROOT = Path(__file__).resolve().parent.parent


def load_audits(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"error: required file not found: {path}")
    audits = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            valid = (isinstance(row["id"], str) and isinstance(row["findings"], list)
                     and set(row["verdicts"]) == set(CATEGORIES))
        except (json.JSONDecodeError, KeyError, TypeError):
            valid = False
        if not valid:
            raise SystemExit(f"error: bad audit line in {path}")
        audits.append(row)
    if not audits:
        raise SystemExit(f"error: no audits in {path}")
    return audits


def summarize(run: str, audits: list[dict]) -> dict:
    failed = {cat: sum(a["verdicts"][cat] == "fail" for a in audits) for cat in CATEGORIES}
    clean = sum(all(v == "pass" for v in a["verdicts"].values()) for a in audits)
    return {"run": run, "audited": len(audits), "clean": clean,
            "clean_pct": round(100 * clean / len(audits), 1),
            "failed_by_category": failed,
            "findings": sum(len(a["findings"]) for a in audits)}


def print_report(summary: dict, audits: list[dict], say=print) -> None:
    say(f"conversations audited: {summary['audited']}")
    say(f"with no failed category: {summary['clean']} ({summary['clean_pct']}%)")
    say("failed conversations per category:")
    for cat, count in summary["failed_by_category"].items():
        say(f"  {cat}: {count}")
    say(f"findings: {summary['findings']}")
    for audit in audits:
        for f in audit["findings"]:
            say("")
            say(f"{audit['id']}  {f['category']}  rules: {', '.join(f['rules']) or '-'}  "
                f"turns: {', '.join(str(t) for t in f['turns']) or '-'}")
            say(f"  evidence: {f['evidence']}")
            say(f"  why: {f['why']}")


def main(argv=None, root: Path = ROOT, say=print) -> int:
    p = argparse.ArgumentParser(prog="python -m retention.report")
    p.add_argument("--run", metavar="NAME", required=True,
                   help="report on data/audits/NAME/audits.jsonl")
    args = p.parse_args(argv)
    if not RUN_NAME.fullmatch(args.run):
        p.error("--run NAME may only hold letters, digits, '.', '_' and '-'")
    run_dir = Path(root) / "data" / "audits" / args.run
    audits = load_audits(run_dir / "audits.jsonl")
    summary = summarize(args.run, audits)
    print_report(summary, audits, say)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
