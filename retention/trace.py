"""Trace check of an agent run: what the agent sent to the tools, against what the
customer had said. Reads conversations.jsonl and tool_calls.jsonl of the run, and
with --audit also audits.jsonl of an audit run. Never reads labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retention.audit import RUN_NAME
from retention.report import load_audits
from retention.tools import ungrounded

ROOT = Path(__file__).resolve().parent.parent
VIOLATIONS = ("ungrounded_call", "ended_on_question")
GUARDS = ("asked_a_question", "not_said_by_customer")


def _load(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise SystemExit(f"error: required file not found: {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ok = isinstance(row["id"], str) and isinstance(
                row.get("turns", row.get("calls")), list)
        except (json.JSONDecodeError, KeyError, TypeError):
            ok = False
        if not ok:
            raise SystemExit(f"error: bad line in {path}")
        rows[row["id"]] = row
    return rows


def _guard_of(result) -> str | None:
    """The guard that refused a call, or None when the call was processed."""
    if isinstance(result, dict):
        if "blocked" in result:
            return result["blocked"]
        if result.get("reason") == "not_said_by_customer":
            return "not_said_by_customer"
    return None


def check_contact(turns: list[dict], calls: list[dict]) -> tuple[list[dict], dict]:
    """Return (violations, calls blocked per guard) for one contact."""
    violations, blocked = [], {g: 0 for g in GUARDS}
    for call in calls:
        guard = _guard_of(call["result"])
        if guard is not None:
            if guard in blocked:
                blocked[guard] += 1
            continue
        k = call["after_turn"]
        said = [t["text"] for t in turns[:k] if t["speaker"] == "customer"]
        missing = ungrounded(call["name"], call["arguments"], said)
        if missing:
            violations.append({"type": "ungrounded_call", "tool": call["name"],
                               "after_turn": k, "params": missing})
        if call["name"] == "end_contact":
            agent_turn = turns[k] if k < len(turns) and turns[k]["speaker"] == "agent" else None
            if agent_turn and "?" in agent_turn["text"]:
                violations.append({"type": "ended_on_question", "tool": "end_contact",
                                   "after_turn": k})
    return violations, blocked


def trace_run(run: str, conversations: dict[str, dict], tool_calls: dict[str, dict],
              audits: list[dict] | None = None, audit_run: str | None = None) -> dict:
    if set(conversations) != set(tool_calls):
        raise SystemExit("error: conversations.jsonl and tool_calls.jsonl hold different ids")
    contacts, blocked = [], {g: 0 for g in GUARDS}
    for cid, conv in conversations.items():
        violations, counts = check_contact(conv["turns"], tool_calls[cid]["calls"])
        contacts.append({"id": cid, "violations": violations})
        for guard, n in counts.items():
            blocked[guard] += n
    total = len(contacts)
    result = {"run": run, "contacts": contacts, "totals": {
        "contacts": total,
        **{v: sum(any(x["type"] == v for x in c["violations"]) for c in contacts)
           for v in VIOLATIONS},
        "no_violation": sum(not c["violations"] for c in contacts),
        "blocked": blocked}}
    if audits is not None:
        if {a["id"] for a in audits} != set(conversations):
            raise SystemExit(f"error: audit run {audit_run} and run {run} hold different ids")
        failed = {a["id"] for a in audits if "fail" in a["verdicts"].values()}
        ids = [c["id"] for c in contacts if not c["violations"] and c["id"] not in failed]
        result["audit"] = {"run": audit_run, "clean_both": {
            "ids": ids, "count": len(ids), "pct": round(100 * len(ids) / total, 1) if total else 0.0}}
    return result


def _describe(v: dict) -> str:
    params = f": {', '.join(v['params'])}" if v.get("params") else ""
    return f"{v['type']}[{v['tool']}@{v['after_turn']}{params}]"


def print_trace(result: dict, say=print) -> None:
    for c in result["contacts"]:
        say(f"{c['id']}  " + ("; ".join(_describe(v) for v in c["violations"]) or "none"))
    t = result["totals"]
    say("")
    say(f"contacts: {t['contacts']}")
    for v in VIOLATIONS:
        say(f"contacts with {v}: {t[v]}")
    say(f"contacts with no violation: {t['no_violation']}")
    for guard, n in t["blocked"].items():
        say(f"calls blocked ({guard}): {n}")
    if "audit" in result:
        both = result["audit"]["clean_both"]
        say(f"contacts with no failed category in {result['audit']['run']} and no violation: "
            f"{both['count']} ({both['pct']}%)")
        for cid in both["ids"]:
            say(f"  {cid}")


def main(argv=None, root: Path = ROOT, say=print) -> int:
    p = argparse.ArgumentParser(prog="python -m retention.trace")
    p.add_argument("--run", metavar="NAME", required=True,
                   help="check data/agent_runs/NAME/")
    p.add_argument("--audit", metavar="AUDIT_RUN",
                   help="also list contacts clean in data/audits/AUDIT_RUN/audits.jsonl")
    args = p.parse_args(argv)
    for flag in ("run", "audit"):
        if getattr(args, flag) is not None and not RUN_NAME.fullmatch(getattr(args, flag)):
            p.error(f"--{flag} NAME may only hold letters, digits, '.', '_' and '-'")
    run_dir = Path(root) / "data" / "agent_runs" / args.run
    audits = (load_audits(Path(root) / "data" / "audits" / args.audit / "audits.jsonl")
              if args.audit else None)
    result = trace_run(args.run, _load(run_dir / "conversations.jsonl"),
                       _load(run_dir / "tool_calls.jsonl"), audits, args.audit)
    print_trace(result, say)
    (run_dir / "trace.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
