"""Trace check of an agent run: what the agent sent to the tools, against what the
customer had said. Reads conversations.jsonl and tool_calls.jsonl of the run, and
with --audit also audits.jsonl of an audit run. With --expect it checks the
escalations against an expected_escalations file, and with --outcomes the effects of
each contact against an expected_outcomes file. Never reads labels."""

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


# ---------------------------------------------------------------- expectations

def _load_json(path: Path, what: str) -> dict:
    if not Path(path).is_file():
        raise SystemExit(f"error: required file not found: {path}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise SystemExit(f"error: {what} file {path} must map contact ids to objects")
    return data


def load_expected_escalations(path: Path) -> dict[str, dict]:
    data = _load_json(path, "escalations")
    for cid, spec in data.items():
        if not (isinstance(spec.get("reason"), str) and isinstance(spec.get("team"), str)):
            raise SystemExit(f"error: bad escalation for {cid} in {path}")
    return data


def load_expected_outcomes(path: Path) -> dict[str, dict]:
    data = _load_json(path, "outcomes")
    for cid, spec in data.items():
        lists = all(isinstance(spec.get(k, []), list)
                    and all(isinstance(x, str) for x in spec.get(k, []))
                    for k in ("must", "must_not"))
        number = spec.get("max_installment") is None or (
            isinstance(spec["max_installment"], (int, float))
            and not isinstance(spec["max_installment"], bool))
        date = spec.get("last_date") is None or isinstance(spec["last_date"], str)
        if not (lists and number and date):
            raise SystemExit(f"error: bad outcome for {cid} in {path}")
    return data


def _processed(calls: list[dict]):
    """Calls that ran: not discarded by the ask-then-wait or the grounding guard."""
    return [c for c in calls if _guard_of(c["result"]) is None and isinstance(c["result"], dict)]


def _ticket_calls(calls: list[dict]) -> list[dict]:
    return [c for c in _processed(calls) if c["name"] == "escalate" and "ticket" in c["result"]]


def check_escalations(expected: dict[str, dict], tool_calls: dict[str, dict]) -> dict:
    """Required escalations made correctly: a processed escalate call that returned a
    ticket, with the expected reason and team. Contacts not listed must not escalate.
    Ids of the file that are not in the run are listed apart and not counted."""
    required = [cid for cid in expected if cid in tool_calls]
    wrong, unneeded = [], []
    for cid in required:
        spec = expected[cid]
        if not any(isinstance(c["arguments"], dict)
                   and c["arguments"].get("reason") == spec["reason"]
                   and c["arguments"].get("team") == spec["team"]
                   for c in _ticket_calls(tool_calls[cid]["calls"])):
            wrong.append(cid)
    for cid, row in tool_calls.items():
        if cid not in expected and _ticket_calls(row["calls"]):
            unneeded.append(cid)
    return {"required": len(required), "correct": len(required) - len(wrong),
            "missed_or_wrong": wrong, "unneeded": unneeded,
            "not_in_run": [cid for cid in expected if cid not in tool_calls]}


def _amount(value) -> str:
    value = float(value)
    return str(int(value)) if value == int(value) else repr(round(value, 2))


def effects(calls: list[dict]) -> list[str]:
    """What a contact changed, from its processed, successful calls."""
    found = []
    for c in _processed(calls):
        name, result = c["name"], c["result"]
        args = c["arguments"] if isinstance(c["arguments"], dict) else {}
        if name == "apply_offer" and result.get("applied") is True:
            details = result.get("details") or {}
            offer = details.get("offer") or args.get("offer")
            plan = details.get("new_plan") or args.get("target_plan")
            found.append(f"offer:{offer}" + (f":{plan}" if offer == "downgrade" and plan else ""))
        elif name == "set_payment_plan" and result.get("set") is True:
            found.append("plan")
        elif name == "cancel_service" and result.get("cancelled") is True:
            found.append("cancel")
        elif name == "take_payment" and result.get("received") is True:
            found.append(f"payment:{_amount(args['amount'])}" if "amount" in args else "payment")
        elif name == "escalate" and "ticket" in result:
            found.append(f"escalate:{args.get('reason')}:{result.get('team') or args.get('team')}")
        elif name == "verify_identity" and result.get("result") in (
                "verified_account_holder", "verified_authorized_user"):
            found.append("verified")
    return found


def _matches(expected: str, effect: str) -> bool:
    return effect == expected or effect.startswith(expected + ":")


def check_outcome(spec: dict, calls: list[dict]) -> list[str]:
    """What fails in one contact's outcome; [] when it is correct."""
    done, failed = effects(calls), []
    failed += [f"missing {m}" for m in spec.get("must", [])
               if not any(_matches(m, e) for e in done)]
    failed += [f"forbidden {m}" for m in spec.get("must_not", [])
               if any(_matches(m, e) for e in done)]
    plans = [c for c in _processed(calls) if c["name"] == "set_payment_plan"
             and c["result"].get("set") is True]
    if plans:
        installments = plans[-1]["result"].get("installments") or []
        limit = spec.get("max_installment")
        if limit is not None and any(i["amount"] > limit for i in installments):
            failed.append(f"installment above {_amount(limit)}")
        last = spec.get("last_date")
        if last is not None and any(i["date"] > last for i in installments):
            failed.append(f"plan ends after {last}")
    for kind in ("payment", "cancel", "escalate"):
        n = sum(_matches(kind, e) for e in done)
        if n > 1:
            failed.append(f"{n} {kind} effects")
    return failed


def check_outcomes(expected: dict[str, dict], tool_calls: dict[str, dict]) -> dict:
    contacts = [{"id": cid, "failed": check_outcome(expected[cid], row["calls"])}
                for cid, row in tool_calls.items() if cid in expected]
    return {"total": len(contacts), "correct": sum(not c["failed"] for c in contacts),
            "contacts": contacts,
            "no_expectation": [cid for cid in tool_calls if cid not in expected],
            "not_in_run": [cid for cid in expected if cid not in tool_calls]}


def trace_run(run: str, conversations: dict[str, dict], tool_calls: dict[str, dict],
              audits: list[dict] | None = None, audit_run: str | None = None,
              escalations: dict[str, dict] | None = None,
              outcomes: dict[str, dict] | None = None) -> dict:
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
    if escalations is not None:
        result["escalations"] = check_escalations(escalations, tool_calls)
    if outcomes is not None:
        result["outcomes"] = check_outcomes(outcomes, tool_calls)
    if audits is not None:
        if {a["id"] for a in audits} != set(conversations):
            raise SystemExit(f"error: audit run {audit_run} and run {run} hold different ids")
        failed = {a["id"] for a in audits if "fail" in a["verdicts"].values()}

        def share(ids):
            return {"ids": ids, "count": len(ids),
                    "pct": round(100 * len(ids) / total, 1) if total else 0.0}

        ids = [c["id"] for c in contacts if not c["violations"] and c["id"] not in failed]
        result["audit"] = {"run": audit_run, "clean_both": share(ids)}
        if outcomes is not None:
            right = {c["id"] for c in result["outcomes"]["contacts"] if not c["failed"]}
            result["audit"]["clean_all"] = share([i for i in ids if i in right])
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
        if "clean_all" in result["audit"]:
            every = result["audit"]["clean_all"]
            say(f"contacts clean in {result['audit']['run']}, with no violation and a correct "
                f"outcome: {every['count']} ({every['pct']}%)")
            for cid in every["ids"]:
                say(f"  {cid}")
    if "escalations" in result:
        e = result["escalations"]
        say("")
        say(f"required escalations made correctly: {e['correct']} of {e['required']}")
        say(f"required escalations missed or wrong: {', '.join(e['missed_or_wrong']) or 'none'}")
        say(f"escalated without need: {', '.join(e['unneeded']) or 'none'}")
        if e["not_in_run"]:
            say(f"expected but not in the run: {', '.join(e['not_in_run'])}")
    if "outcomes" in result:
        o = result["outcomes"]
        say("")
        say(f"correct outcomes: {o['correct']} of {o['total']}")
        for c in o["contacts"]:
            if c["failed"]:
                say(f"  {c['id']}: {'; '.join(c['failed'])}")
        if o["no_expectation"]:
            say(f"no expected outcome (not counted): {', '.join(o['no_expectation'])}")
        if o["not_in_run"]:
            say(f"expected but not in the run: {', '.join(o['not_in_run'])}")


def main(argv=None, root: Path = ROOT, say=print) -> int:
    p = argparse.ArgumentParser(prog="python -m retention.trace")
    p.add_argument("--run", metavar="NAME", required=True,
                   help="check data/agent_runs/NAME/")
    p.add_argument("--audit", metavar="AUDIT_RUN",
                   help="also list contacts clean in data/audits/AUDIT_RUN/audits.jsonl")
    p.add_argument("--expect", metavar="PATH",
                   help="expected escalations (contact id -> reason and team); check them")
    p.add_argument("--outcomes", metavar="PATH",
                   help="expected outcomes (contact id -> must, must_not, ...); check them")
    args = p.parse_args(argv)
    for flag in ("run", "audit"):
        if getattr(args, flag) is not None and not RUN_NAME.fullmatch(getattr(args, flag)):
            p.error(f"--{flag} NAME may only hold letters, digits, '.', '_' and '-'")
    run_dir = Path(root) / "data" / "agent_runs" / args.run
    audits = (load_audits(Path(root) / "data" / "audits" / args.audit / "audits.jsonl")
              if args.audit else None)
    result = trace_run(
        args.run, _load(run_dir / "conversations.jsonl"), _load(run_dir / "tool_calls.jsonl"),
        audits, args.audit,
        load_expected_escalations(Path(args.expect)) if args.expect else None,
        load_expected_outcomes(Path(args.outcomes)) if args.outcomes else None)
    print_trace(result, say)
    (run_dir / "trace.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
