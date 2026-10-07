"""Auditor entry point: connectivity check, model listing and the audit run.

Run mode reads conversations only. It never reads labels; scoring is done by
retention/evaluate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from retention import config
from retention.generate import RULE_ID, InvalidOutput, parse_output, read_done_ids
from retention.llm import ChatClient, LLMError, RateLimited, TransientError

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("batch2", "batch3")
CATEGORIES = (
    "false_resolution",
    "missed_escalation",
    "unfulfillable_promise",
    "aggravating_tone",
    "premature_closure",
    "privacy_breach",
)
PING_MESSAGE = "Reply with OK"

DEFAULT_MAX_CALLS = 150
MAX_ATTEMPTS = 2            # attempts per conversation that end in invalid output
RATE_LIMIT_WAIT = 20        # seconds before retrying after a RateLimited
RATE_LIMIT_RETRIES = 3      # per conversation, not counted as attempts
TRANSIENT_WAIT = 10         # seconds before retrying after a TransientError
TRANSIENT_RETRIES = 1       # per conversation, not counted as attempts
FAIL_STREAK_LIMIT = 5       # conversations failed in a row before the run stops

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
RULE_TEXT_LIMIT = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- validation

def validate_audit(data: dict, n_turns: int, known_rules: set[str]) -> tuple[dict, list, list]:
    """Return (verdicts, findings, warnings) normalized, or raise InvalidOutput.

    Reasons are fixed text built from indices and the six category names, never
    from the model output, so they cannot carry conversation text.
    """
    verdicts_in = data.get("verdicts")
    if not isinstance(verdicts_in, dict) or set(verdicts_in) != set(CATEGORIES):
        raise InvalidOutput("'verdicts' must have exactly the six categories")
    verdicts = {}
    for cat in CATEGORIES:
        value = verdicts_in[cat]
        value = value.strip().lower() if isinstance(value, str) else None
        if value not in ("pass", "fail"):
            raise InvalidOutput(f"verdict for {cat} is not pass or fail")
        verdicts[cat] = value

    findings_in = data.get("findings")
    if not isinstance(findings_in, list):
        raise InvalidOutput("'findings' is not a list")
    findings, warnings = [], []
    for i, item in enumerate(findings_in, start=1):
        if not isinstance(item, dict):
            raise InvalidOutput(f"finding {i}: not an object")
        cat = item.get("category")
        if not isinstance(cat, str) or cat not in CATEGORIES:
            raise InvalidOutput(f"finding {i}: unknown category")
        rules = item.get("rules")
        if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
            raise InvalidOutput(f"finding {i}: 'rules' is not a list of strings")
        turns = item.get("turns")
        if not isinstance(turns, list) or not all(
                isinstance(t, int) and not isinstance(t, bool) for t in turns):
            raise InvalidOutput(f"finding {i}: 'turns' is not a list of integers")
        text = {}
        for key in ("evidence", "why"):
            value = item.get(key, "")
            if not isinstance(value, str):
                raise InvalidOutput(f"finding {i}: '{key}' is not a string")
            text[key] = value
        findings.append({"category": cat, "rules": rules, "turns": turns, **text})
        for rule in rules:
            if rule.strip() not in known_rules:
                warnings.append(f"finding {i}: rule not in policy: {rule.strip()[:RULE_TEXT_LIMIT]}")
        for t in turns:
            if not 1 <= t <= n_turns:
                warnings.append(f"finding {i}: turn {t} outside 1..{n_turns}")

    failed = {c for c, v in verdicts.items() if v == "fail"}
    with_finding = {f["category"] for f in findings}
    for cat in CATEGORIES:
        if cat in failed and cat not in with_finding:
            raise InvalidOutput(f"{cat} is marked fail without a finding")
        if cat in with_finding and cat not in failed:
            raise InvalidOutput(f"finding for {cat}, which is not marked fail")
    return verdicts, findings, warnings


# ---------------------------------------------------------------- data and prompt

def read_conversation_file(path: Path, seen: set[str]) -> list[dict]:
    """The conversations of one file. `seen` collects ids to reject duplicates."""
    if not path.is_file():
        raise SystemExit(f"error: required file not found: {path}")
    conversations: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            conv = json.loads(line)
            complete = all(k in conv for k in ("id", "channel", "account", "turns"))
        except (json.JSONDecodeError, TypeError):
            complete = False
        if not complete:
            raise SystemExit(f"error: bad conversation line in {path}")
        if conv["id"] in seen:
            raise SystemExit(f"error: duplicate conversation id {conv['id']}")
        seen.add(conv["id"])
        conversations.append(conv)
    return conversations


def load_conversations(data_dir: Path) -> list[dict]:
    """All conversations of DATASETS, in dataset order. Only conversation files are read."""
    conversations: list[dict] = []
    seen: set[str] = set()
    for name in DATASETS:
        conversations += read_conversation_file(data_dir / name / "conversations.jsonl", seen)
    return conversations


def select_conversations(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / "dev_split.json"
    if not path.is_file():
        raise SystemExit(f"error: required file not found: {path}")
    try:
        parts = json.loads(path.read_text(encoding="utf-8"))
        ids = parts["tune"] + parts["check"] if split == "all" else parts[split]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise SystemExit(f"error: {path.name} must hold 'tune' and 'check' id lists") from None
    conversations = load_conversations(data_dir)
    by_id = {c["id"]: c for c in conversations}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        shown = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise SystemExit(f"error: {len(unknown)} split id(s) not found in {', '.join(DATASETS)}: {shown}")
    wanted = set(ids)
    return [c for c in conversations if c["id"] in wanted]


def build_system_prompt(prompt_path: Path, policy_path: Path) -> tuple[str, str, str]:
    """Return (system prompt, template sha256, policy sha256)."""
    for path in (prompt_path, policy_path):
        if not path.is_file():
            raise SystemExit(f"error: required file not found: {path}")
    template = prompt_path.read_text(encoding="utf-8")
    policy = policy_path.read_text(encoding="utf-8")
    return template.replace("{policy}", policy), _sha256(template), _sha256(policy)


def build_user_message(conv: dict) -> str:
    """The conversation without its id."""
    payload = {
        "channel": conv["channel"],
        "account": conv["account"],
        "turns": [{"n": n, "speaker": t["speaker"], "text": t["text"]}
                  for n, t in enumerate(conv["turns"], start=1)],
    }
    return json.dumps(payload, ensure_ascii=False)


def policy_rule_ids(policy_text: str) -> set[str]:
    return {m.group(0) for m in RULE_ID.finditer(policy_text)}


# ---------------------------------------------------------------- run metadata

def check_run_meta(run_path: Path, current: dict) -> bool:
    """True if run.json exists and matches. Exit if it exists and differs."""
    if not run_path.is_file():
        return False
    try:
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        same = all(saved[k] == current[k] for k in ("model", "prompt_sha256", "policy_sha256"))
    except (json.JSONDecodeError, KeyError, TypeError):
        same = False
    if not same:
        raise SystemExit(f"run {current['run']} was started with a different model, "
                         "prompt or policy; use a new run name")
    return True


def write_run_meta(run_path: Path, meta: dict) -> None:
    run_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = run_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, run_path)


# ---------------------------------------------------------------- run

class _Stats:
    def __init__(self):
        self.ok = self.failed = self.skipped = 0
        self.calls = self.input_tokens = self.output_tokens = 0
        self.stopped = False


def run_audits(pending, client, model, system_prompt, known_rules, out_dir: Path,
               max_calls: int, sleep=time.sleep, now=_utc_now, say=print) -> _Stats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _Stats()
    streak = 0
    total = len(pending)

    with open(out_dir / "audits.jsonl", "a", encoding="utf-8") as out_f, \
            open(out_dir / "log.jsonl", "a", encoding="utf-8") as log_f:

        def log(conv_id, attempt, status, error, usage):
            row = {"ts": now(), "id": conv_id, "attempt": attempt, "status": status,
                   "error": error, "input_tokens": usage["input_tokens"],
                   "output_tokens": usage["output_tokens"]}
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()

        def audit_one(conv) -> str:
            """Return "ok", "failed" or "skipped"."""
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_message(conv)}]
            none = {"input_tokens": None, "output_tokens": None}
            attempts = rate_retries = transient_retries = 0
            wait = 0
            while True:
                if stats.calls >= max_calls:
                    return "skipped"
                if wait:
                    sleep(wait)
                    wait = 0
                stats.calls += 1
                try:
                    text, usage = client.complete_with_usage(model, messages)
                except RateLimited as exc:
                    log(conv["id"], attempts + 1, "error", str(exc), none)
                    if rate_retries >= RATE_LIMIT_RETRIES:
                        return "failed"
                    rate_retries += 1
                    wait = RATE_LIMIT_WAIT
                    continue
                except TransientError as exc:
                    log(conv["id"], attempts + 1, "error", str(exc), none)
                    if transient_retries >= TRANSIENT_RETRIES:
                        return "failed"
                    transient_retries += 1
                    wait = TRANSIENT_WAIT
                    continue
                except LLMError as exc:
                    log(conv["id"], attempts + 1, "error", str(exc), none)
                    return "failed"
                attempts += 1
                stats.input_tokens += usage["input_tokens"] or 0
                stats.output_tokens += usage["output_tokens"] or 0
                try:
                    verdicts, findings, warnings = validate_audit(
                        parse_output(text), len(conv["turns"]), known_rules)
                except InvalidOutput as exc:
                    log(conv["id"], attempts, "invalid", str(exc), usage)
                    if attempts >= MAX_ATTEMPTS:
                        return "failed"
                    continue
                record = {"id": conv["id"], "verdicts": verdicts, "findings": findings,
                          "warnings": warnings, "model": model,
                          "input_tokens": usage["input_tokens"],
                          "output_tokens": usage["output_tokens"]}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                log(conv["id"], attempts, "ok", None, usage)
                return "ok"

        for n, conv in enumerate(pending, start=1):
            if stats.stopped:
                status = "skipped"
            else:
                status = audit_one(conv)
                if status == "failed":
                    streak += 1
                    stats.stopped = streak >= FAIL_STREAK_LIMIT
                elif status == "ok":
                    streak = 0
            setattr(stats, status, getattr(stats, status) + 1)
            say(f"[{n}/{total}] {conv['id']} {status}")
    return stats


def print_summary(stats: _Stats, max_calls: int, say=print) -> None:
    say(f"ok: {stats.ok}  failed: {stats.failed}  skipped: {stats.skipped}")
    say(f"calls: {stats.calls}/{max_calls}  input tokens: {stats.input_tokens}"
        f"  output tokens: {stats.output_tokens}")
    if stats.stopped:
        say(f"stopped: {FAIL_STREAK_LIMIT} conversations failed in a row")
    elif stats.skipped:
        say("stopped: call budget used up")


def run_mode(args, client, model, root: Path, sleep, now, say=print) -> int:
    data_dir = root / "data"
    system_prompt, prompt_sha, policy_sha = build_system_prompt(
        root / "prompts" / "audit_conversation.md", root / "docs" / "policy.md")
    if args.conversations is not None:
        selected = read_conversation_file(Path(args.conversations), set())
    else:
        selected = select_conversations(data_dir, args.split)

    out_dir = data_dir / "audits" / args.run
    meta = {"run": args.run, "model": model, "prompt_sha256": prompt_sha,
            "policy_sha256": policy_sha, "created": now()}
    meta_exists = check_run_meta(out_dir / "run.json", meta)

    done = read_done_ids(out_dir / "audits.jsonl")
    pending = [c for c in selected if c["id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    n_done = sum(c["id"] in done for c in selected)

    if args.dry_run:
        estimate = sum((len(system_prompt) + len(build_user_message(c))) // 4 for c in pending)
        say(f"selected: {len(selected)}  done: {n_done}  pending: {len(pending)}")
        say(f"estimated input tokens: {estimate}")
        return 0

    say(f"selected: {len(selected)}  done: {n_done}  to attempt: {len(pending)}")
    if not pending:
        return 0
    if not meta_exists:
        write_run_meta(out_dir / "run.json", meta)
    stats = run_audits(pending, client(), model, system_prompt, policy_rule_ids(
        (root / "docs" / "policy.md").read_text(encoding="utf-8")), out_dir,
        args.max_calls, sleep, now, say)
    print_summary(stats, args.max_calls, say)
    return 1 if stats.stopped else 0


# ---------------------------------------------------------------- other modes

def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m retention.audit")
    p.add_argument("--ping", action="store_true", help="one call to AUDITOR_MODEL")
    p.add_argument("--list-models", action="store_true", help="list model ids")
    p.add_argument("--contains", metavar="TEXT",
                   help="with --list-models: only ids containing TEXT (case-insensitive)")
    p.add_argument("--run", metavar="NAME", help="audit a split; output in data/audits/NAME/")
    p.add_argument("--split", choices=("tune", "check", "all"), help="with --run: ids to audit")
    p.add_argument("--conversations", metavar="PATH",
                   help="with --run, instead of --split: audit every conversation in this file")
    p.add_argument("--limit", type=int, metavar="N", help="with --run: audit at most N pending")
    p.add_argument("--max-calls", type=int, metavar="N",
                   help=f"with --run: call budget (default {DEFAULT_MAX_CALLS})")
    p.add_argument("--dry-run", action="store_true",
                   help="with --run: show counts and a token estimate, call nothing")
    args = p.parse_args(argv)
    run_only = (args.split, args.conversations, args.limit, args.max_calls, args.dry_run or None)
    modes = [args.ping, args.list_models, args.run is not None]
    if (sum(modes) != 1
            or (args.contains is not None and not args.list_models)
            or (args.run is None and any(v is not None for v in run_only))):
        p.print_usage()
        return p, None
    if args.run is not None:
        if (args.split is None) == (args.conversations is None):
            p.error("--run needs exactly one of --split and --conversations")
        if not RUN_NAME.fullmatch(args.run):
            p.error("--run NAME may only hold letters, digits, '.', '_' and '-'")
        for flag in ("limit", "max_calls"):
            if getattr(args, flag) is not None and getattr(args, flag) < 1:
                p.error(f"--{flag.replace('_', '-')} must be at least 1")
        if args.max_calls is None:
            args.max_calls = DEFAULT_MAX_CALLS
    return p, args


def ping(client, model: str) -> int:
    try:
        client.complete(model, [{"role": "user", "content": PING_MESSAGE}])
    except LLMError as exc:
        status = f" {exc.status}" if exc.status else ""
        detail = " ".join(exc.detail.split())[:200]
        print(f"{model}: error {exc.kind}{status} {detail}".rstrip())
        return 1
    print(f"{model}: ok")
    return 0


def list_models(client, contains: str | None) -> int:
    try:
        ids = client.list_models()
    except LLMError as exc:
        print(f"error: {exc}")
        return 1
    if contains is not None:
        ids = [m for m in ids if contains.lower() in m.lower()]
    for model_id in sorted(ids):
        print(model_id)
    return 0


def main(argv=None, client=None, env_path=None, root: Path = ROOT,
         sleep=time.sleep, now=_utc_now) -> int:
    _, args = _parse_args(argv)
    if args is None:
        return 2
    try:
        base_url, api_key, model = config.load_auditor_settings(env_path)
    except config.ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    if args.run is not None:
        return run_mode(args, lambda: client or ChatClient(base_url, api_key),
                        model, Path(root), sleep, now)
    client = client or ChatClient(base_url, api_key)
    if args.ping:
        return ping(client, model)
    return list_models(client, args.contains)


if __name__ == "__main__":
    sys.exit(main())
