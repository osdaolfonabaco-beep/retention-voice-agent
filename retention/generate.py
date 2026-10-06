"""Generate one synthetic retention conversation per sealed scenario card.

Run: python -m retention.generate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from retention import config
from retention.llm import ChatClient, LLMError, RateLimited, TransientError
from retention.sealed import CardError, load_cards

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "generate_conversation.md"
POLICY_PATH = ROOT / "docs" / "policy.md"

# Pacing and budget per model. Keys are normalized names ("models/" stripped).
# "*" applies to any other model. Budgets are per run.
PACING = {
    "gemini-3.8-flash": {"interval": 13, "max_calls": 20},
    "gemini-3.5-flash-lite": {"interval": 5, "max_calls": 100},
    "*": {"interval": 13, "max_calls": 20},
}
MAX_ATTEMPTS = 2
TRANSIENT_WAIT = 30   # seconds to wait before the retry that follows a TransientError
TRANSIENT_LIMIT = 3   # consecutive cards failed by TransientError before a model is dropped

RULE_ID = re.compile(r"\b(VR|FL|OF|PP|PR|ES|CL|TN)-\d+\b")
SPEAKERS = {"agent", "customer"}


def norm_model(name: str) -> str:
    """Strip a leading 'models/' so both spellings share one pacing entry."""
    return name[len("models/"):] if name.startswith("models/") else name


def limits_for(key: str) -> dict:
    return PACING.get(key, PACING["*"])


class InvalidOutput(Exception):
    """The model output failed parsing or validation. The reason never holds card text."""


# ---------------------------------------------------------------- parsing

def parse_output(text: str) -> dict:
    t = text.strip()
    fence = re.match(r"^```[A-Za-z0-9]*[ \t]*\n(.*?)\n?```\s*$", t, re.S)
    if fence:
        t = fence.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise InvalidOutput("no JSON object in output")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        raise InvalidOutput("malformed JSON") from None
    if not isinstance(data, dict):
        raise InvalidOutput("output is not a JSON object")
    return data


def validate_turns(data: dict, card: dict) -> list[dict]:
    turns = data.get("turns")
    if not isinstance(turns, list):
        raise InvalidOutput("'turns' is not a list")
    lo, hi = card["min_turns"], card["max_turns"]
    if not lo <= len(turns) <= hi:
        raise InvalidOutput(f"turn count {len(turns)} outside {lo}-{hi}")
    clean = []
    for i, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise InvalidOutput(f"turn {i}: not an object")
        speaker = turn.get("speaker")
        speaker = speaker.strip().lower() if isinstance(speaker, str) else None
        if speaker not in SPEAKERS:
            raise InvalidOutput(f"turn {i}: bad speaker")
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidOutput(f"turn {i}: empty or non-string text")
        if RULE_ID.search(text):
            raise InvalidOutput(f"turn {i}: cites a rule ID")
        clean.append({"speaker": speaker, "text": text})
    if clean[0]["speaker"] != "agent":
        raise InvalidOutput("first turn is not the agent")
    for i in range(1, len(clean)):
        if clean[i]["speaker"] == clean[i - 1]["speaker"]:
            raise InvalidOutput(f"turns {i} and {i + 1} have the same speaker")
    return clean


# ---------------------------------------------------------------- prompt

def build_system_prompt() -> str:
    for path in (PROMPT_PATH, POLICY_PATH):
        if not path.is_file():
            raise SystemExit(f"error: required file not found: {path}")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    policy = POLICY_PATH.read_text(encoding="utf-8")
    return template.replace("{policy}", policy)


def build_messages(system_prompt: str, card: dict) -> list[dict]:
    user = {k: card[k] for k in ("channel", "account", "brief", "min_turns", "max_turns")}
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------- pacing

class Pacer:
    """Per-model pacing, call budget and stop flag. Sleep and clock are injected."""

    def __init__(self, sleep=time.sleep, clock=time.monotonic):
        self.sleep, self.clock = sleep, clock
        self.calls: Counter = Counter()
        self.last: dict[str, float] = {}
        self.stopped: dict[str, str] = {}
        self.transient_streak: Counter = Counter()

    def exhausted(self, key: str) -> bool:
        return self.calls[key] >= limits_for(key)["max_calls"]

    def available(self, key: str) -> bool:
        return key not in self.stopped and not self.exhausted(key)

    def before_call(self, key: str) -> None:
        """Wait out the model's interval, then count the call."""
        if key in self.last:
            wait = limits_for(key)["interval"] - (self.clock() - self.last[key])
            if wait > 0:
                self.sleep(wait)
        self.last[key] = self.clock()
        self.calls[key] += 1


# ---------------------------------------------------------------- run

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def run_cards(pending, client, system_prompt, out_path, log_path, pacer,
              now=_utc_now, say=print) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    skipped_by_model: Counter = Counter()
    total = len(pending)

    with open(out_path, "a", encoding="utf-8") as out_f, \
            open(log_path, "a", encoding="utf-8") as log_f:

        def log(card, attempt, status, error=None):
            row = {"ts": now(), "id": card["id"], "model": card["model"],
                   "attempt": attempt, "status": status, "error": error}
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()

        for n, card in enumerate(pending, start=1):
            key = norm_model(card["model"])
            status = "failed"
            last_transient = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if not pacer.available(key):
                    status = "skipped"
                    break
                if last_transient:
                    pacer.sleep(TRANSIENT_WAIT)
                last_transient = False
                pacer.before_call(key)
                try:
                    text = client.complete(card["model"], build_messages(system_prompt, card))
                except RateLimited as exc:
                    pacer.stopped[key] = "rate limited"
                    log(card, attempt, "error", str(exc))
                    break
                except TransientError as exc:
                    last_transient = True
                    log(card, attempt, "error", str(exc))
                    continue
                except LLMError as exc:
                    pacer.stopped[key] = "provider error"
                    log(card, attempt, "error", str(exc))
                    break
                pacer.transient_streak[key] = 0
                try:
                    turns = validate_turns(parse_output(text), card)
                except InvalidOutput as exc:
                    log(card, attempt, "invalid", str(exc))
                    continue
                record = {"id": card["id"], "channel": card["channel"],
                          "account": card["account"], "turns": turns}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                log(card, attempt, "ok")
                status = "ok"
                break
            if status == "failed" and last_transient:
                pacer.transient_streak[key] += 1
                if pacer.transient_streak[key] >= TRANSIENT_LIMIT:
                    pacer.stopped[key] = "unavailable"
            totals[status] += 1
            if status == "skipped":
                skipped_by_model[key] += 1
            say(f"[{n}/{total}] {card['id']} {status}")

    return {"ok": totals["ok"], "failed": totals["failed"], "skipped": totals["skipped"],
            "skipped_by_model": skipped_by_model, "calls": pacer.calls}


def print_summary(summary: dict, pacer: Pacer, say=print) -> None:
    say(f"ok: {summary['ok']}  failed: {summary['failed']}  skipped: {summary['skipped']}")
    for key, n in sorted(summary["skipped_by_model"].items()):
        reason = pacer.stopped.get(key) or "budget used up"
        say(f"skipped {n} for {key}: {reason}")
    for key, n in sorted(summary["calls"].items()):
        say(f"calls {key}: {n}/{limits_for(key)['max_calls']}")


# ---------------------------------------------------------------- other modes

def dry_run(cards, done, limit, say=print) -> None:
    for path in (PROMPT_PATH, POLICY_PATH):
        if not path.is_file():
            raise SystemExit(f"error: required file not found: {path}")
    pending = [c for c in cards if c["id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    say(f"cards: {len(cards)}  done: {sum(c['id'] in done for c in cards)}  pending: {len(pending)}")
    per_model = Counter(norm_model(c["model"]) for c in pending)
    longest = 0.0
    for key, n in sorted(per_model.items()):
        lim = limits_for(key)
        usable = min(n, lim["max_calls"])
        longest = max(longest, max(usable - 1, 0) * lim["interval"])
        say(f"pending {key}: {n} (budget {lim['max_calls']} calls, {lim['interval']} s apart)")
    say(f"minimum run time: {longest:.0f} s")


def list_models(client, say=print) -> None:
    ids = {norm_model(m) for m in client.list_models()}
    for mid in sorted(ids):
        if "flash" in mid.lower() or "gemma" in mid.lower():
            say(mid)


def ping(client, pacer, say=print) -> None:
    for key in [k for k in PACING if k != "*"]:
        if not pacer.available(key):
            say(f"{key}: skipped (budget)")
            continue
        pacer.before_call(key)
        try:
            client.complete(key, [{"role": "user", "content": "Reply with OK"}])
        except LLMError as exc:
            detail = " ".join(exc.detail.split())[:200]
            say(f"{key}: error {exc} {detail}".rstrip())
        else:
            say(f"{key}: ok")


# ---------------------------------------------------------------- CLI

def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m retention.generate")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data/pilot/scenarios.sealed")
    p.add_argument("--out", type=Path, default=ROOT / "data/pilot/conversations.jsonl")
    p.add_argument("--log", type=Path, default=ROOT / "data/pilot/generation_log.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--ping", action="store_true")
    return p.parse_args(argv)


def _make_client():
    try:
        base_url, api_key = config.load_settings()
    except config.ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    return ChatClient(base_url, api_key)


def main(argv=None, client=None, sleep=time.sleep, clock=time.monotonic) -> int:
    args = _parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be at least 1")
    pacer = Pacer(sleep, clock)

    if args.list_models or args.ping:
        client = client or _make_client()
        try:
            if args.list_models:
                list_models(client)
            else:
                ping(client, pacer)
        except LLMError as exc:
            raise SystemExit(f"error: {exc}") from None
        return 0

    try:
        cards = load_cards(args.scenarios)
    except CardError as exc:
        raise SystemExit(f"error: {exc}") from None
    done = read_done_ids(args.out)

    if args.dry_run:
        dry_run(cards, done, args.limit)
        return 0

    system_prompt = build_system_prompt()
    pending = [c for c in cards if c["id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"cards: {len(cards)}  done: {sum(c['id'] in done for c in cards)}"
          f"  to attempt: {len(pending)}")
    if not pending:
        return 0

    client = client or _make_client()
    summary = run_cards(pending, client, system_prompt, args.out, args.log, pacer)
    print_summary(summary, pacer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
