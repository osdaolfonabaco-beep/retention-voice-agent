"""Retention agent: the turn loop with tools, and a runner that plays it against a
simulated customer.

The agent never receives the account or the customer brief directly: it sees what
the tools return and what the customer says. The runner reads from a scenario only
the id, channel, agent name, account (to back the tools and fill the date) and the
customer brief (for the simulator); nothing else from a scenario reaches a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retention import config
from retention.audit import (RATE_LIMIT_RETRIES, RATE_LIMIT_WAIT, RUN_NAME,
                             TRANSIENT_RETRIES, TRANSIENT_WAIT, _sha256, _utc_now,
                             write_run_meta)
from retention.generate import read_done_ids
from retention.llm import ChatClient, LLMError, RateLimited, TransientError
from retention.tools import TOOL_SPECS, AccountSession

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS = Path("data") / "agent" / "scenarios.jsonl"
DEFAULT_MAX_CALLS = 2000
MAX_TOOL_ROUNDS = 6         # rounds of tool execution in one agent turn
MAX_TURNS = 40              # recorded turns (agent and customer together)
FAIL_STREAK_LIMIT = 3       # failed scenarios in a row before the run stops
AGENT_REASONING_EFFORT = "none"   # gpt-6-luna rejects function tools with reasoning on chat completions
START_MESSAGE = "[contact started]"
END_MARKER = "[END]"
SCENARIO_KEYS = ("id", "channel", "agent_name", "account", "customer_brief")
MISMATCH = ("run {run} was started with different models, prompts, policy or scenarios; "
            "use a new run name")


# ---------------------------------------------------------------- agent turn

def agent_turn(client, model: str, system_prompt: str, messages: list[dict],
               session: AccountSession,
               reasoning_effort: str | None = AGENT_REASONING_EFFORT,
               allow_tools: bool = True) -> tuple[str, list[dict], dict]:
    """One agent turn. `messages` is the conversation without the system prompt.

    Calls the model, runs any tool calls through `session.call`, feeds the results
    back, and repeats for up to MAX_TOOL_ROUNDS rounds. After the last round the
    model is called once more without tools so that it ends with text. Returns
    (agent text for this turn, the extended messages, summed usage). With
    `allow_tools=False` the model is called once, without tools, and no tool runs.
    """
    messages = list(messages)
    system = {"role": "system", "content": system_prompt}
    texts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        tools = TOOL_SPECS if allow_tools and round_no < MAX_TOOL_ROUNDS else None
        message, used = client.chat_with_tools(model, [system] + messages, tools,
                                               reasoning_effort=reasoning_effort)
        usage["input_tokens"] += used.get("input_tokens") or 0
        usage["output_tokens"] += used.get("output_tokens") or 0
        content = message["content"]
        if content.strip():
            texts.append(content.strip())
        calls = message["tool_calls"] if tools else []
        if not calls:
            messages.append({"role": "assistant", "content": content})
            break
        messages.append({
            "role": "assistant", "content": content or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}}
                           for c in calls]})
        for call in calls:
            result = session.call(call["name"], call["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    return " ".join(texts), messages, usage


# ---------------------------------------------------------------- model calls for a run

class _BudgetExhausted(Exception):
    pass


class _ScenarioFailed(Exception):
    """A model call failed for good, or a speaker produced nothing. `str` is safe to print."""


class _Stats:
    def __init__(self):
        self.ok = self.failed = self.skipped = 0
        self.calls = 0
        self.tokens = {role: {"input": 0, "output": 0} for role in ("agent", "customer")}
        self.stopped = False
        self.out_of_budget = False


class _RoleClient:
    """What `agent_turn` and the customer see as the client for one role."""

    def __init__(self, guard: "_Guard", role: str):
        self._guard = guard
        self._role = role

    def chat_with_tools(self, model, messages, tools=None, reasoning_effort=None):
        return self._guard.call(self._role, lambda c: c.chat_with_tools(
            model, messages, tools, reasoning_effort=reasoning_effort))

    def complete_with_usage(self, model, messages):
        return self._guard.call(self._role, lambda c: c.complete_with_usage(model, messages))


class _Guard:
    """Budget, retries, log and token counts around every model call of a run."""

    def __init__(self, client, max_calls: int, log_f, stats: _Stats, sleep, now):
        self.client, self.max_calls, self.log_f = client, max_calls, log_f
        self.stats, self.sleep, self.now = stats, sleep, now
        self.scenario_id = ""

    def role_client(self, role: str) -> _RoleClient:
        return _RoleClient(self, role)

    def _log(self, role, status, error, usage):
        row = {"ts": self.now(), "id": self.scenario_id, "role": role, "status": status,
               "error": error, "input_tokens": usage["input_tokens"],
               "output_tokens": usage["output_tokens"]}
        self.log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log_f.flush()

    def call(self, role: str, send):
        none = {"input_tokens": None, "output_tokens": None}
        rate_retries = transient_retries = 0
        wait = 0
        while True:
            if self.stats.calls >= self.max_calls:
                raise _BudgetExhausted
            if wait:
                self.sleep(wait)
                wait = 0
            self.stats.calls += 1
            try:
                result, usage = send(self.client)
            except RateLimited as exc:
                self._log(role, "error", str(exc), none)
                if rate_retries >= RATE_LIMIT_RETRIES:
                    raise _ScenarioFailed(f"{role} call failed: {exc}") from None
                rate_retries += 1
                wait = RATE_LIMIT_WAIT
                continue
            except TransientError as exc:
                self._log(role, "error", str(exc), none)
                if transient_retries >= TRANSIENT_RETRIES:
                    raise _ScenarioFailed(f"{role} call failed: {exc}") from None
                transient_retries += 1
                wait = TRANSIENT_WAIT
                continue
            except LLMError as exc:
                self._log(role, "error", str(exc), none)
                raise _ScenarioFailed(f"{role} call failed: {exc}") from None
            tokens = self.stats.tokens[role]
            tokens["input"] += usage["input_tokens"] or 0
            tokens["output"] += usage["output_tokens"] or 0
            self._log(role, "ok", None, usage)
            return result, usage


# ---------------------------------------------------------------- one scenario

def build_agent_prompt(template: str, policy: str, scenario: dict) -> str:
    text = (template.replace("{channel}", scenario["channel"])
            .replace("{today}", scenario["account"]["today"])
            .replace("{agent_name}", scenario["agent_name"]))
    return text.replace("{policy}", policy)


def build_customer_prompt(template: str, scenario: dict) -> str:
    return template.replace("{channel}", scenario["channel"]).replace(
        "{brief}", scenario["customer_brief"])


def play_scenario(scenario: dict, agent_client, customer_client, agent_model: str,
                  customer_model: str, agent_system: str, customer_system: str
                  ) -> tuple[list[dict], AccountSession, str]:
    """Play one contact. Returns (turns, session, stop_reason)."""
    session = AccountSession(scenario["account"], scenario["id"])
    agent_messages = [{"role": "user", "content": START_MESSAGE}]   # not recorded
    customer_messages = [{"role": "system", "content": customer_system}]
    turns: list[dict] = []
    opening = True      # no tool can be needed before the customer speaks
    while True:
        session.turn = len(turns)
        text, agent_messages, _ = agent_turn(
            agent_client, agent_model, agent_system, agent_messages, session,
            allow_tools=not opening)
        opening = False
        if text:
            turns.append({"speaker": "agent", "text": text})
        if session.ended:
            return turns, session, "end_contact"
        if not text:
            raise _ScenarioFailed("the agent said nothing and did not end the contact")
        if len(turns) >= MAX_TURNS:
            return turns, session, "max_turns"
        customer_messages.append({"role": "user", "content": text})
        reply, _ = customer_client.complete_with_usage(customer_model, customer_messages)
        reply = reply.strip()
        if reply == END_MARKER:
            return turns, session, "customer_end"
        if not reply:
            raise _ScenarioFailed("the customer said nothing")
        customer_messages.append({"role": "assistant", "content": reply})
        turns.append({"speaker": "customer", "text": reply})
        agent_messages.append({"role": "user", "content": reply})
        if len(turns) >= MAX_TURNS:
            return turns, session, "max_turns"


# ---------------------------------------------------------------- run

def _require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"error: required file not found: {path}")
    return path


def load_scenarios(path: Path) -> tuple[list[dict], str]:
    """Return (scenarios, sha256 of the file). Only the fields the runner needs are kept."""
    text = _require(path).read_text(encoding="utf-8")
    scenarios, seen = [], set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            ok = (all(k in raw for k in SCENARIO_KEYS) and isinstance(raw["account"], dict)
                  and isinstance(raw["account"].get("today"), str))
        except (json.JSONDecodeError, TypeError):
            ok = False
        if not ok:
            raise SystemExit(f"error: bad scenario line in {path}")
        if raw["id"] in seen:
            raise SystemExit(f"error: duplicate scenario id {raw['id']}")
        seen.add(raw["id"])
        scenarios.append({k: raw[k] for k in SCENARIO_KEYS})
    return scenarios, _sha256(text)


def check_run_meta(run_path: Path, current: dict) -> bool:
    """True if run.json exists and matches. Exit if it exists and differs."""
    if not run_path.is_file():
        return False
    try:
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        same = all(saved[k] == v for k, v in current.items() if k != "created")
    except (json.JSONDecodeError, KeyError, TypeError):
        same = False
    if not same:
        raise SystemExit(MISMATCH.format(run=current["run"]))
    return True


def run_scenarios(pending, client, agent_model, customer_model, agent_template,
                  customer_template, policy, out_dir: Path, max_calls: int,
                  sleep=time.sleep, now=_utc_now, say=print) -> _Stats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _Stats()
    streak = 0
    total = len(pending)
    with open(out_dir / "conversations.jsonl", "a", encoding="utf-8") as conv_f, \
            open(out_dir / "tool_calls.jsonl", "a", encoding="utf-8") as calls_f, \
            open(out_dir / "log.jsonl", "a", encoding="utf-8") as log_f:
        guard = _Guard(client, max_calls, log_f, stats, sleep, now)
        agent_client, customer_client = guard.role_client("agent"), guard.role_client("customer")
        for n, scenario in enumerate(pending, start=1):
            note = ""
            if stats.stopped or stats.out_of_budget:
                status = "skipped"
            else:
                guard.scenario_id = scenario["id"]
                try:
                    turns, session, stop_reason = play_scenario(
                        scenario, agent_client, customer_client, agent_model, customer_model,
                        build_agent_prompt(agent_template, policy, scenario),
                        build_customer_prompt(customer_template, scenario))
                except _BudgetExhausted:
                    status, stats.out_of_budget = "skipped", True
                except _ScenarioFailed as exc:
                    status, note = "failed", f": {exc}"
                else:
                    calls_f.write(json.dumps(
                        {"id": scenario["id"], "end_status": session.end_status,
                         "stop_reason": stop_reason, "calls": session.calls},
                        ensure_ascii=False) + "\n")
                    calls_f.flush()
                    conv_f.write(json.dumps(
                        {"id": scenario["id"], "channel": scenario["channel"],
                         "account": scenario["account"], "turns": turns},
                        ensure_ascii=False) + "\n")
                    conv_f.flush()
                    status = "ok"
                if status == "failed":
                    streak += 1
                    stats.stopped = streak >= FAIL_STREAK_LIMIT
                elif status == "ok":
                    streak = 0
            setattr(stats, status, getattr(stats, status) + 1)
            say(f"[{n}/{total}] {scenario['id']} {status}{note}")
    return stats


def print_summary(stats: _Stats, max_calls: int, say=print) -> None:
    say(f"ok: {stats.ok}  failed: {stats.failed}  skipped: {stats.skipped}")
    say(f"calls: {stats.calls}/{max_calls}")
    for role, t in stats.tokens.items():
        say(f"{role}: input tokens {t['input']}  output tokens {t['output']}")
    if stats.stopped:
        say(f"stopped: {FAIL_STREAK_LIMIT} scenarios failed in a row")
    elif stats.skipped:
        say("stopped: call budget used up")


def run_mode(args, client, models: tuple[str, str], root: Path, sleep, now, say=print) -> int:
    agent_model, customer_model = models
    agent_template = _require(root / "prompts" / "agent_system.md").read_text(encoding="utf-8")
    customer_template = _require(root / "prompts" / "customer_simulator.md").read_text(
        encoding="utf-8")
    policy = _require(root / "docs" / "policy.md").read_text(encoding="utf-8")
    scenarios, scenarios_sha = load_scenarios(
        Path(args.scenarios) if args.scenarios else root / DEFAULT_SCENARIOS)

    out_dir = root / "data" / "agent_runs" / args.run
    meta = {"run": args.run, "agent_model": agent_model, "customer_model": customer_model,
            "agent_prompt_sha256": _sha256(agent_template),
            "customer_prompt_sha256": _sha256(customer_template),
            "policy_sha256": _sha256(policy), "scenarios_sha256": scenarios_sha,
            "agent_reasoning_effort": AGENT_REASONING_EFFORT, "created": now()}
    meta_exists = check_run_meta(out_dir / "run.json", meta)

    done = read_done_ids(out_dir / "conversations.jsonl")
    pending = [s for s in scenarios if s["id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    n_done = sum(s["id"] in done for s in scenarios)

    if args.dry_run:
        say(f"scenarios: {len(scenarios)}  done: {n_done}  pending: {len(pending)}")
        return 0

    say(f"scenarios: {len(scenarios)}  done: {n_done}  to attempt: {len(pending)}")
    if not pending:
        return 0
    if not meta_exists:
        write_run_meta(out_dir / "run.json", meta)
    stats = run_scenarios(pending, client(), agent_model, customer_model, agent_template,
                          customer_template, policy, out_dir, args.max_calls, sleep, now, say)
    print_summary(stats, args.max_calls, say)
    return 1 if stats.stopped else 0


# ---------------------------------------------------------------- command line

def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m retention.agent")
    p.add_argument("--run", metavar="NAME", required=True,
                   help="output in data/agent_runs/NAME/")
    p.add_argument("--scenarios", metavar="PATH",
                   help=f"scenario file (default {DEFAULT_SCENARIOS})")
    p.add_argument("--limit", type=int, metavar="N", help="play at most N pending scenarios")
    p.add_argument("--max-calls", type=int, metavar="N", default=DEFAULT_MAX_CALLS,
                   help=f"model call budget (default {DEFAULT_MAX_CALLS})")
    p.add_argument("--dry-run", action="store_true", help="show counts, call nothing")
    args = p.parse_args(argv)
    if not RUN_NAME.fullmatch(args.run):
        p.error("--run NAME may only hold letters, digits, '.', '_' and '-'")
    for flag in ("limit", "max_calls"):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            p.error(f"--{flag.replace('_', '-')} must be at least 1")
    return args


def main(argv=None, client=None, env_path=None, root: Path = ROOT,
         sleep=time.sleep, now=_utc_now) -> int:
    args = _parse_args(argv)
    try:
        base_url, api_key, agent_model, customer_model = config.load_agent_settings(env_path)
    except config.ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    return run_mode(args, lambda: client or ChatClient(base_url, api_key),
                    (agent_model, customer_model), Path(root), sleep, now)


if __name__ == "__main__":
    sys.exit(main())
