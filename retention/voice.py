"""Voice layer for the retention agent: a cascade of speech-to-text, the same
`agent_turn`, and text-to-speech, with its timings measured.

`python -m retention.voice --run NAME` plays the scenarios by voice and writes the same
`conversations.jsonl` and `tool_calls.jsonl` as the text runner (so `retention.audit` and
`retention.trace` work on it unchanged), plus `voice_turns.jsonl`. `--report NAME`
summarizes `voice_turns.jsonl`. Audio is never saved. Nothing here reads labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

from retention import agent, config
from retention.agent import (END_MARKER, MAX_TURNS, START_MESSAGE, _BudgetExhausted, _Guard,
                             _require, _ScenarioFailed, _Stats, agent_turn)
from retention.audit import RUN_NAME, _sha256, _utc_now, write_run_meta
from retention.generate import read_done_ids
from retention.llm import ChatClient
from retention.tools import TOOLS_VERSION, AccountSession, _numbers
from retention.trace import _guard_of

ROOT = Path(__file__).resolve().parent.parent
STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
AGENT_VOICE = "coral"
CUSTOMER_VOICE = "ash"
DEFAULT_MAX_AUDIO_SECONDS = 3600
INTERRUPT_MIN_SECONDS = 8.0      # audio length at which the customer may cut in
INTERRUPT_MARKER = " [interrupted]"
INTERRUPT_LINE = "Sorry, hold on. Can you repeat that last part?"
ACCOUNT_CHANGES = {"apply_offer": "applied", "set_payment_plan": "set",
                   "cancel_service": "cancelled", "take_payment": "received"}
NO_USAGE = {"input_tokens": None, "output_tokens": None}


# ---------------------------------------------------------------- speech text

_LEAK = re.compile(r"to=functions\.\w+")
_CLOSERS = ".!?\"')]"


def speakable(text: str) -> str:
    """What is sent to text-to-speech: no leaked tool call, no markdown markers
    (`**`, `*`, `#`, backticks), whitespace collapsed.

    A leaked call is a tail starting at `to=functions.<name>`; the one token glued
    just before it (such as "Nprovidername") goes too, unless it ends a sentence.
    """
    leak = _LEAK.search(text)
    if leak:
        head = text[:leak.start()].rstrip()
        last = head.rsplit(None, 1)[-1] if head else ""
        if last and last[-1] not in _CLOSERS:
            head = head[:len(head) - len(last)]
        text = head
    return " ".join(re.sub(r"[*#`]", "", text).split())


def cut_spoken(spoken: str) -> tuple[str, str]:
    """(heard part, unspoken remainder): `spoken` cut at half of its characters, back to
    a word boundary."""
    limit = len(spoken) // 2
    head = spoken[:limit]
    inside_word = 0 < limit < len(spoken) and not spoken[limit].isspace() \
        and not spoken[limit - 1].isspace()
    if inside_word:
        head = head.rsplit(None, 1)[0] if len(head.split()) > 1 else ""
    head = head.rstrip()
    return head, spoken[len(head):].strip()


def _words(text: str) -> list[str]:
    text = re.sub(r"[’']", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).split()


def word_error_rate(intended: str, heard: str) -> float:
    """Word-level edit distance of `heard` against `intended`, over the intended words."""
    ref, hyp = _words(intended), _words(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, start=1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / len(ref)


# ---------------------------------------------------------------- audio through the run guard

class _AudioBudgetExhausted(_BudgetExhausted):
    pass


class _VoiceStats(_Stats):
    def __init__(self):
        super().__init__()
        self.tokens.update({role: {"input": 0, "output": 0} for role in ("stt", "tts")})
        self.audio = {"agent": 0.0, "customer": 0.0}     # seconds of synthesized audio
        self.audio_out = False


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


class _Audio:
    """Speech calls with the run's call budget, retries and log, plus the audio budget."""

    def __init__(self, guard: _Guard, stats: _VoiceStats, max_audio_seconds: float, clock):
        self.guard, self.stats, self.max_audio, self.clock = guard, stats, max_audio_seconds, clock

    def speak(self, who: str, voice: str, text: str) -> tuple[bytes, float, float, float]:
        """(wav, audio seconds, first byte ms, total ms)."""
        if sum(self.stats.audio.values()) >= self.max_audio:
            raise _AudioBudgetExhausted
        (wav, seconds, first, total), _ = self.guard.call(
            "tts", lambda c: (c.speak(TTS_MODEL, voice, text), NO_USAGE))
        self.stats.audio[who] += seconds
        return wav, seconds, _ms(first), _ms(total)

    def transcribe(self, wav: bytes) -> tuple[str, float]:
        """(text, ms)."""
        def send(c):
            start = self.clock()
            text = c.transcribe(STT_MODEL, wav)
            return (text, _ms(self.clock() - start)), NO_USAGE

        result, _ = self.guard.call("stt", send)
        return result


class _Counting:
    """Counts the model calls of an agent turn."""

    def __init__(self, inner):
        self.inner, self.count = inner, 0

    def chat_with_tools(self, model, messages, tools=None, reasoning_effort=None):
        self.count += 1
        return self.inner.chat_with_tools(model, messages, tools,
                                          reasoning_effort=reasoning_effort)


# ---------------------------------------------------------------- one scenario

def _got_account(session: AccountSession) -> bool:
    return any(c["name"] == "get_account" and _guard_of(c["result"]) is None
               for c in session.calls)


def _keep_only_heard(messages: list[dict], start: int, heard_text: str) -> list[dict]:
    """The agent's history with the turn that begins at `start` ending in `heard_text`:
    the last assistant message holds it, earlier text of the turn is cleared, and tool
    calls and results stay as they are."""
    messages = list(messages)
    last = max(i for i in range(start, len(messages)) if messages[i]["role"] == "assistant")
    for i in range(start, last):
        if messages[i]["role"] == "assistant":
            messages[i] = {**messages[i], "content": None}
    messages[last] = {**messages[last], "content": heard_text}
    return messages


def play_voice_scenario(scenario: dict, agent_client, customer_client, audio: _Audio,
                        agent_model: str, customer_model: str, agent_system: str,
                        customer_system: str, interrupt: bool = True, clock=time.perf_counter
                        ) -> tuple[list[dict], AccountSession, str, list[dict]]:
    """Play one contact by voice. Returns (turns, session, stop_reason, voice_turns rows)."""
    sid = scenario["id"]
    session = AccountSession(scenario["account"], sid, grounding=True, name_tolerance=True)
    agent_messages = [{"role": "user", "content": START_MESSAGE}]   # not recorded
    customer_messages = [{"role": "system", "content": customer_system}]
    counting = _Counting(agent_client)
    turns: list[dict] = []
    rows: list[dict] = []
    opening, interrupted_once = True, False
    last_stt_ms = 0.0               # speech-to-text time of the customer turn before
    while True:
        session.turn = len(turns)
        start_index = len(agent_messages)
        counting.count = 0
        began = clock()
        text, agent_messages, _ = agent_turn(
            counting, agent_model, agent_system, agent_messages, session,
            allow_tools=not opening, guards=True)
        model_ms = _ms(clock() - began)
        opening = False
        spoken = speakable(text)
        seconds = first_ms = total_ms = 0.0
        if spoken:
            _, seconds, first_ms, total_ms = audio.speak("agent", AGENT_VOICE, spoken)
        recorded, row = text, {
            "id": sid, "turn": len(turns), "speaker": "agent", "text": text, "spoken": spoken,
            "audio_seconds": round(seconds, 3), "model_ms": model_ms,
            "model_calls": counting.count, "tts_first_byte_ms": first_ms,
            "tts_total_ms": total_ms, "response_ms": round(last_stt_ms + model_ms + first_ms, 1)}
        cut_in = (interrupt and not interrupted_once and not session.ended and spoken
                  and seconds >= INTERRUPT_MIN_SECONDS and _got_account(session)
                  and len(turns) + 1 < MAX_TURNS)
        if cut_in:
            interrupted_once = True
            heard, rest = cut_spoken(spoken)
            recorded = heard + INTERRUPT_MARKER
            agent_messages = _keep_only_heard(agent_messages, start_index, recorded)
            row.update({"text": recorded, "spoken": heard, "interrupted": True,
                        "full_text": spoken, "unspoken": rest,
                        "cut_in_seconds": round(seconds / 2, 3)})
        if text:
            turns.append({"speaker": "agent", "text": recorded})
            rows.append(row)
        if session.ended:
            return turns, session, "end_contact", rows
        if not text:
            raise _ScenarioFailed("the agent said nothing and did not end the contact")
        if len(turns) >= MAX_TURNS:
            return turns, session, "max_turns", rows
        customer_messages.append({"role": "user", "content": recorded})
        if cut_in:
            intended = INTERRUPT_LINE
        else:
            intended, _ = customer_client.complete_with_usage(customer_model, customer_messages)
            intended = intended.strip()
            if intended == END_MARKER:
                return turns, session, "customer_end", rows
            if not intended:
                raise _ScenarioFailed("the customer said nothing")
        customer_messages.append({"role": "assistant", "content": intended})
        wav, c_seconds, _, _ = audio.speak("customer", CUSTOMER_VOICE, intended)
        heard_text, last_stt_ms = audio.transcribe(wav)
        heard_text = heard_text.strip()
        if not heard_text:
            raise _ScenarioFailed("the customer said nothing")
        rows.append({"id": sid, "turn": len(turns), "speaker": "customer", "text": heard_text,
                     "intended": intended, "audio_seconds": round(c_seconds, 3),
                     "stt_ms": last_stt_ms,
                     "word_error_rate": round(word_error_rate(intended, heard_text), 4)})
        turns.append({"speaker": "customer", "text": heard_text})
        session.hear(heard_text)
        agent_messages.append({"role": "user", "content": heard_text})
        if len(turns) >= MAX_TURNS:
            return turns, session, "max_turns", rows


# ---------------------------------------------------------------- run

def _dump(f, row: dict) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def run_voice_scenarios(pending, client, agent_model, customer_model, agent_template,
                        customer_template, policy, out_dir: Path, max_calls: int,
                        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
                        interrupt: bool = True, sleep=time.sleep, now=_utc_now, say=print,
                        clock=time.perf_counter) -> _VoiceStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _VoiceStats()
    streak, total = 0, len(pending)
    with open(out_dir / "conversations.jsonl", "a", encoding="utf-8") as conv_f, \
            open(out_dir / "tool_calls.jsonl", "a", encoding="utf-8") as calls_f, \
            open(out_dir / "voice_turns.jsonl", "a", encoding="utf-8") as voice_f, \
            open(out_dir / "log.jsonl", "a", encoding="utf-8") as log_f:
        guard = _Guard(client, max_calls, log_f, stats, sleep, now)
        audio = _Audio(guard, stats, max_audio_seconds, clock)
        agent_client, customer_client = guard.role_client("agent"), guard.role_client("customer")
        for n, scenario in enumerate(pending, start=1):
            note = ""
            if stats.stopped or stats.out_of_budget or stats.audio_out:
                status = "skipped"
            else:
                guard.scenario_id = scenario["id"]
                try:
                    turns, session, stop_reason, rows = play_voice_scenario(
                        scenario, agent_client, customer_client, audio, agent_model,
                        customer_model, agent.build_agent_prompt(agent_template, policy, scenario),
                        agent.build_customer_prompt(customer_template, scenario), interrupt, clock)
                except _AudioBudgetExhausted:
                    status, stats.audio_out = "skipped", True
                except _BudgetExhausted:
                    status, stats.out_of_budget = "skipped", True
                except _ScenarioFailed as exc:
                    status, note = "failed", f": {exc}"
                else:
                    for row in rows:        # conversations.jsonl last: it marks the scenario done
                        _dump(voice_f, row)
                    _dump(calls_f, {"id": scenario["id"], "end_status": session.end_status,
                                    "stop_reason": stop_reason, "calls": session.calls})
                    _dump(conv_f, {"id": scenario["id"], "channel": scenario["channel"],
                                   "account": scenario["account"], "turns": turns})
                    status = "ok"
                if status == "failed":
                    streak += 1
                    stats.stopped = streak >= agent.FAIL_STREAK_LIMIT
                elif status == "ok":
                    streak = 0
            setattr(stats, status, getattr(stats, status) + 1)
            say(f"[{n}/{total}] {scenario['id']} {status}{note}")
    return stats


def print_voice_summary(stats: _VoiceStats, max_calls: int, max_audio_seconds: float,
                        say=print) -> None:
    say(f"ok: {stats.ok}  failed: {stats.failed}  skipped: {stats.skipped}")
    say(f"calls: {stats.calls}/{max_calls}")
    say(f"audio seconds: agent {stats.audio['agent']:.1f}  customer {stats.audio['customer']:.1f}"
        f"  (limit {max_audio_seconds:g})")
    for role in ("agent", "customer"):
        t = stats.tokens[role]
        say(f"{role}: input tokens {t['input']}  output tokens {t['output']}")
    if stats.stopped:
        say(f"stopped: {agent.FAIL_STREAK_LIMIT} scenarios failed in a row")
    elif stats.audio_out:
        say("stopped: audio budget used up")
    elif stats.skipped:
        say("stopped: call budget used up")


def run_mode(args, client, models: tuple[str, str], root: Path, sleep, now, say=print,
             clock=time.perf_counter) -> int:
    agent_model, customer_model = models
    agent_template = _require(root / args.agent_prompt).read_text(encoding="utf-8")
    customer_template = _require(root / "prompts" / "customer_simulator.md").read_text(
        encoding="utf-8")
    policy = _require(root / "docs" / "policy.md").read_text(encoding="utf-8")
    scenarios, scenarios_sha = agent.load_scenarios(
        Path(args.scenarios) if args.scenarios else root / agent.DEFAULT_SCENARIOS)

    out_dir = root / "data" / "agent_runs" / args.run
    meta = {"run": args.run, "agent_model": agent_model, "customer_model": customer_model,
            "agent_prompt": args.agent_prompt, "guards": True, "name_tolerance": True,
            "agent_prompt_sha256": _sha256(agent_template),
            "customer_prompt_sha256": _sha256(customer_template),
            "policy_sha256": _sha256(policy), "scenarios_sha256": scenarios_sha,
            "agent_reasoning_effort": agent.AGENT_REASONING_EFFORT,
            "tools_version": TOOLS_VERSION, "stt_model": STT_MODEL, "tts_model": TTS_MODEL,
            "agent_voice": AGENT_VOICE, "customer_voice": CUSTOMER_VOICE,
            "interruptions": not args.no_interrupt, "created": now()}
    meta_exists = agent.check_run_meta(out_dir / "run.json", meta)

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
    stats = run_voice_scenarios(
        pending, client(), agent_model, customer_model, agent_template, customer_template,
        policy, out_dir, args.max_calls, args.max_audio_seconds, not args.no_interrupt,
        sleep, now, say, clock)
    print_voice_summary(stats, args.max_calls, args.max_audio_seconds, say)
    return 1 if stats.stopped else 0


# ---------------------------------------------------------------- report

def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"error: required file not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ok = isinstance(row, dict) and isinstance(row["id"], str)
        except (json.JSONDecodeError, KeyError):
            ok = False
        if not ok:
            raise SystemExit(f"error: bad line in {path}")
        rows.append(row)
    return rows


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    rank = max(math.ceil(0.95 * len(ordered)), 1)           # nearest rank
    return {"n": len(ordered), "median": round(statistics.median(ordered), 1),
            "p95": round(ordered[rank - 1], 1), "max": round(ordered[-1], 1)}


def _has_number(value: float, numbers: list[float]) -> bool:
    return any(abs(value - n) < 0.005 for n in numbers)


def interruption_stats(voice_rows: list[dict], tool_rows: list[dict]) -> dict:
    """What happened after each simulated interruption.

    The next agent turn is two turns after the interrupted one. `no_account_change` counts
    interruptions whose next agent turn made no successful apply_offer, set_payment_plan,
    cancel_service or take_payment. `numbers_repeated` counts interruptions where every
    number written with digits in the unspoken remainder appears again in a later agent
    turn of the contact; interruptions whose remainder holds no such number are left out.
    """
    calls = {r["id"]: r["calls"] for r in tool_rows}
    agent_rows = [r for r in voice_rows if r["speaker"] == "agent"]
    cuts = [r for r in agent_rows if r.get("interrupted")]
    with_next = no_change = with_numbers = repeated = 0
    for cut in cuts:
        same = [r for r in agent_rows if r["id"] == cut["id"]]
        nxt = cut["turn"] + 2
        if any(r["turn"] == nxt for r in same):
            with_next += 1
            changed = any(
                c["after_turn"] == nxt and ACCOUNT_CHANGES.get(c["name"]) is not None
                and isinstance(c["result"], dict)
                and c["result"].get(ACCOUNT_CHANGES[c["name"]]) is True
                for c in calls.get(cut["id"], []))
            no_change += not changed
        numbers = _numbers(cut.get("unspoken", ""))
        if numbers:
            with_numbers += 1
            later = _numbers(" ".join(r["text"] for r in same if r["turn"] > cut["turn"]))
            repeated += all(_has_number(n, later) for n in numbers)
    return {"simulated": len(cuts), "with_next_agent_turn": with_next,
            "next_turn_no_account_change": no_change,
            "with_numbers_in_remainder": with_numbers, "numbers_repeated_later": repeated}


def more_proof_stats(tool_rows: list[dict]) -> dict:
    """Contacts where verify_identity returned more_proof_needed, and how many of those
    ended verified (a verified result in the contact; verification does not wear off)."""
    asked = ended_verified = 0
    for row in tool_rows:
        results = [c["result"].get("result") for c in row["calls"]
                   if c["name"] == "verify_identity" and isinstance(c["result"], dict)]
        if "more_proof_needed" in results:
            asked += 1
            ended_verified += any(r in ("verified_account_holder", "verified_authorized_user")
                                  for r in results)
    return {"asked": asked, "ended_verified": ended_verified}


def summarize_voice(run: str, voice_rows: list[dict], tool_rows: list[dict]) -> dict:
    customer = [r for r in voice_rows if r["speaker"] == "customer"]
    agent_rows = [r for r in voice_rows if r["speaker"] == "agent"]
    rates = [r["word_error_rate"] for r in customer]
    return {
        "run": run, "turns": len(voice_rows),
        "customer_turns": len(customer), "agent_turns": len(agent_rows),
        "timings_ms": {
            "stt_ms": _percentiles([r["stt_ms"] for r in customer]),
            **{k: _percentiles([r[k] for r in agent_rows])
               for k in ("model_ms", "tts_first_byte_ms", "response_ms")}},
        "audio_seconds": {"agent": round(sum(r["audio_seconds"] for r in agent_rows), 1),
                          "customer": round(sum(r["audio_seconds"] for r in customer), 1)},
        "word_error": {"mean_rate": round(sum(rates) / len(rates), 4) if rates else None,
                       "no_error_share": round(sum(x == 0 for x in rates) / len(rates), 4)
                       if rates else None},
        "interruptions": interruption_stats(voice_rows, tool_rows),
        "more_proof": more_proof_stats(tool_rows)}


def print_voice_report(s: dict, say=print) -> None:
    say(f"run: {s['run']}  turns: {s['turns']} "
        f"(customer {s['customer_turns']}, agent {s['agent_turns']})")
    say("ms            median      p95      max")
    for key, p in s["timings_ms"].items():
        cells = [("-" if p[k] is None else f"{p[k]:.1f}") for k in ("median", "p95", "max")]
        say(f"{key:<16}{cells[0]:>8}{cells[1]:>9}{cells[2]:>9}")
    say(f"audio seconds: agent {s['audio_seconds']['agent']}  customer {s['audio_seconds']['customer']}")
    w = s["word_error"]
    if w["mean_rate"] is not None:
        say(f"word error rate: mean {w['mean_rate']:.4f}  "
            f"customer turns with no error: {100 * w['no_error_share']:.1f}%")
    i = s["interruptions"]
    say(f"interruptions simulated: {i['simulated']}")
    say(f"  next agent turn made no successful account change: "
        f"{i['next_turn_no_account_change']} of {i['with_next_agent_turn']}")
    say(f"  every digit number of the unspoken part said again later: "
        f"{i['numbers_repeated_later']} of {i['with_numbers_in_remainder']}")
    m = s["more_proof"]
    say(f"asked for more proof: {m['asked']} contacts, {m['ended_verified']} ended verified")


def report_mode(run: str, root: Path, say=print) -> int:
    run_dir = Path(root) / "data" / "agent_runs" / run
    summary = summarize_voice(run, _read_rows(run_dir / "voice_turns.jsonl"),
                              _read_rows(run_dir / "tool_calls.jsonl"))
    print_voice_report(summary, say)
    (run_dir / "voice_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
    return 0


# ---------------------------------------------------------------- command line

def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m retention.voice")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", metavar="NAME", help="play by voice; output in data/agent_runs/NAME/")
    mode.add_argument("--report", metavar="NAME",
                      help="summarize data/agent_runs/NAME/voice_turns.jsonl")
    p.add_argument("--scenarios", metavar="PATH",
                   help=f"scenario file (default {agent.DEFAULT_SCENARIOS})")
    p.add_argument("--limit", type=int, metavar="N", help="play at most N pending scenarios")
    p.add_argument("--max-calls", type=int, metavar="N", default=agent.DEFAULT_MAX_CALLS,
                   help=f"call budget, audio calls included (default {agent.DEFAULT_MAX_CALLS})")
    p.add_argument("--max-audio-seconds", type=float, metavar="N",
                   default=DEFAULT_MAX_AUDIO_SECONDS,
                   help=f"seconds of synthesized audio, both voices (default "
                        f"{DEFAULT_MAX_AUDIO_SECONDS})")
    p.add_argument("--agent-prompt", metavar="PATH", default=agent.DEFAULT_AGENT_PROMPT,
                   help=f"agent prompt file (default {agent.DEFAULT_AGENT_PROMPT})")
    p.add_argument("--no-interrupt", action="store_true",
                   help="do not simulate the customer cutting in")
    p.add_argument("--dry-run", action="store_true", help="show counts, call nothing")
    args = p.parse_args(argv)
    for flag in ("run", "report"):
        name = getattr(args, flag)
        if name is not None and not RUN_NAME.fullmatch(name):
            p.error(f"--{flag} NAME may only hold letters, digits, '.', '_' and '-'")
    for flag in ("limit", "max_calls", "max_audio_seconds"):
        if getattr(args, flag) is not None and not getattr(args, flag) >= 1:
            p.error(f"--{flag.replace('_', '-')} must be at least 1")
    return args


def main(argv=None, client=None, env_path=None, root: Path = ROOT, sleep=time.sleep,
         now=_utc_now, say=print, clock=time.perf_counter) -> int:
    args = _parse_args(argv)
    if args.report:
        return report_mode(args.report, Path(root), say)
    try:
        base_url, api_key, agent_model, customer_model = config.load_agent_settings(env_path)
    except config.ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    return run_mode(args, lambda: client or ChatClient(base_url, api_key),
                    (agent_model, customer_model), Path(root), sleep, now, say, clock)


if __name__ == "__main__":
    sys.exit(main())
