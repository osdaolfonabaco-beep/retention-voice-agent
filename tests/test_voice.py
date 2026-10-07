"""Offline tests for retention/voice.py, with a fake audio client."""

import json

import pytest

from retention import trace, voice
from retention.audit import TRANSIENT_WAIT
from retention.llm import LLMError, TransientError
from retention.tools import AccountSession
from test_agent import (ADDRESS, NOW, FakeClient, make_account, make_root, make_scenario, say,
                        use, write_env)

AG01_TAIL = ('Thanks for confirming. Thank you for calling Tolvane, and goodbye. '
             'Nprovidername to=functions.end_contact #+#+#+#+commentary 〔json〕{"status":"resolved"}')
VERIFY = {"full_name": "Teresa Galloway", "service_address": ADDRESS, "pin": "4410"}
CALLER = "I'm Teresa Galloway at 412 Juniper Ct, pin 4410."
LONG = ("Your plan is Plus at $60 a month, and your next bill of $60 is due on October 24th. "
        "You can also get twenty five percent off for three months if you stay with us.")


# ---------------------------------------------------------------- fakes

class Clock:
    """Every reading is 10 ms later than the one before."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.01
        return self.t


class VoiceClient(FakeClient):
    """FakeClient plus the two audio calls. A WAV is b"WAV|" + the text; a word takes
    `per_word` seconds of audio; `heard` maps intended text to its transcript."""

    def __init__(self, agent_script=(), customer_script=(), heard=None, per_word=0.5):
        super().__init__(agent_script, customer_script)
        self.heard, self.per_word = heard or {}, per_word
        self.spoken, self.transcribed, self.speak_errors = [], [], []

    def speak(self, model, voice_name, text):
        if self.speak_errors:
            raise self.speak_errors.pop(0)
        self.spoken.append((model, voice_name, text))
        return b"WAV|" + text.encode(), len(text.split()) * self.per_word, 0.2, 0.9

    def transcribe(self, model, wav, language="en"):
        text = wav[4:].decode()
        self.transcribed.append((model, text, language))
        return self.heard.get(text, text)


ACCOUNT = make_account(on_file_for_verification={"last_payment_amount": 60.0, "pin": "4410"})


class VRun:
    def __init__(self, tmp_path, ids=("ag-01",)):
        self.tmp_path, self.root = tmp_path, make_root(tmp_path, ids)
        (self.root / "data" / "agent" / "scenarios.jsonl").write_text(
            "".join(json.dumps(make_scenario(i, account=ACCOUNT)) + "\n" for i in ids),
            encoding="utf-8")
        self.sleeps, self.out = [], []

    @property
    def out_dir(self):
        return self.root / "data" / "agent_runs" / "r1"

    def go(self, client, *argv):
        env = write_env(self.tmp_path)
        return voice.main(["--run", "r1", *argv], client=client, env_path=env, root=self.root,
                          sleep=self.sleeps.append, now=lambda: NOW, say=self.out.append,
                          clock=Clock())

    def rows(self, name):
        path = self.out_dir / name
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def simple_client(**kw):
    """Greeting; the customer identifies herself; the agent verifies, reads the account and
    answers; the customer ends."""
    return VoiceClient(
        [say("Hello, this is Marisa."), use("verify_identity", VERIFY, call_id="c1"),
         use("get_account", {}, call_id="c2"), say("Thanks, Teresa. You are on Plus.")],
        [CALLER, "[END]"], **kw)


# ---------------------------------------------------------------- speakable

def test_speakable_cuts_the_leaked_tool_call_of_ag01():
    assert voice.speakable(AG01_TAIL) == (
        "Thanks for confirming. Thank you for calling Tolvane, and goodbye.")


@pytest.mark.parametrize("text, expected", [
    ("Your plan is **Plus** now.", "Your plan is Plus now."),
    ("# Summary\n* one\n* two", "Summary one two"),
    ("Use `code` here", "Use code here"),
    ("  many \n\n spaces\t here ", "many spaces here"),
    ("Plain sentence.", "Plain sentence."),
    ("", ""),
    ("Goodbye. to=functions.end_contact {\"status\":\"resolved\"}", "Goodbye."),
    ("Goodbye! junk to=functions.end_contact {}", "Goodbye!"),
    ("Goodbye?\" to=functions.end_contact {}", "Goodbye?\""),
    ("to=functions.end_contact {}", ""),
    ("All **done** Nxyz to=functions.escalate {\"a\": 1} and more", "All done"),
])
def test_speakable(text, expected):
    assert voice.speakable(text) == expected


def test_cut_spoken_goes_back_to_a_word_boundary():
    heard, rest = voice.cut_spoken("alpha beta gamma delta")        # half = 11, inside "gamma"
    assert (heard, rest) == ("alpha beta", "gamma delta")
    assert voice.cut_spoken("abcd efgh") == ("abcd", "efgh")        # half = 4: a space follows
    heard, rest = voice.cut_spoken(LONG)
    assert heard + " " + rest == LONG and len(heard) <= len(LONG) // 2
    assert voice.cut_spoken("supercalifragilistic") == ("", "supercalifragilistic")


@pytest.mark.parametrize("intended, heard, rate", [
    ("I want to pay sixty dollars", "I want to pay sixty dollars", 0.0),
    ("I want to pay sixty dollars", "i want to pay Sixty dollars!", 0.0),
    ("Don't go", "dont go", 0.0),
    ("one two three four", "one three four", 0.25),                 # a deletion
    ("one two three four", "one 2 three four", 0.25),               # a substitution
    ("one two", "one two three", 0.5),                              # an insertion
    ("one two", "", 1.0),
])
def test_word_error_rate(intended, heard, rate):
    assert voice.word_error_rate(intended, heard) == pytest.approx(rate)


# ---------------------------------------------------------------- turn flow and timings

def test_the_agent_receives_the_transcript_and_the_record_holds_it(tmp_path):
    run = VRun(tmp_path)
    heard = "I'm Teresa Galloway at 412 Juniper Ct, pin four four one zero."
    client = simple_client(heard={CALLER: heard})
    assert run.go(client) == 0
    [conv] = run.rows("conversations.jsonl")
    assert [t["speaker"] for t in conv["turns"]] == ["agent", "customer", "agent"]
    assert conv["turns"][1]["text"] == heard
    assert client.agent_calls[1]["messages"][-1] == {"role": "user", "content": heard}
    [calls] = run.rows("tool_calls.jsonl")      # the PIN said in words grounded session.hear
    assert calls["calls"][0]["result"]["result"] == "verified_account_holder"
    assert calls["calls"][0]["after_turn"] == 2
    # the simulator saw the agent's text and keeps its own words, not the transcript
    assert client.customer_calls[1]["messages"][-1] == {
        "role": "user", "content": "Thanks, Teresa. You are on Plus."}
    assert {"role": "assistant", "content": CALLER} in client.customer_calls[1]["messages"]


def test_voice_turns_rows_and_timing_fields(tmp_path):
    run = VRun(tmp_path)
    heard = "I'm Teresa Galloway at 412 Juniper Ct, pin four four one zero."
    run.go(simple_client(heard={CALLER: heard}))
    greeting, customer, answer = run.rows("voice_turns.jsonl")
    assert greeting == {
        "id": "ag-01", "turn": 0, "speaker": "agent", "text": "Hello, this is Marisa.",
        "spoken": "Hello, this is Marisa.", "audio_seconds": 2.0, "model_ms": 10.0,
        "model_calls": 1, "tts_first_byte_ms": 200.0, "tts_total_ms": 900.0,
        "response_ms": 210.0}                                   # no speech-to-text part
    assert customer["speaker"] == "customer" and customer["turn"] == 1
    assert customer["text"] == heard and customer["intended"] == CALLER
    assert customer["audio_seconds"] == 4.5 and customer["stt_ms"] == 10.0
    assert customer["word_error_rate"] == round(voice.word_error_rate(CALLER, heard), 4) > 0
    assert answer["turn"] == 2 and answer["model_calls"] == 3
    assert answer["model_ms"] == 10.0
    assert answer["response_ms"] == 10.0 + 10.0 + 200.0         # stt of the turn before + model + first byte
    assert "interrupted" not in answer


def test_end_marker_is_never_synthesized_and_the_run_stops(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    run.go(client)
    assert [text for _, _, text in client.spoken] == [
        "Hello, this is Marisa.", CALLER, "Thanks, Teresa. You are on Plus."]
    assert all("[END]" not in text for _, _, text in client.spoken)
    [calls] = run.rows("tool_calls.jsonl")
    assert calls["stop_reason"] == "customer_end"


def test_models_and_voices(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    run.go(client)
    assert [(m, v) for m, v, _ in client.spoken] == [
        (voice.TTS_MODEL, "coral"), (voice.TTS_MODEL, "ash"), (voice.TTS_MODEL, "coral")]
    assert client.transcribed == [(voice.STT_MODEL, CALLER, "en")]
    assert (voice.STT_MODEL, voice.TTS_MODEL) == ("gpt-4o-mini-transcribe", "gpt-4o-mini-tts")
    assert (voice.AGENT_VOICE, voice.CUSTOMER_VOICE) == ("coral", "ash")


def test_the_record_keeps_the_original_text_and_the_voice_turn_what_was_spoken(tmp_path):
    run = VRun(tmp_path)
    client = VoiceClient([say("**Hello**, I am Marisa. Nxx to=functions.end_contact {}")],
                         ["[END]"])
    run.go(client)
    [conv] = run.rows("conversations.jsonl")
    assert conv["turns"] == [{"speaker": "agent",
                              "text": "**Hello**, I am Marisa. Nxx to=functions.end_contact {}"}]
    [row] = run.rows("voice_turns.jsonl")
    assert row["spoken"] == "Hello, I am Marisa."
    assert client.spoken == [(voice.TTS_MODEL, "coral", "Hello, I am Marisa.")]


def test_a_turn_with_nothing_to_speak_skips_text_to_speech(tmp_path):
    run = VRun(tmp_path)
    client = VoiceClient([say("to=functions.end_contact {}")], ["[END]"])
    run.go(client)
    [row] = run.rows("voice_turns.jsonl")
    assert row["spoken"] == "" and row["audio_seconds"] == 0 and row["tts_first_byte_ms"] == 0
    assert client.spoken == []


def test_voice_run_files_work_with_trace_and_audit_loaders(tmp_path):
    run = VRun(tmp_path)
    run.go(simple_client())
    out = []
    assert trace.main(["--run", "r1"], root=run.root, say=out.append) == 0
    assert out[0] == "ag-01  none"
    for name in ("conversations.jsonl", "tool_calls.jsonl", "voice_turns.jsonl", "run.json",
                 "log.jsonl"):
        assert (run.out_dir / name).is_file()


def test_run_json_records_the_voice_settings(tmp_path):
    run = VRun(tmp_path)
    run.go(simple_client())
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["stt_model"] == "gpt-4o-mini-transcribe" and meta["tts_model"] == "gpt-4o-mini-tts"
    assert meta["agent_voice"] == "coral" and meta["customer_voice"] == "ash"
    assert meta["interruptions"] is True and meta["guards"] is True
    assert meta["tools_version"] == 3 and meta["agent_prompt"] == "prompts/agent_system.md"
    assert meta["name_tolerance"] is True
    assert meta["agent_reasoning_effort"] == "none" and meta["created"] == NOW
    assert {"agent_model", "customer_model", "agent_prompt_sha256", "customer_prompt_sha256",
            "policy_sha256", "scenarios_sha256"} <= set(meta)


def test_no_interrupt_is_recorded_and_part_of_the_resume_check(tmp_path):
    run = VRun(tmp_path, ids=("ag-01", "ag-02"))
    run.go(simple_client(), "--no-interrupt", "--limit", "1")
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["interruptions"] is False
    with pytest.raises(SystemExit):
        run.go(simple_client())                  # interruptions on: a different run
    assert run.go(simple_client(), "--no-interrupt") == 0     # resumes the second contact
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-01", "ag-02"]


def test_a_text_run_cannot_be_resumed_as_a_voice_run(tmp_path):
    from retention import agent
    run = VRun(tmp_path)
    env = write_env(tmp_path)
    agent.main(["--run", "r1"], client=FakeClient([say("Hi.")], ["[END]"]), env_path=env,
               root=run.root, sleep=lambda s: None, now=lambda: NOW)
    client = simple_client()
    with pytest.raises(SystemExit):
        run.go(client)
    assert client.agent_calls == []


def test_dry_run_calls_nothing_and_writes_nothing(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    assert run.go(client, "--dry-run", "--limit", "1") == 0
    assert run.out == ["scenarios: 1  done: 0  pending: 1"]
    assert client.agent_calls == [] and client.spoken == []
    assert not run.out_dir.exists()


def test_report_and_run_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        voice.main(["--run", "a", "--report", "b"])
    with pytest.raises(SystemExit):
        voice.main([])


# ---------------------------------------------------------------- interruption

def interrupt_client(first="", **kw):
    """The agent reads the account in a long answer; the customer cuts in; the agent
    repeats; the customer ends."""
    return VoiceClient(
        [say("Hello, this is Marisa."), use("verify_identity", VERIFY, call_id="c1"),
         use("get_account", {}, text=first, call_id="c2"), say(LONG),
         say("Your next bill of $60 is due on October 24th."), say("Anything else?")],
        [CALLER, "[END]"], **kw)


def test_interruption_cuts_the_recorded_turn_at_half_and_fixes_the_customer_line(tmp_path):
    run = VRun(tmp_path)
    client = interrupt_client()
    run.go(client)
    heard, rest = voice.cut_spoken(LONG)
    [conv] = run.rows("conversations.jsonl")
    assert [t["speaker"] for t in conv["turns"]] == ["agent", "customer", "agent", "customer", "agent"]
    assert conv["turns"][2] == {"speaker": "agent", "text": heard + " [interrupted]"}
    assert conv["turns"][3] == {"speaker": "customer", "text": voice.INTERRUPT_LINE}
    # the fixed line went through text-to-speech (customer voice) and speech-to-text
    assert ("gpt-4o-mini-tts", "ash", voice.INTERRUPT_LINE) in client.spoken
    assert (voice.STT_MODEL, voice.INTERRUPT_LINE, "en") in client.transcribed
    # the simulator was called only for the first customer turn and the [END] one
    assert len(client.customer_calls) == 2 and client.customer_script == []


def test_interruption_history_of_the_agent_and_of_the_simulator(tmp_path):
    run = VRun(tmp_path)
    client = interrupt_client()
    run.go(client)
    heard, _ = voice.cut_spoken(LONG)
    cut = heard + " [interrupted]"
    after = client.agent_calls[4]["messages"]       # the agent turn after the interruption
    assert after[-2:] == [{"role": "assistant", "content": cut},
                          {"role": "user", "content": voice.INTERRUPT_LINE}]
    names = [m["tool_calls"][0]["function"]["name"] for m in after if m.get("tool_calls")]
    assert names == ["verify_identity", "get_account"]          # tool calls stay
    assert [m["role"] for m in after if m["role"] == "tool"] == ["tool", "tool"]
    sim = client.customer_calls[1]["messages"]                   # [END] turn: after the repeat
    assert sim[-3:-1] == [{"role": "user", "content": cut},
                          {"role": "assistant", "content": voice.INTERRUPT_LINE}]


def test_interruption_row_in_voice_turns(tmp_path):
    run = VRun(tmp_path)
    run.go(interrupt_client())
    heard, rest = voice.cut_spoken(LONG)
    rows = run.rows("voice_turns.jsonl")
    row = next(r for r in rows if r.get("interrupted"))
    assert row["turn"] == 2 and row["speaker"] == "agent"
    assert row["text"] == heard + " [interrupted]" and row["spoken"] == heard
    assert row["full_text"] == LONG and row["unspoken"] == rest
    assert row["audio_seconds"] == len(LONG.split()) * 0.5
    assert row["cut_in_seconds"] == row["audio_seconds"] / 2
    fixed = rows[rows.index(row) + 1]
    assert fixed["intended"] == voice.INTERRUPT_LINE and fixed["word_error_rate"] == 0.0
    assert sum(bool(r.get("interrupted")) for r in rows) == 1


def test_interruption_with_text_in_an_earlier_round_keeps_the_calls_and_clears_that_text(tmp_path):
    run = VRun(tmp_path)
    client = interrupt_client(first="One moment please.")
    run.go(client)
    full = voice.speakable("One moment please. " + LONG)
    heard, _ = voice.cut_spoken(full)
    after = client.agent_calls[4]["messages"]
    assert after[-2] == {"role": "assistant", "content": heard + " [interrupted]"}
    get_account = next(m for m in after if m.get("tool_calls")
                       and m["tool_calls"][0]["function"]["name"] == "get_account")
    assert get_account["content"] is None


def test_at_most_one_interruption_per_contact(tmp_path):
    run = VRun(tmp_path)
    client = VoiceClient(
        [say("Hello, this is Marisa."), use("verify_identity", VERIFY, call_id="c1"),
         use("get_account", {}, call_id="c2"), say(LONG), say(LONG), say("Anything else?")],
        [CALLER, "[END]"])
    run.go(client)
    rows = run.rows("voice_turns.jsonl")
    assert sum(bool(r.get("interrupted")) for r in rows) == 1
    assert len(client.customer_calls) == 2         # the second long turn used the simulator


@pytest.mark.parametrize("script, customer, why", [
    ([say(LONG), say("Bye.")], [CALLER, "[END]"], "no get_account yet"),
    ([say("Hello."), use("verify_identity", VERIFY, call_id="c1"),
      use("get_account", {}, call_id="c2"), say("Short answer here."), say("Bye.")],
     [CALLER, "Thanks.", "[END]"], "audio under 8 seconds"),
])
def test_no_interruption_without_the_trigger(tmp_path, script, customer, why):
    run = VRun(tmp_path)
    client = VoiceClient(script, customer)
    run.go(client)
    assert not any(r.get("interrupted") for r in run.rows("voice_turns.jsonl")), why


def test_a_blocked_get_account_does_not_count(tmp_path):
    run = VRun(tmp_path)
    client = VoiceClient(
        [say("Hello."), use("get_account", {}, text="Can you confirm your name?", call_id="c1"),
         say(LONG), say("Bye.")], [CALLER, "Thanks.", "[END]"])
    run.go(client)                       # the ask-then-wait guard blocked the call
    assert not any(r.get("interrupted") for r in run.rows("voice_turns.jsonl"))


def test_the_turn_that_ends_the_contact_is_not_interrupted(tmp_path):
    run = VRun(tmp_path)
    client = VoiceClient(
        [say("Hello."), use("verify_identity", VERIFY, call_id="c1"),
         use("get_account", {}, call_id="c2"),
         use("end_contact", {"status": "resolved"}, text=LONG, call_id="c3"), say("Goodbye.")],
        [CALLER])
    run.go(client)
    [conv] = run.rows("conversations.jsonl")
    assert conv["turns"][-1] == {"speaker": "agent", "text": LONG + " Goodbye."}
    assert not any(r.get("interrupted") for r in run.rows("voice_turns.jsonl"))


def test_no_interrupt_option_turns_it_off(tmp_path):
    run = VRun(tmp_path)
    client = interrupt_client()
    client.customer_script = [CALLER, "Thanks.", "[END]"]
    client.agent_script = client.agent_script[:5] + [say("Anything else?")]
    run.go(client, "--no-interrupt")
    assert not any(r.get("interrupted") for r in run.rows("voice_turns.jsonl"))
    [conv] = run.rows("conversations.jsonl")
    assert conv["turns"][2] == {"speaker": "agent", "text": LONG}


# ---------------------------------------------------------------- budgets, retries, log

def test_audio_budget_skips_the_rest_and_keeps_only_finished_contacts(tmp_path):
    run = VRun(tmp_path, ids=("ag-01", "ag-02", "ag-03"))
    # each contact: 2 s greeting + 4.5 s customer + 3 s answer = 9.5 s of audio; the second
    # contact's greeting still goes out (the limit is checked before each call), then it stops
    client = VoiceClient(
        [say("Hello, this is Marisa."), use("verify_identity", VERIFY, call_id="c1"),
         use("get_account", {}, call_id="c2"), say("Thanks, Teresa. You are on Plus.")] * 3,
        [CALLER, "[END]"] * 3)
    assert run.go(client, "--no-interrupt", "--max-audio-seconds", "10") == 0
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-01"]
    assert {r["id"] for r in run.rows("voice_turns.jsonl")} == {"ag-01"}
    assert {r["id"] for r in run.rows("tool_calls.jsonl")} == {"ag-01"}
    assert "[1/3] ag-01 ok" in run.out and "[2/3] ag-02 skipped" in run.out
    assert "[3/3] ag-03 skipped" in run.out
    assert "ok: 1  failed: 0  skipped: 2" in run.out and "stopped: audio budget used up" in run.out
    assert any(line.startswith("audio seconds: agent 7.0  customer 4.5") for line in run.out)


def test_audio_budget_stops_in_the_middle_of_a_contact(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    run.go(client, "--max-audio-seconds", "3")
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")
    assert not (run.out_dir / "voice_turns.jsonl").read_text(encoding="utf-8")
    assert "stopped: audio budget used up" in run.out


def test_audio_seconds_count_both_voices(tmp_path):
    run = VRun(tmp_path)
    run.go(simple_client(), "--max-audio-seconds", "5")      # 2 s agent + 4 s customer
    assert "[1/1] ag-01 skipped" in run.out


def test_audio_calls_count_against_the_call_budget(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    run.go(client, "--max-calls", "3")       # agent, agent text-to-speech, simulator, then no more
    assert "[1/1] ag-01 skipped" in run.out and "stopped: call budget used up" in run.out
    assert "calls: 3/2000" not in run.out and "calls: 3/3" in run.out
    assert (run.out_dir / "log.jsonl").read_text(encoding="utf-8").count('"role": "tts"') == 1


def test_audio_calls_are_logged_with_their_role_and_retried_like_model_calls(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    client.speak_errors = [TransientError("Timeout")]
    run.go(client)
    assert run.sleeps == [TRANSIENT_WAIT]
    log = run.rows("log.jsonl")
    roles = [(r["role"], r["status"]) for r in log]
    assert roles[:3] == [("agent", "ok"), ("tts", "error"), ("tts", "ok")]
    assert {"stt", "tts", "agent", "customer"} <= {r["role"] for r in log}
    stt = next(r for r in log if r["role"] == "stt")
    assert stt["status"] == "ok" and stt["id"] == "ag-01" and stt["input_tokens"] is None
    assert all("text" not in r for r in log)


def test_a_speech_error_that_is_not_transient_fails_the_scenario(tmp_path):
    run = VRun(tmp_path)
    client = simple_client()
    client.speak_errors = [LLMError("BadRequestError", 400)]
    run.go(client)
    assert "[1/1] ag-01 failed: tts call failed: BadRequestError 400" in run.out


def test_an_empty_transcript_fails_the_scenario(tmp_path):
    run = VRun(tmp_path)
    client = simple_client(heard={CALLER: "   "})
    run.go(client)
    assert "[1/1] ag-01 failed: the customer said nothing" in run.out
    assert not (run.out_dir / "voice_turns.jsonl").read_text(encoding="utf-8")


# ---------------------------------------------------------------- report

def test_percentiles_use_nearest_rank():
    assert voice._percentiles(list(range(1, 21))) == {"n": 20, "median": 10.5, "p95": 19,
                                                      "max": 20}
    assert voice._percentiles([5.0]) == {"n": 1, "median": 5.0, "p95": 5.0, "max": 5.0}
    assert voice._percentiles([]) == {"n": 0, "median": None, "p95": None, "max": None}


def agent_row(cid, turn, n, text="Hello.", **extra):
    row = {"id": cid, "turn": turn, "speaker": "agent", "text": text, "spoken": text,
           "audio_seconds": 2.0, "model_ms": 10.0 * n, "model_calls": 1,
           "tts_first_byte_ms": 100.0 + n, "tts_total_ms": 500.0, "response_ms": 300.0 + n}
    row.update(extra)
    return row


def customer_row(cid, turn, stt, wer):
    return {"id": cid, "turn": turn, "speaker": "customer", "text": "x", "intended": "x",
            "audio_seconds": 1.0, "stt_ms": stt, "word_error_rate": wer}


def cut_row(cid, turn, n, unspoken, **extra):
    return agent_row(cid, turn, n, text="Part [interrupted]", interrupted=True,
                     full_text="Part " + unspoken, unspoken=unspoken, **extra)


def report_fixture():
    rows = [
        # A: the remainder's numbers come back, the next turn changes the account
        agent_row("A", 0, 1), customer_row("A", 1, 100, 0.0), cut_row("A", 2, 2, "Your bill of $60 is due in 24 days."),
        customer_row("A", 3, 200, 0.1), agent_row("A", 4, 3, "Your bill is $60, due in 24 days."),
        # B: a number is not repeated, no account change
        agent_row("B", 0, 4), customer_row("B", 1, 300, 0.0), cut_row("B", 2, 5, "That is $80 a month."),
        customer_row("B", 3, 400, 0.0), agent_row("B", 4, 6, "Okay."),
        # C: nothing numeric; the next turn only has refused changes
        agent_row("C", 0, 7), customer_row("C", 1, 500, 0.25), cut_row("C", 2, 8, "Nothing numeric here."),
        customer_row("C", 3, 600, 0.0), agent_row("C", 4, 9, "Let me try."),
        # D: interrupted, then the contact stops: no next agent turn
        agent_row("D", 0, 1), customer_row("D", 1, 700, 0.0), cut_row("D", 2, 1, "It costs $5."),
        customer_row("D", 3, 800, 0.0),
    ]
    calls = [
        {"id": "A", "calls": [{"after_turn": 4, "name": "apply_offer", "arguments": {},
                               "result": {"applied": True, "details": {}}}]},
        {"id": "B", "calls": [{"after_turn": 2, "name": "apply_offer", "arguments": {},
                               "result": {"applied": True, "details": {}}}]},   # not the next turn
        {"id": "C", "calls": [{"after_turn": 4, "name": "take_payment", "arguments": {},
                               "result": {"received": False, "reason": "x"}},
                              {"after_turn": 4, "name": "apply_offer", "arguments": {},
                               "result": {"blocked": "asked_a_question"}},
                              {"after_turn": 4, "name": "get_account", "arguments": {},
                               "result": {"plan": "Plus"}}]},
        {"id": "D", "calls": []},
    ]
    return rows, calls


def test_summary_numbers():
    rows, calls = report_fixture()
    s = voice.summarize_voice("r1", rows, calls)
    assert (s["turns"], s["customer_turns"], s["agent_turns"]) == (19, 8, 11)
    assert s["timings_ms"]["stt_ms"] == {"n": 8, "median": 450.0, "p95": 800.0, "max": 800.0}
    assert s["timings_ms"]["model_ms"] == {"n": 11, "median": 40.0, "p95": 90.0, "max": 90.0}
    assert s["timings_ms"]["tts_first_byte_ms"]["n"] == 11
    assert s["timings_ms"]["response_ms"]["max"] == 309.0
    assert s["audio_seconds"] == {"agent": 22.0, "customer": 8.0}
    assert s["word_error"] == {"mean_rate": round(0.35 / 8, 4), "no_error_share": 0.75}
    assert s["interruptions"] == {
        "simulated": 4, "with_next_agent_turn": 3, "next_turn_no_account_change": 2,
        "with_numbers_in_remainder": 3, "numbers_repeated_later": 1}


def test_a_number_must_come_back_in_a_later_agent_turn_of_the_same_contact():
    rows = [agent_row("A", 0, 1), customer_row("A", 1, 1, 0), cut_row("A", 2, 1, "Pay $60 or $75."),
            customer_row("A", 3, 1, 0), agent_row("A", 4, 1, "That is $60 only."),
            agent_row("B", 0, 1), customer_row("B", 1, 1, 0), cut_row("B", 2, 1, "Pay $60."),
            customer_row("B", 3, 1, 0), agent_row("B", 4, 1, "Bye.")]
    s = voice.interruption_stats(rows, [{"id": "A", "calls": []}, {"id": "B", "calls": []}])
    assert s["with_numbers_in_remainder"] == 2 and s["numbers_repeated_later"] == 0
    rows[4]["text"] = "That is $60.00 or $75."
    assert voice.interruption_stats(
        rows, [{"id": "A", "calls": []}, {"id": "B", "calls": []}])["numbers_repeated_later"] == 1


def test_report_command_prints_and_writes_voice_summary(tmp_path):
    run = VRun(tmp_path)
    run.go(interrupt_client())
    out = []
    assert voice.main(["--report", "r1"], root=run.root, say=out.append) == 0
    summary = json.loads((run.out_dir / "voice_summary.json").read_text(encoding="utf-8"))
    assert summary["run"] == "r1" and summary["interruptions"]["simulated"] == 1
    assert summary["interruptions"]["with_next_agent_turn"] == 1
    assert summary["audio_seconds"]["agent"] > 0 and summary["audio_seconds"]["customer"] > 0
    text = "\n".join(out)
    for label in ("stt_ms", "model_ms", "tts_first_byte_ms", "response_ms", "median", "p95", "max",
                  "audio seconds", "word error rate", "interruptions simulated: 1"):
        assert label in text


def test_report_needs_the_files(tmp_path):
    with pytest.raises(SystemExit):
        voice.main(["--report", "nope"], root=tmp_path)


# ---------------------------------------------------------------- close names

HEARD_CALLER = "I'm Teresa Galoway at 412 Juniper Ct, pin 4410."
CLOSE_VERIFY = {**VERIFY, "full_name": "Teresa Galoway"}


def close_name_client():
    """The speech-to-text hears "Galoway"; the tool asks for the ZIP; the customer gives it."""
    return VoiceClient(
        [say("Hello, this is Marisa."), use("verify_identity", CLOSE_VERIFY, call_id="c1"),
         say("Thanks. And what is the ZIP code on the account?"),
         use("verify_identity", {**CLOSE_VERIFY, "zip_code": "86001"}, call_id="c2"),
         say("Thanks, Teresa.")],
        [CALLER, "86001", "[END]"], heard={CALLER: HEARD_CALLER})


def test_the_voice_session_has_name_tolerance_and_run_json_records_it(tmp_path):
    run = VRun(tmp_path)
    run.go(close_name_client(), "--no-interrupt")
    [row] = run.rows("tool_calls.jsonl")
    assert [c["result"] for c in row["calls"]] == [
        {"result": "more_proof_needed", "ask_for": "zip_code", "failed_attempts": 0},
        {"result": "verified_account_holder", "failed_attempts": 0, "name_match": "close"}]
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["name_tolerance"] is True


def test_a_run_json_without_name_tolerance_cannot_be_resumed(tmp_path):
    run = VRun(tmp_path, ids=("ag-01", "ag-02"))
    run.go(simple_client(), "--limit", "1")
    path = run.out_dir / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["name_tolerance"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        run.go(simple_client())


def test_the_trace_check_reads_the_close_name_run_as_clean(tmp_path):
    run = VRun(tmp_path)
    run.go(close_name_client(), "--no-interrupt")
    out = []
    assert trace.main(["--run", "r1"], root=run.root, say=out.append) == 0
    assert out[0] == "ag-01  none"


def proof_rows(*per_contact):
    """tool_calls rows: one list of verify_identity results per contact."""
    return [{"id": f"c{i}", "calls": [
        {"after_turn": 2, "name": "verify_identity", "arguments": {}, "result": r}
        for r in results]} for i, results in enumerate(per_contact)]


ASKED = {"result": "more_proof_needed", "ask_for": "zip_code", "failed_attempts": 0}
HOLDER = {"result": "verified_account_holder", "failed_attempts": 0, "name_match": "close"}
NOT_VERIFIED = {"result": "not_verified", "failed_attempts": 1}


def test_more_proof_counts_contacts_asked_and_those_that_ended_verified():
    rows = proof_rows([ASKED, HOLDER],                                  # asked, verified
                      [ASKED, ASKED, NOT_VERIFIED, NOT_VERIFIED],       # asked, never verified
                      [ASKED],                                          # asked, left
                      [{"result": "verified_account_holder", "failed_attempts": 0}],   # never asked
                      [NOT_VERIFIED], [])
    assert voice.more_proof_stats(rows) == {"asked": 3, "ended_verified": 1}
    assert voice.more_proof_stats([]) == {"asked": 0, "ended_verified": 0}


def test_more_proof_ignores_discarded_calls_and_other_tools():
    rows = [{"id": "c0", "calls": [
        {"after_turn": 2, "name": "verify_identity", "arguments": {},
         "result": {"blocked": "asked_a_question"}},
        {"after_turn": 2, "name": "get_account", "arguments": {}, "result": ASKED}]}]
    assert voice.more_proof_stats(rows) == {"asked": 0, "ended_verified": 0}


def test_the_summary_and_the_report_line_carry_the_more_proof_counts(tmp_path):
    rows, calls = report_fixture()
    assert voice.summarize_voice("r1", rows, calls)["more_proof"] == {"asked": 0, "ended_verified": 0}
    run = VRun(tmp_path)
    run.go(close_name_client(), "--no-interrupt")
    out = []
    assert voice.main(["--report", "r1"], root=run.root, say=out.append) == 0
    assert "asked for more proof: 1 contacts, 1 ended verified" in out
    summary = json.loads((run.out_dir / "voice_summary.json").read_text(encoding="utf-8"))
    assert summary["more_proof"] == {"asked": 1, "ended_verified": 1}
