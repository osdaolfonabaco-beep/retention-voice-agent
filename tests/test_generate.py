import json

import pytest

from retention import generate
from retention.generate import InvalidOutput, norm_model, parse_output, validate_turns
from retention.llm import LLMError, RateLimited, TransientError
from retention.sealed import seal_cards

BRIEF = "SECRET_BRIEF_TEXT"
FLASH = "gemini-3.8-flash"
LITE = "gemini-3.5-flash-lite"


def make_card(cid, model=LITE, lo=2, hi=4):
    return {"id": cid, "model": model, "channel": "chat", "account": {"plan": "x"},
            "brief": BRIEF, "min_turns": lo, "max_turns": hi}


def good(n=2):
    roles = ["Agent", "customer"]
    turns = [{"speaker": roles[i % 2], "text": f"line {i}"} for i in range(n)]
    return json.dumps({"turns": turns})


class FakeClient:
    """handler(model, n) returns text or an exception; n counts calls per model."""

    def __init__(self, handler=lambda model, n: good(), models=()):
        self.handler, self.models = handler, list(models)
        self.calls: list[str] = []

    def complete(self, model, messages):
        self.calls.append(model)
        result = self.handler(model, self.calls.count(model))
        if isinstance(result, Exception):
            raise result
        return result

    def list_models(self):
        return self.models


class FakeTime:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def clock(self):
        return self.now


@pytest.fixture
def env(tmp_path, monkeypatch):
    prompt, policy = tmp_path / "prompt.md", tmp_path / "policy.md"
    prompt.write_text("SYS {policy} END")
    policy.write_text("POLICY TEXT")
    monkeypatch.setattr(generate, "PROMPT_PATH", prompt)
    monkeypatch.setattr(generate, "POLICY_PATH", policy)

    class Env:
        scenarios = tmp_path / "s.sealed"
        out = tmp_path / "out.jsonl"
        log = tmp_path / "log.jsonl"

        def run(self, cards, client=None, extra=()):
            seal_cards(cards, self.scenarios)
            self.client = client or FakeClient()
            self.time = FakeTime()
            argv = ["--scenarios", str(self.scenarios), "--out", str(self.out),
                    "--log", str(self.log), *extra]
            return generate.main(argv, client=self.client,
                                 sleep=self.time.sleep, clock=self.time.clock)

        def out_rows(self):
            return [json.loads(l) for l in self.out.read_text().splitlines()]

        def log_rows(self):
            return [json.loads(l) for l in self.log.read_text().splitlines()]

    return Env()


# ---- parsing and validation

def test_fenced_json_and_surrounding_prose_accepted():
    body = good()
    assert parse_output(f"```json\n{body}\n```")["turns"]
    assert parse_output(f"Here you go:\n{body}\nDone.")["turns"]


def test_speaker_is_lowercased():
    card = make_card("a")
    turns = validate_turns(parse_output(good()), card)
    assert [t["speaker"] for t in turns] == ["agent", "customer"]


@pytest.mark.parametrize("data", [
    {"turns": "x"},
    {"turns": [{"speaker": "agent", "text": "a"}]},  # below min 2
    {"turns": [{"speaker": "agent", "text": "a"}] * 5},  # above max 4
    {"turns": [{"speaker": "bot", "text": "a"}, {"speaker": "agent", "text": "b"}]},
    {"turns": [{"speaker": "agent", "text": ""}, {"speaker": "agent", "text": "b"}]},
    {"turns": [{"speaker": "agent", "text": 5}, {"speaker": "agent", "text": "b"}]},
    {"turns": [{"speaker": "agent", "text": "per VR-3 we..."}, {"speaker": "agent", "text": "b"}]},
    {"turns": [{"speaker": "agent", "text": "a"}, {"speaker": "customer", "text": "see tn-1 and TN-12"}]},
])
def test_validation_rejects(data):
    with pytest.raises(InvalidOutput):
        validate_turns(data, make_card("a"))


def test_rule_id_regex_needs_word_boundary():
    data = {"turns": [{"speaker": "agent", "text": "PR-1x is fine? no: SPR-1"},
                      {"speaker": "customer", "text": "ok"}]}
    assert validate_turns(data, make_card("a"))


def test_not_json_rejected():
    with pytest.raises(InvalidOutput):
        parse_output("no braces here")
    with pytest.raises(InvalidOutput):
        parse_output("{not json}")


# ---- run behavior

def test_output_lines_have_exactly_four_keys(env):
    env.run([make_card("a"), make_card("b")])
    rows = env.out_rows()
    assert [set(r) for r in rows] == [{"id", "channel", "account", "turns"}] * 2


def test_resume_skips_existing_ids(env):
    env.out.write_text(json.dumps({"id": "a"}) + "\n")
    env.run([make_card("a"), make_card("b")])
    assert len(env.client.calls) == 1
    assert [r["id"] for r in env.out_rows()] == ["a", "b"]


def test_one_retry_then_failure(env, capsys):
    env.run([make_card("a")], FakeClient(lambda m, n: "garbage"))
    assert len(env.client.calls) == 2
    assert not env.out.read_text()
    assert [r["status"] for r in env.log_rows()] == ["invalid", "invalid"]
    assert "failed: 1" in capsys.readouterr().out


def test_retry_succeeds_after_invalid_or_transient(env):
    seq = {1: "garbage", 2: good()}
    env.run([make_card("a")], FakeClient(lambda m, n: seq[n]))
    assert [r["status"] for r in env.log_rows()] == ["invalid", "ok"]
    seq2 = {1: TransientError("APITimeoutError"), 2: good()}
    env.run([make_card("b")], FakeClient(lambda m, n: seq2[n]))
    assert [r["status"] for r in env.log_rows()][-2:] == ["error", "ok"]


def test_rate_limit_stops_that_model_only(env, capsys):
    def handler(model, n):
        return RateLimited("RateLimitError", 429, "slow down") if model == FLASH else good()

    env.run([make_card("a", FLASH), make_card("b", FLASH), make_card("c", LITE)],
            FakeClient(handler))
    assert env.client.calls == [FLASH, LITE]
    assert [r["id"] for r in env.out_rows()] == ["c"]
    out = capsys.readouterr().out
    assert "[2/3] b skipped" in out and "rate limited" in out


def test_other_llm_error_does_not_retry_and_stops_model(env):
    env.run([make_card("a"), make_card("b")],
            FakeClient(lambda m, n: LLMError("BadRequestError", 400, "x")))
    assert len(env.client.calls) == 1


def test_budget_cap(env, monkeypatch, capsys):
    monkeypatch.setitem(generate.PACING, "tiny", {"interval": 0, "max_calls": 2})
    env.run([make_card(f"c{i}", "tiny") for i in range(4)])
    assert len(env.client.calls) == 2
    out = capsys.readouterr().out
    assert "skipped: 2" in out and "budget used up" in out and "calls tiny: 2/2" in out


def test_retries_count_against_budget(env, monkeypatch):
    monkeypatch.setitem(generate.PACING, "tiny", {"interval": 0, "max_calls": 2})
    env.run([make_card("a", "tiny"), make_card("b", "tiny")],
            FakeClient(lambda m, n: "garbage"))
    assert len(env.client.calls) == 2


def test_pacing_sleeps_per_model(env):
    env.run([make_card("a", FLASH), make_card("b", LITE), make_card("c", FLASH)])
    assert env.time.sleeps == [13]


def test_models_prefix_shares_pacing_entry_and_is_sent_as_written(env):
    assert norm_model("models/x") == "x" == norm_model("x")
    env.run([make_card("a", "models/" + LITE), make_card("b", LITE)])
    assert env.client.calls == ["models/" + LITE, LITE]
    assert env.time.sleeps == [5]


def test_system_prompt_replaces_policy_token(env):
    seen = []

    def handler(model, n):
        return good()

    client = FakeClient(handler)
    orig = client.complete
    client.complete = lambda model, messages: (seen.append(messages), orig(model, messages))[1]
    env.run([make_card("a")], client)
    system, user = seen[0]
    assert system["content"] == "SYS POLICY TEXT END"
    assert set(json.loads(user["content"])) == {
        "channel", "account", "brief", "min_turns", "max_turns"}


def test_log_and_console_never_contain_brief(env, capsys):
    def handler(model, n):
        return {1: "garbage", 2: good()}[n] if model == LITE else LLMError("E", 400, BRIEF)

    env.run([make_card("a"), make_card("b", FLASH)], FakeClient(handler))
    captured = capsys.readouterr()
    assert BRIEF not in captured.out + captured.err + env.log.read_text()
    assert set(env.log_rows()[0]) == {"ts", "id", "model", "attempt", "status", "error"}
    for line in captured.out.splitlines():
        if line.startswith("["):
            assert line.split()[1] in {"a", "b"} and LITE not in line and FLASH not in line


# ---- other modes

def test_dry_run_makes_no_calls(env, capsys):
    env.out.write_text(json.dumps({"id": "a"}) + "\n")
    env.run([make_card("a"), make_card("b"), make_card("c", FLASH)], extra=["--dry-run"])
    assert env.client.calls == []
    out = capsys.readouterr().out
    assert "done: 1  pending: 2" in out and "pending " + LITE + ": 1" in out
    assert not env.log.exists()


def test_limit(env):
    env.run([make_card("a"), make_card("b"), make_card("c")], extra=["--limit", "2"])
    assert [r["id"] for r in env.out_rows()] == ["a", "b"]


def test_list_models_filters_and_strips_prefix(env, capsys):
    client = FakeClient(models=["models/gemini-3.8-flash", "gpt-x", "gemma-9", "gemini-pro"])
    env.run([], client, extra=["--list-models"])
    assert capsys.readouterr().out.split() == ["gemini-3.8-flash", "gemma-9"]


def test_ping_reports_ok_and_truncated_error(env, capsys):
    def handler(model, n):
        return LLMError("NotFoundError", 404, "bad " + "x" * 500) if model == FLASH else "OK"

    env.run([], FakeClient(handler), extra=["--ping"])
    lines = capsys.readouterr().out.splitlines()
    assert f"{LITE}: ok" in lines
    err = next(l for l in lines if l.startswith(FLASH))
    assert "NotFoundError 404 bad" in err and len(err.split("404 ", 1)[1]) == 200
