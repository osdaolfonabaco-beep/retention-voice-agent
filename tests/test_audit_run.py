"""Offline tests for the audit run mode: fake client, temporary project root."""

import hashlib
import json
from pathlib import Path

import pytest

from retention import audit
from retention.generate import InvalidOutput
from retention.llm import LLMError, RateLimited, TransientError

TEMPLATE = 'SYSTEM {"shape": "json"}\n<policy>\n{policy}\n</policy>\n'
POLICY = "Rules VR-1 and CL-7 apply.\n"
NOW = "2026-01-01T00:00:00+00:00"
CATS = audit.CATEGORIES

# id -> dataset; dataset order is batch2 then batch3
CONVS = {"b2-001": "batch2", "b2-002": "batch2", "b3-001": "batch3", "b3-002": "batch3"}
SPLIT = {"tune": ["b2-001", "b3-001", "b3-002"], "check": ["b2-002"]}


def write_env(tmp_path, model="model-a"):
    path = tmp_path / ".env"
    path.write_text(f"AUDITOR_BASE_URL=https://h/v1\nAUDITOR_API_KEY=sk-x\nAUDITOR_MODEL={model}\n",
                    encoding="utf-8")
    return path


def conv(cid):
    return {"id": cid, "channel": "chat",
            "account": {"today": "2026-10-05", "plan": "Giga"},
            "turns": [{"speaker": "agent", "text": f"hello from {cid}"},
                      {"speaker": "customer", "text": "I want to cancel"}]}


def make_root(tmp_path, ids=None, split=None):
    root = tmp_path / "proj"
    (root / "prompts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "prompts" / "audit_conversation.md").write_text(TEMPLATE, encoding="utf-8")
    (root / "docs" / "policy.md").write_text(POLICY, encoding="utf-8")
    ids = list(CONVS) if ids is None else ids
    for dataset in audit.DATASETS:
        d = root / "data" / dataset
        d.mkdir(parents=True)
        lines = [json.dumps(conv(i)) for i in ids if CONVS.get(i, "batch3") == dataset]
        (d / "conversations.jsonl").write_text("".join(line + "\n" for line in lines),
                                               encoding="utf-8")
    (root / "data" / "dev_split.json").write_text(json.dumps(split or SPLIT), encoding="utf-8")
    return root


def reply(fail=None, **overrides):
    """A valid auditor reply; `fail` maps category -> finding overrides."""
    verdicts = {c: "pass" for c in CATS}
    findings = []
    for cat, extra in (fail or {}).items():
        verdicts[cat] = "fail"
        findings.append({"category": cat, "rules": ["CL-7"], "turns": [2],
                         "evidence": "I want to cancel", "why": "because", **extra})
    return json.dumps({"verdicts": verdicts, "findings": findings, **overrides})


class FakeClient:
    """Plays a script, one item per call: str, (str, usage), or an exception."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete_with_usage(self, model, messages):
        self.calls.append((model, messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return item
        return item, {"input_tokens": 100, "output_tokens": 10}


class Run:
    def __init__(self, tmp_path, root=None, model="model-a"):
        self.tmp_path = tmp_path
        self.root = root or make_root(tmp_path)
        self.model = model
        self.sleeps = []

    @property
    def out_dir(self):
        return self.root / "data" / "audits" / "r1"

    def go(self, client, *argv, split="tune"):
        return audit.main(["--run", "r1", "--split", split, *argv], client=client,
                          env_path=write_env(self.tmp_path, self.model), root=self.root,
                          sleep=self.sleeps.append, now=lambda: NOW)

    def rows(self, name):
        path = self.out_dir / name
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------- validation

def good(**kw):
    return json.loads(reply(**kw))


def test_validate_normalizes_case_and_defaults():
    data = good(fail={"privacy_breach": {}})
    data["verdicts"]["privacy_breach"] = " FAIL "
    data["verdicts"]["false_resolution"] = "Pass"
    del data["findings"][0]["evidence"]
    verdicts, findings, warnings = audit.validate_audit(data, 2, {"CL-7"})
    assert verdicts["privacy_breach"] == "fail"
    assert verdicts["false_resolution"] == "pass"
    assert findings == [{"category": "privacy_breach", "rules": ["CL-7"], "turns": [2],
                         "evidence": "", "why": "because"}]
    assert warnings == []


def test_validate_clean_conversation_has_no_findings():
    verdicts, findings, warnings = audit.validate_audit(good(), 2, set())
    assert set(verdicts.values()) == {"pass"} and findings == [] and warnings == []


def bad_cases():
    def mut(fn, fail=True):
        data = good(fail={"privacy_breach": {}} if fail else None)
        fn(data)
        return data

    return {
        "no verdicts": mut(lambda d: d.pop("verdicts")),
        "missing category": mut(lambda d: d["verdicts"].pop("privacy_breach")),
        "extra category": mut(lambda d: d["verdicts"].update(extra="pass")),
        "bad value": mut(lambda d: d["verdicts"].update(privacy_breach="maybe")),
        "non-string value": mut(lambda d: d["verdicts"].update(privacy_breach=True)),
        "no findings key": mut(lambda d: d.pop("findings")),
        "findings not list": mut(lambda d: d.update(findings={})),
        "fail without finding": mut(lambda d: d.update(findings=[])),
        "finding for pass category": mut(lambda d: d["verdicts"].update(privacy_breach="pass")),
        "finding not object": mut(lambda d: d["findings"].append("x")),
        "unknown finding category": mut(lambda d: d["findings"][0].update(category="other")),
        "category not string": mut(lambda d: d["findings"][0].update(category=3)),
        "rules not list": mut(lambda d: d["findings"][0].update(rules="CL-7")),
        "rule not string": mut(lambda d: d["findings"][0].update(rules=[7])),
        "turns not list": mut(lambda d: d["findings"][0].update(turns=2)),
        "turn is float": mut(lambda d: d["findings"][0].update(turns=[1.5])),
        "turn is bool": mut(lambda d: d["findings"][0].update(turns=[True])),
        "evidence not string": mut(lambda d: d["findings"][0].update(evidence=["x"])),
        "why not string": mut(lambda d: d["findings"][0].update(why=1)),
    }


@pytest.mark.parametrize("name", list(bad_cases()))
def test_validate_rejects(name):
    with pytest.raises(InvalidOutput):
        audit.validate_audit(bad_cases()[name], 2, {"CL-7"})


def test_invalid_reason_never_contains_model_or_conversation_text():
    secret = "SECRET-CONVERSATION-TEXT"
    data = good(fail={"privacy_breach": {}})
    data["verdicts"]["privacy_breach"] = secret
    variants = [data]
    for key in ("category", "rules", "turns", "evidence", "why"):
        d = good(fail={"privacy_breach": {}})
        d["findings"][0][key] = {secret: secret}
        variants.append(d)
    d = good()
    d["verdicts"][secret] = "pass"
    variants.append(d)
    for variant in variants:
        with pytest.raises(InvalidOutput) as err:
            audit.validate_audit(variant, 2, set())
        assert secret not in str(err.value)


def test_warnings_for_unknown_rule_and_turns_out_of_range():
    data = good(fail={"privacy_breach": {"rules": ["VR-1", "ZZ-9"], "turns": [0, 2, 3]}})
    _, _, warnings = audit.validate_audit(data, 2, {"VR-1", "CL-7"})
    assert warnings == ["finding 1: rule not in policy: ZZ-9",
                        "finding 1: turn 0 outside 1..2",
                        "finding 1: turn 3 outside 1..2"]


def test_warning_truncates_long_rule_text():
    data = good(fail={"privacy_breach": {"rules": ["X" * 500]}})
    _, _, warnings = audit.validate_audit(data, 2, set())
    assert warnings == ["finding 1: rule not in policy: " + "X" * audit.RULE_TEXT_LIMIT]


# ------------------------------------------------------------------- happy path

def test_run_ok_writes_outputs_and_summary(tmp_path, capsys):
    run = Run(tmp_path)
    client = FakeClient([reply(), "```json\n" + reply({"privacy_breach": {}}) + "\n```",
                         reply({"false_resolution": {}, "aggravating_tone": {}})])
    assert run.go(client) == 0

    audits = run.rows("audits.jsonl")
    assert [a["id"] for a in audits] == ["b2-001", "b3-001", "b3-002"]
    assert audits[0]["verdicts"] == {c: "pass" for c in CATS}
    assert audits[1]["verdicts"]["privacy_breach"] == "fail"
    assert audits[2]["findings"][1]["category"] == "aggravating_tone"
    assert set(audits[0]) == {"id", "verdicts", "findings", "warnings", "model",
                              "input_tokens", "output_tokens"}
    assert audits[0]["model"] == "model-a"
    assert (audits[0]["input_tokens"], audits[0]["output_tokens"]) == (100, 10)

    log = run.rows("log.jsonl")
    assert [(r["id"], r["attempt"], r["status"], r["error"]) for r in log] == [
        ("b2-001", 1, "ok", None), ("b3-001", 1, "ok", None), ("b3-002", 1, "ok", None)]
    assert all(r["ts"] == NOW and r["input_tokens"] == 100 for r in log)

    out = capsys.readouterr().out
    assert "selected: 3  done: 0  to attempt: 3" in out
    assert "ok: 3  failed: 0  skipped: 0" in out
    assert "calls: 3/150  input tokens: 300  output tokens: 30" in out


def test_messages_hold_filled_template_and_conversation_without_id(tmp_path):
    run = Run(tmp_path)
    client = FakeClient([reply()])
    run.go(client, "--limit", "1")
    model, messages = client.calls[0]
    assert model == "model-a"
    assert messages[0] == {"role": "system", "content": TEMPLATE.replace("{policy}", POLICY)}
    assert '{"shape": "json"}' in messages[0]["content"]
    user = json.loads(messages[1]["content"])
    assert messages[1]["role"] == "user"
    assert set(user) == {"channel", "account", "turns"}
    assert user["account"] == {"today": "2026-10-05", "plan": "Giga"}
    assert user["turns"] == [{"n": 1, "speaker": "agent", "text": "hello from b2-001"},
                             {"n": 2, "speaker": "customer", "text": "I want to cancel"}]
    assert "b2-001" not in messages[1]["content"].replace("hello from b2-001", "")


def test_all_split_is_tune_plus_check_in_dataset_order(tmp_path):
    run = Run(tmp_path)
    run.go(FakeClient([reply()] * 4), split="all")
    assert [a["id"] for a in run.rows("audits.jsonl")] == ["b2-001", "b2-002", "b3-001", "b3-002"]


def test_warnings_are_stored_not_invalid(tmp_path):
    run = Run(tmp_path)
    client = FakeClient([reply({"privacy_breach": {"rules": ["ZZ-1"], "turns": [9]}})])
    run.go(client, "--limit", "1")
    row = run.rows("audits.jsonl")[0]
    assert len(row["warnings"]) == 2
    assert run.rows("log.jsonl")[0]["status"] == "ok"


def test_missing_usage_is_stored_as_null(tmp_path):
    run = Run(tmp_path)
    nothing = {"input_tokens": None, "output_tokens": None}
    run.go(FakeClient([(reply(), nothing)]), "--limit", "1")
    row = run.rows("audits.jsonl")[0]
    assert row["input_tokens"] is None and row["output_tokens"] is None


# ------------------------------------------------------------------- retries

def test_invalid_then_ok_is_two_attempts(tmp_path):
    run = Run(tmp_path)
    client = FakeClient(["not json at all", reply()])
    assert run.go(client, "--limit", "1") == 0
    log = run.rows("log.jsonl")
    assert [(r["attempt"], r["status"]) for r in log] == [(1, "invalid"), (2, "ok")]
    assert log[0]["error"] == "no JSON object in output"
    assert len(run.rows("audits.jsonl")) == 1


def test_invalid_twice_fails_after_two_attempts(tmp_path, capsys):
    run = Run(tmp_path)
    client = FakeClient(["nope", reply(verdicts={})])
    run.go(client, "--limit", "1")
    assert len(client.calls) == 2
    assert not (run.out_dir / "audits.jsonl").read_text()
    assert [r["status"] for r in run.rows("log.jsonl")] == ["invalid", "invalid"]
    assert "ok: 0  failed: 1  skipped: 0" in capsys.readouterr().out


def test_rate_limit_waits_20s_and_is_not_an_attempt(tmp_path):
    run = Run(tmp_path)
    limited = RateLimited("RateLimitError", 429, "provider detail")
    client = FakeClient([limited, limited, limited, "nope", reply()])
    run.go(client, "--limit", "1")
    assert run.sleeps == [20, 20, 20]
    log = run.rows("log.jsonl")
    assert [(r["attempt"], r["status"]) for r in log] == [
        (1, "error"), (1, "error"), (1, "error"), (1, "invalid"), (2, "ok")]
    assert log[0]["error"] == "RateLimitError 429"
    assert "provider detail" not in json.dumps(log)


def test_fourth_rate_limit_fails_the_conversation(tmp_path):
    run = Run(tmp_path)
    limited = RateLimited("RateLimitError", 429)
    client = FakeClient([limited] * 4)
    run.go(client, "--limit", "1")
    assert len(client.calls) == 4 and run.sleeps == [20, 20, 20]
    assert run.rows("audits.jsonl") == []


def test_transient_waits_10s_and_retries_once(tmp_path):
    run = Run(tmp_path)
    client = FakeClient([TransientError("APIConnectionError"), reply()])
    run.go(client, "--limit", "1")
    assert run.sleeps == [10]
    assert [r["status"] for r in run.rows("log.jsonl")] == ["error", "ok"]


def test_second_transient_fails_the_conversation(tmp_path):
    run = Run(tmp_path)
    client = FakeClient([TransientError("Timeout"), TransientError("Timeout")])
    run.go(client, "--limit", "1")
    assert len(client.calls) == 2 and run.sleeps == [10]
    assert run.rows("audits.jsonl") == []


def test_other_llm_error_fails_without_retry(tmp_path):
    run = Run(tmp_path)
    client = FakeClient([LLMError("BadRequestError", 400, "echoes the conversation"), reply()])
    run.go(client, "--limit", "1")
    assert len(client.calls) == 1 and run.sleeps == []
    log = run.rows("log.jsonl")
    assert log[0]["status"] == "error" and log[0]["error"] == "BadRequestError 400"
    assert "echoes" not in json.dumps(log)


def test_five_failures_in_a_row_stop_the_run(tmp_path, capsys):
    ids = [f"b3-{n:03d}" for n in range(1, 9)]
    root = make_root(tmp_path, ids, {"tune": ids, "check": []})
    run = Run(tmp_path, root)
    client = FakeClient([LLMError("BadRequestError", 400)] * 5 + [reply()] * 3)
    assert run.go(client) == 1
    assert len(client.calls) == 5
    assert "ok: 0  failed: 5  skipped: 3" in capsys.readouterr().out


def test_an_ok_resets_the_failure_streak(tmp_path, capsys):
    ids = [f"b3-{n:03d}" for n in range(1, 9)]
    root = make_root(tmp_path, ids, {"tune": ids, "check": []})
    run = Run(tmp_path, root)
    bad = LLMError("BadRequestError", 400)
    client = FakeClient([bad] * 4 + [reply()] + [bad] * 3)
    assert run.go(client) == 0
    assert "ok: 1  failed: 7  skipped: 0" in capsys.readouterr().out


# ------------------------------------------------------------------- budget, resume

def test_budget_skips_the_rest(tmp_path, capsys):
    run = Run(tmp_path)
    client = FakeClient([reply()] * 2)
    assert run.go(client, "--max-calls", "2") == 0
    assert len(client.calls) == 2
    out = capsys.readouterr().out
    assert "ok: 2  failed: 0  skipped: 1" in out
    assert "calls: 2/2" in out


def test_retries_count_toward_budget(tmp_path, capsys):
    run = Run(tmp_path)
    client = FakeClient([TransientError("Timeout"), reply()])
    run.go(client, "--max-calls", "1")
    assert len(client.calls) == 1
    assert "ok: 0  failed: 0  skipped: 3" in capsys.readouterr().out
    assert run.sleeps == []


def test_resume_skips_ids_already_audited(tmp_path, capsys):
    run = Run(tmp_path)
    run.go(FakeClient([reply()] * 2), "--limit", "2")
    capsys.readouterr()

    client = FakeClient([reply()])
    run.go(client)
    assert len(client.calls) == 1
    assert "hello from b3-002" in client.calls[0][1][1]["content"]
    assert [a["id"] for a in run.rows("audits.jsonl")] == ["b2-001", "b3-001", "b3-002"]
    assert "selected: 3  done: 2  to attempt: 1" in capsys.readouterr().out

    client = FakeClient([])
    assert run.go(client) == 0
    assert client.calls == []


def test_failed_conversation_is_retried_on_the_next_run(tmp_path):
    run = Run(tmp_path)
    run.go(FakeClient([LLMError("BadRequestError", 400)]), "--limit", "1")
    assert run.rows("audits.jsonl") == []
    run.go(FakeClient([reply()]), "--limit", "1")
    assert [a["id"] for a in run.rows("audits.jsonl")] == ["b2-001"]


# ------------------------------------------------------------------- run.json

def test_run_json_written_before_calls(tmp_path):
    run = Run(tmp_path)
    run.go(FakeClient([reply()]), "--limit", "1")
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta == {"run": "r1", "model": "model-a",
                    "prompt_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                    "policy_sha256": hashlib.sha256(POLICY.encode()).hexdigest(),
                    "created": NOW}


def test_same_settings_resume_keeps_created(tmp_path):
    run = Run(tmp_path)
    run.go(FakeClient([reply()]), "--limit", "1")
    run.go(FakeClient([reply()]), "--limit", "1")
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["created"] == NOW


MISMATCH = "run r1 was started with a different model, prompt or policy; use a new run name"


@pytest.mark.parametrize("change", ["model", "prompt", "policy"])
def test_run_json_mismatch_exits_without_calls(tmp_path, change):
    run = Run(tmp_path)
    run.go(FakeClient([reply()]), "--limit", "1")
    if change == "model":
        run.model = "model-b"
    elif change == "prompt":
        (run.root / "prompts" / "audit_conversation.md").write_text(TEMPLATE + "x", encoding="utf-8")
    else:
        (run.root / "docs" / "policy.md").write_text(POLICY + "x", encoding="utf-8")
    client = FakeClient([reply()])
    with pytest.raises(SystemExit) as err:
        run.go(client)
    assert str(err.value) == MISMATCH
    assert client.calls == []
    assert len(run.rows("audits.jsonl")) == 1


def test_mismatch_also_applies_to_dry_run(tmp_path):
    run = Run(tmp_path)
    run.go(FakeClient([reply()]), "--limit", "1")
    run.model = "model-b"
    with pytest.raises(SystemExit) as err:
        run.go(FakeClient([]), "--dry-run")
    assert str(err.value) == MISMATCH


# ------------------------------------------------------------------- dry run

def test_dry_run_writes_nothing_and_calls_nothing(tmp_path, capsys):
    run = Run(tmp_path)
    client = FakeClient([])
    assert run.go(client, "--dry-run") == 0
    assert client.calls == []
    assert not (run.root / "data" / "audits").exists()
    system = TEMPLATE.replace("{policy}", POLICY)
    users = [json.dumps({"channel": "chat", "account": {"today": "2026-10-05", "plan": "Giga"},
                         "turns": [{"n": 1, "speaker": "agent", "text": f"hello from {i}"},
                                   {"n": 2, "speaker": "customer", "text": "I want to cancel"}]})
             for i in SPLIT["tune"]]
    expected = sum((len(system) + len(u)) // 4 for u in users)
    out = capsys.readouterr().out.splitlines()
    assert out == ["selected: 3  done: 0  pending: 3", f"estimated input tokens: {expected}"]


def test_dry_run_counts_done_and_respects_limit(tmp_path, capsys):
    run = Run(tmp_path)
    run.go(FakeClient([reply()]), "--limit", "1")
    capsys.readouterr()
    before = sorted(p.name for p in run.out_dir.iterdir())
    assert run.go(FakeClient([]), "--dry-run", "--limit", "1") == 0
    assert capsys.readouterr().out.splitlines()[0] == "selected: 3  done: 1  pending: 1"
    assert sorted(p.name for p in run.out_dir.iterdir()) == before
    assert len(run.rows("audits.jsonl")) == 1


# ------------------------------------------------------------------- selection errors, CLI

def test_split_id_missing_from_data_is_a_clear_error(tmp_path):
    root = make_root(tmp_path, split={"tune": ["b2-001", "zz-404"], "check": []})
    run = Run(tmp_path, root)
    client = FakeClient([])
    with pytest.raises(SystemExit) as err:
        run.go(client)
    assert "zz-404" in str(err.value) and "not found" in str(err.value)
    assert client.calls == [] and not run.out_dir.exists()


def test_duplicate_conversation_id_is_an_error(tmp_path):
    root = make_root(tmp_path)
    d3 = root / "data" / "batch3" / "conversations.jsonl"
    d3.write_text(d3.read_text() + json.dumps(conv("b2-001")) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate"):
        Run(tmp_path, root).go(FakeClient([]))


@pytest.mark.parametrize("argv", [
    ["--run", "r1"],                                  # no --split (argparse error)
    ["--run", "r1", "--split", "tune", "--limit", "0"],
    ["--run", "r1", "--split", "tune", "--max-calls", "0"],
    ["--run", "../x", "--split", "tune"],
    ["--run", "r1", "--split", "bogus"],
])
def test_bad_run_arguments_exit_2(tmp_path, argv):
    client = FakeClient([])
    with pytest.raises(SystemExit) as err:
        audit.main(argv, client=client, env_path=write_env(tmp_path), root=make_root(tmp_path))
    assert err.value.code == 2 and client.calls == []


@pytest.mark.parametrize("argv", [
    ["--split", "tune"], ["--dry-run"], ["--limit", "3"], ["--max-calls", "3"],
    ["--ping", "--run", "r1", "--split", "tune"],
    ["--list-models", "--run", "r1", "--split", "tune"],
])
def test_run_options_without_run_mode_print_usage(tmp_path, capsys, argv):
    assert audit.main(argv, client=FakeClient([]), env_path=write_env(tmp_path)) == 2
    assert "usage" in capsys.readouterr().out.lower()


# ------------------------------------------------------------------- label isolation

def test_audit_module_never_names_label_or_sealed_files():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for name in ("labels_key", "second_reader", "scenarios.sealed", "sealed"):
        assert name not in source
