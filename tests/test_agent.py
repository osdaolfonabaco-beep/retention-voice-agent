"""Offline tests for retention/agent.py and load_agent_settings."""

import copy
import json
from pathlib import Path

import pytest

from retention import agent, config
from retention.llm import LLMError, RateLimited, TransientError
from retention.tools import TOOL_SPECS, AccountSession

NOW = "2026-01-01T00:00:00+00:00"
USAGE = {"input_tokens": 100, "output_tokens": 10}
ADDRESS = "412 Juniper Ct, Flagstaff, AZ 86001"
MISMATCH = ("run r1 was started with different models, prompts, policy or scenarios; "
            "use a new run name")
POLICY = "POLICY-BODY {channel} {today}\n"
AGENT_TEMPLATE = ("AGENT {channel} {today} {agent_name}\n<policy>\n{policy}\n</policy>\n")
CUSTOMER_TEMPLATE = "CUSTOMER {channel}\nBrief: {brief}\n"
CALLER_ID = "I want to cancel. I am Teresa Galloway at 412 Juniper Ct, last payment $60."


# ---------------------------------------------------------------- fakes and builders

def say(text):
    return {"content": text, "tool_calls": []}


def use(name, args, text="", call_id="c1"):
    return {"content": text, "tool_calls": [
        {"id": call_id, "name": name, "arguments": json.dumps(args)}]}


class FakeClient:
    """Plays one script per role: agent items are messages, customer items are str."""

    def __init__(self, agent_script=(), customer_script=()):
        self.agent_script, self.customer_script = list(agent_script), list(customer_script)
        self.agent_calls, self.customer_calls = [], []

    def chat_with_tools(self, model, messages, tools=None, reasoning_effort=None):
        self.agent_calls.append({"model": model, "messages": copy.deepcopy(messages),
                                 "tools": tools, "reasoning_effort": reasoning_effort})
        item = self.agent_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, dict(USAGE)

    def complete_with_usage(self, model, messages):
        self.customer_calls.append({"model": model, "messages": copy.deepcopy(messages)})
        item = self.customer_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, dict(USAGE)


def make_account(**over):
    account = {
        "today": "2026-10-05", "today_weekday": "Monday",
        "account_holder": "Teresa Galloway", "authorized_users": [],
        "service_address": ADDRESS,
        "on_file_for_verification": {"last_payment_amount": 60.0, "pin": "PIN-SENTINEL-77"},
        "plan": "Plus", "monthly_price": 60, "months_active": 20, "stage": "Current",
        "days_past_due": 0, "past_due_balance": 0, "late_fees_included": 0,
        "next_bill": {"due_date": "2026-10-24", "amount": 60},
        "retention_discount_eligible": True, "confirmed_service_issue": None,
    }
    account.update(over)
    return account


def make_scenario(sid, **over):
    scenario = {"id": sid, "channel": "phone", "situation": "SENTINEL-SITUATION",
                "agent_name": "Marisa", "account": make_account(),
                "caller": "SENTINEL-CALLER",
                "customer_brief": "BRIEF-TEXT you want to cancel.",
                "expected": ["SENTINEL-EXPECTED verify first"]}
    scenario.update(over)
    return scenario


def write_scenarios(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(make_scenario(i)) + "\n" for i in ids), encoding="utf-8")


def write_env(tmp_path, agent_model="agent-m", customer_model="cust-m"):
    path = tmp_path / ".env"
    path.write_text(
        "AUDITOR_BASE_URL=https://h/v1\nAUDITOR_API_KEY=sk-x\n"
        f"AGENT_MODEL={agent_model}\nCUSTOMER_MODEL={customer_model}\n", encoding="utf-8")
    return path


def make_root(tmp_path, ids=("ag-01", "ag-02", "ag-03")):
    root = tmp_path / "proj"
    (root / "prompts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "prompts" / "agent_system.md").write_text(AGENT_TEMPLATE, encoding="utf-8")
    (root / "prompts" / "customer_simulator.md").write_text(CUSTOMER_TEMPLATE, encoding="utf-8")
    (root / "docs" / "policy.md").write_text(POLICY, encoding="utf-8")
    write_scenarios(root / "data" / "agent" / "scenarios.jsonl", ids)
    return root


class Run:
    def __init__(self, tmp_path, ids=("ag-01", "ag-02", "ag-03"), agent_model="agent-m",
                 customer_model="cust-m"):
        self.tmp_path, self.root = tmp_path, make_root(tmp_path, ids)
        self.agent_model, self.customer_model = agent_model, customer_model
        self.sleeps = []

    @property
    def out_dir(self):
        return self.root / "data" / "agent_runs" / "r1"

    def go(self, client, *argv, run="r1"):
        env = write_env(self.tmp_path, self.agent_model, self.customer_model)
        return agent.main(["--run", run, *argv], client=client, env_path=env, root=self.root,
                          sleep=self.sleeps.append, now=lambda: NOW)

    def rows(self, name):
        path = self.out_dir / name
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def quick(n=1):
    """Script for n scenarios that end at once: the agent greets, the customer says [END]."""
    return FakeClient([say("Hi, this is Marisa.")] * n, ["[END]"] * n)


# ---------------------------------------------------------------- agent_turn

def test_agent_turn_without_tools_returns_the_text():
    client = FakeClient([say("Hello there.")])
    session = AccountSession(make_account(), "ag-01")
    start = [{"role": "user", "content": "[contact started]"}]
    text, messages, usage = agent.agent_turn(client, "m", "SYSTEM", start, session)
    assert text == "Hello there."
    assert messages == start + [{"role": "assistant", "content": "Hello there."}]
    assert start == [{"role": "user", "content": "[contact started]"}]  # input not mutated
    assert usage == USAGE
    sent = client.agent_calls[0]
    assert sent["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert sent["tools"] == TOOL_SPECS and sent["model"] == "m"


def test_agent_turn_runs_tools_and_feeds_results_back():
    client = FakeClient([
        use("verify_identity", {"full_name": "Teresa Galloway", "service_address": ADDRESS,
                                "last_payment_amount": 60}, text="One moment.", call_id="call-9"),
        say("You are verified."),
    ])
    session = AccountSession(make_account(), "ag-01")
    session.hear("I am Teresa Galloway at 412 Juniper Ct, and my last payment was $60.")
    text, messages, usage = agent.agent_turn(
        client, "m", "SYSTEM", [{"role": "user", "content": "I am Teresa"}], session)
    assert text == "One moment. You are verified."
    assert usage == {"input_tokens": 200, "output_tokens": 20}
    assistant, tool, final = messages[1:]
    assert assistant["role"] == "assistant" and assistant["content"] == "One moment."
    assert assistant["tool_calls"] == [{
        "id": "call-9", "type": "function",
        "function": {"name": "verify_identity", "arguments": json.dumps({
            "full_name": "Teresa Galloway", "service_address": ADDRESS,
            "last_payment_amount": 60})}}]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call-9"
    assert json.loads(tool["content"])["result"] == "verified_account_holder"
    assert final == {"role": "assistant", "content": "You are verified."}
    assert [c["name"] for c in session.calls] == ["verify_identity"]
    second = client.agent_calls[1]["messages"]
    assert second[-1] == tool and second[0]["role"] == "system"


def test_agent_turn_handles_several_tool_calls_in_one_message():
    message = {"content": "", "tool_calls": [
        {"id": "a", "name": "get_account", "arguments": "{}"},
        {"id": "b", "name": "end_contact", "arguments": json.dumps({"status": "no_change"})}]}
    client = FakeClient([message, say("Bye.")])
    session = AccountSession(make_account(), "ag-01")
    _, messages, _ = agent.agent_turn(client, "m", "S", [], session)
    assert [m.get("tool_call_id") for m in messages if m["role"] == "tool"] == ["a", "b"]
    assert messages[0]["content"] is None  # an empty text next to tool calls is sent as null
    assert session.ended


def test_agent_turn_stops_after_six_tool_rounds_with_a_last_call_without_tools():
    rounds = [use("get_account", {}, call_id=f"c{i}") for i in range(6)]
    client = FakeClient(rounds + [say("Done.")])
    session = AccountSession(make_account(), "ag-01")
    text, messages, _ = agent.agent_turn(client, "m", "S", [], session)
    assert text == "Done." and len(client.agent_calls) == 7 and len(session.calls) == 6
    assert [c["tools"] is not None for c in client.agent_calls] == [True] * 6 + [False]
    assert messages[-1] == {"role": "assistant", "content": "Done."}


def test_every_agent_call_in_a_turn_carries_the_reasoning_effort():
    client = FakeClient([use("get_account", {}, call_id=f"c{i}") for i in range(6)] + [say("Done.")])
    agent.agent_turn(client, "m", "S", [], AccountSession(make_account(), "ag-01"))
    assert agent.AGENT_REASONING_EFFORT == "none"
    assert [c["reasoning_effort"] for c in client.agent_calls] == ["none"] * 7
    assert client.agent_calls[-1]["tools"] is None          # the final call too


def test_agent_turn_reasoning_effort_can_be_overridden():
    client = FakeClient([say("Hi.")])
    agent.agent_turn(client, "m", "S", [], AccountSession(make_account(), "ag-01"),
                     reasoning_effort="low")
    assert client.agent_calls[0]["reasoning_effort"] == "low"


def test_agent_run_sends_the_reasoning_effort_through_the_guard(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = quick()
    run.go(client)
    assert [c["reasoning_effort"] for c in client.agent_calls] == ["none"]


def test_agent_turn_ignores_tool_calls_in_the_last_call():
    client = FakeClient([use("get_account", {}, call_id=f"c{i}") for i in range(7)])
    session = AccountSession(make_account(), "ag-01")
    text, _, _ = agent.agent_turn(client, "m", "S", [], session)
    assert text == "" and len(client.agent_calls) == 7 and len(session.calls) == 6


def test_agent_turn_without_tools_makes_one_call_and_runs_no_tool():
    client = FakeClient([use("verify_identity", {"full_name": "X"}, text="Hello!"),
                         say("never reached")])
    session = AccountSession(make_account(), "ag-01")
    text, messages, _ = agent.agent_turn(client, "m", "S", [], session, allow_tools=False)
    assert text == "Hello!" and len(client.agent_calls) == 1
    assert client.agent_calls[0]["tools"] is None
    assert client.agent_calls[0]["reasoning_effort"] == "none"
    assert session.calls == [] and len(client.agent_script) == 1
    assert messages[-1] == {"role": "assistant", "content": "Hello!"}


def test_opening_call_of_a_contact_has_no_tools_and_later_calls_do(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient(
        [use("verify_identity", {"full_name": "Invented", "service_address": "x",
                                 "last_payment_amount": 1}, text="Hello, Marisa here."),
         use("get_account", {}),
         say("How can I help?"),
         use("end_contact", {"status": "no_change"}, text="Bye."),
         say("")],
        ["I want to cancel.", "No."])
    assert run.go(client) == 0
    tools = [c["tools"] for c in client.agent_calls]
    assert tools[0] is None and all(t == TOOL_SPECS for t in tools[1:])
    [calls] = run.rows("tool_calls.jsonl")
    assert "verify_identity" not in [c["name"] for c in calls["calls"]]


# ---------------------------------------------------------------- a full contact

def test_full_contact_with_tools(tmp_path, capsys):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient(
        [say("Hello, Marisa here."),
         use("verify_identity", {"full_name": "Teresa Galloway", "service_address": ADDRESS,
                                 "last_payment_amount": 60}),
         say("Thanks, you are verified. Do you want to cancel?"),
         use("end_contact", {"status": "no_change"}, text="Goodbye!"),
         say("")],
        [CALLER_ID, "Yes."])
    assert run.go(client) == 0

    [conv] = run.rows("conversations.jsonl")
    assert conv == {"id": "ag-01", "channel": "phone", "account": make_account(), "turns": [
        {"speaker": "agent", "text": "Hello, Marisa here."},
        {"speaker": "customer", "text": CALLER_ID},
        {"speaker": "agent", "text": "Thanks, you are verified. Do you want to cancel?"},
        {"speaker": "customer", "text": "Yes."},
        {"speaker": "agent", "text": "Goodbye!"}]}
    [calls] = run.rows("tool_calls.jsonl")
    assert calls["id"] == "ag-01" and calls["end_status"] == "no_change"
    assert calls["stop_reason"] == "end_contact"
    assert [(c["after_turn"], c["name"]) for c in calls["calls"]] == [
        (2, "verify_identity"), (4, "end_contact")]
    assert calls["calls"][0]["arguments"]["last_payment_amount"] == 60
    assert calls["calls"][0]["result"] == {"result": "verified_account_holder", "failed_attempts": 0}

    # the agent starts from an unrecorded first user message
    first = client.agent_calls[0]["messages"]
    assert first[1:] == [{"role": "user", "content": "[contact started]"}]
    assert client.agent_calls[0]["model"] == "agent-m"
    # the customer sees agent turns as user and its own as assistant
    customer = client.customer_calls
    assert customer[0]["model"] == "cust-m"
    assert customer[0]["messages"][1:] == [
        {"role": "user", "content": "Hello, Marisa here."}]
    assert customer[1]["messages"][1:] == [
        {"role": "user", "content": "Hello, Marisa here."},
        {"role": "assistant", "content": CALLER_ID},
        {"role": "user", "content": "Thanks, you are verified. Do you want to cancel?"}]
    # the agent sees customer turns as user messages
    assert client.agent_calls[1]["messages"][-1] == {
        "role": "user", "content": CALLER_ID}

    log = run.rows("log.jsonl")
    assert [r["role"] for r in log] == ["agent", "customer", "agent", "agent", "customer",
                                        "agent", "agent"]
    assert log[0] == {"ts": NOW, "id": "ag-01", "role": "agent", "status": "ok", "error": None,
                      "input_tokens": 100, "output_tokens": 10}
    out = capsys.readouterr().out
    assert "ok: 1  failed: 0  skipped: 0" in out and "calls: 7/2000" in out
    assert "agent: input tokens 500  output tokens 50" in out
    assert "customer: input tokens 200  output tokens 20" in out


def test_prompts_are_filled_and_policy_text_is_not_substituted_again(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = quick()
    run.go(client)
    system = client.agent_calls[0]["messages"][0]["content"]
    assert system == ("AGENT phone 2026-10-05 Marisa\n<policy>\nPOLICY-BODY {channel} {today}\n\n"
                      "</policy>\n")
    customer = client.customer_calls[0]["messages"][0]["content"]
    assert customer == "CUSTOMER phone\nBrief: BRIEF-TEXT you want to cancel.\n"


def test_stop_when_the_customer_replies_end(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient([say("Hi.")], ["  [END]\n"])
    run.go(client)
    [conv] = run.rows("conversations.jsonl")
    assert conv["turns"] == [{"speaker": "agent", "text": "Hi."}]  # [END] is not recorded
    [calls] = run.rows("tool_calls.jsonl")
    assert calls["stop_reason"] == "customer_end" and calls["end_status"] is None


def test_end_contact_without_text_records_nothing_for_that_turn(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient([say("Hi."), use("end_contact", {"status": "resolved"}), say("")], ["Bye."])
    run.go(client)
    [conv] = run.rows("conversations.jsonl")
    assert [t["speaker"] for t in conv["turns"]] == ["agent", "customer"]
    assert run.rows("tool_calls.jsonl")[0]["end_status"] == "resolved"


def test_stop_after_forty_turns(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient([say(f"agent {i}") for i in range(20)],
                        [f"customer {i}" for i in range(20)])
    run.go(client)
    [conv] = run.rows("conversations.jsonl")
    assert len(conv["turns"]) == 40 and conv["turns"][-1]["speaker"] == "customer"
    assert run.rows("tool_calls.jsonl")[0]["stop_reason"] == "max_turns"
    assert client.agent_script == [] and client.customer_script == []


# ---------------------------------------------------------------- what models may see

def test_expected_and_the_account_never_reach_a_model(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient(
        [say("Hello."),
         use("get_account", {}),                       # refused: not verified yet
         use("verify_identity", {"full_name": "Teresa Galloway", "service_address": ADDRESS,
                                 "last_payment_amount": 60}, call_id="c2"),
         use("get_account", {}, call_id="c3"),
         say("I can see your plan."),
         use("end_contact", {"status": "no_change"}), say("")],
        ["Please check my plan.", "Thanks, bye."])
    run.go(client)

    everything = json.dumps([client.agent_calls, client.customer_calls])
    for word in ("SENTINEL-EXPECTED", "SENTINEL-SITUATION", "SENTINEL-CALLER", "expected"):
        assert word not in everything
    contents = [m["content"] for c in client.agent_calls + client.customer_calls
                for m in c["messages"] if isinstance(m.get("content"), str)]
    assert not any("on_file_for_verification" in c for c in contents)
    assert not any(json.dumps(make_account()) in c for c in contents)
    agent_text = json.dumps(client.agent_calls)
    assert "PIN-SENTINEL-77" not in agent_text      # on-file value, never exposed
    assert "BRIEF-TEXT" not in agent_text           # the brief is for the customer only
    # before any tool result, the agent knows nothing about the account
    first = json.dumps(client.agent_calls[0]["messages"])
    assert "Teresa" not in first and "Juniper" not in first and "Plus" not in first
    # the customer side never gets tool traffic or the account
    customer_text = json.dumps(client.customer_calls)
    assert "tool_call" not in customer_text and "Flagstaff" not in customer_text
    # get_account does reach the agent, after verification
    assert "Teresa Galloway" in json.dumps(client.agent_calls[-1]["messages"])


def test_empty_agent_text_without_end_contact_fails_the_scenario(tmp_path, capsys):
    run = Run(tmp_path, ids=("ag-01",))
    assert run.go(FakeClient([say("")])) == 0
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")
    assert "[1/1] ag-01 failed: the agent said nothing and did not end the contact" in (
        capsys.readouterr().out)


def test_empty_customer_reply_fails_the_scenario(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    run.go(FakeClient([say("Hi.")], ["   "]))
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")


# ---------------------------------------------------------------- resume, limit, dry run

def test_resume_skips_done_ids(tmp_path, capsys):
    run = Run(tmp_path)
    assert run.go(quick(), "--limit", "1") == 0
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-01"]
    created = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))["created"]

    capsys.readouterr()
    second = quick(2)
    assert run.go(second) == 0
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-01", "ag-02", "ag-03"]
    assert len(second.agent_calls) == 2
    assert "scenarios: 3  done: 1  to attempt: 2" in capsys.readouterr().out
    assert json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))["created"] == created

    third = quick()
    assert run.go(third) == 0 and third.agent_calls == []


def test_run_json_has_models_and_hashes(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    run.go(quick())
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert set(meta) == {"run", "agent_model", "customer_model", "agent_prompt",
                         "agent_prompt_sha256", "customer_prompt_sha256", "policy_sha256",
                         "scenarios_sha256", "agent_reasoning_effort", "guards", "created"}
    assert meta["guards"] is True and meta["agent_prompt"] == "prompts/agent_system.md"
    assert meta["agent_reasoning_effort"] == "none"
    assert meta["run"] == "r1" and meta["agent_model"] == "agent-m"
    assert meta["customer_model"] == "cust-m" and meta["created"] == NOW
    assert len(meta["policy_sha256"]) == 64


def edit(path_parts, text):
    def change(run):
        (run.root.joinpath(*path_parts)).write_text(text, encoding="utf-8")
    return change


@pytest.mark.parametrize("change", [
    lambda run: setattr(run, "agent_model", "agent-other"),
    lambda run: setattr(run, "customer_model", "cust-other"),
    edit(("prompts", "agent_system.md"), AGENT_TEMPLATE + "more\n"),
    edit(("prompts", "customer_simulator.md"), CUSTOMER_TEMPLATE + "more\n"),
    edit(("docs", "policy.md"), POLICY + "more\n"),
    lambda run: write_scenarios(run.root / "data" / "agent" / "scenarios.jsonl",
                                ["ag-01", "ag-02"]),
])
def test_run_json_mismatch_exits_before_any_call(tmp_path, change):
    run = Run(tmp_path, ids=("ag-01", "ag-02", "ag-03"))
    run.go(quick(), "--limit", "1")
    change(run)
    client = quick()
    with pytest.raises(SystemExit) as err:
        run.go(client)
    assert str(err.value) == MISMATCH and client.agent_calls == []
    with pytest.raises(SystemExit) as err:           # a dry run is refused too
        run.go(client, "--dry-run")
    assert str(err.value) == MISMATCH


def test_run_json_with_a_different_reasoning_effort_is_refused(tmp_path, monkeypatch):
    run = Run(tmp_path, ids=("ag-01", "ag-02"))
    run.go(quick(), "--limit", "1")
    monkeypatch.setattr(agent, "AGENT_REASONING_EFFORT", "low")
    client = quick()
    with pytest.raises(SystemExit) as err:
        run.go(client)
    assert str(err.value) == MISMATCH and client.agent_calls == []


def test_run_json_without_the_reasoning_effort_is_refused(tmp_path):
    run = Run(tmp_path, ids=("ag-01", "ag-02"))
    run.go(quick(), "--limit", "1")
    path = run.out_dir / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["agent_reasoning_effort"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        run.go(quick())
    assert str(err.value) == MISMATCH


def test_unreadable_run_json_counts_as_a_mismatch(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    run.go(quick())
    (run.out_dir / "run.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        run.go(quick())
    assert str(err.value) == MISMATCH


def test_dry_run_calls_nothing_and_writes_nothing(tmp_path, capsys):
    run = Run(tmp_path)
    client = quick()
    assert run.go(client, "--dry-run") == 0
    assert capsys.readouterr().out == "scenarios: 3  done: 0  pending: 3\n"
    assert client.agent_calls == [] and not (run.root / "data" / "agent_runs").exists()

    run.go(quick(), "--limit", "1")
    capsys.readouterr()
    assert run.go(client, "--dry-run") == 0
    assert capsys.readouterr().out == "scenarios: 3  done: 1  pending: 2\n"
    assert run.go(client, "--dry-run", "--limit", "1") == 0
    assert "pending: 1" in capsys.readouterr().out


def test_custom_scenarios_path(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    other = tmp_path / "other.jsonl"
    write_scenarios(other, ["zz-1", "zz-2"])
    run.go(quick(2), "--scenarios", str(other))
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["zz-1", "zz-2"]


# ---------------------------------------------------------------- failures, retries, budget

def test_failed_scenario_writes_nothing_and_the_run_goes_on(tmp_path, capsys):
    run = Run(tmp_path, ids=("ag-01", "ag-02"))
    client = FakeClient([say("Hi."), LLMError("BadRequestError", 400, "SECRET detail"),
                         say("Hi again.")],
                        ["Hello, I need help.", "[END]"])
    assert run.go(client) == 0
    # ag-01: the agent's second turn failed after one customer turn; ag-02 is complete
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-02"]
    assert [r["id"] for r in run.rows("tool_calls.jsonl")] == ["ag-02"]
    out = capsys.readouterr().out
    assert "[1/2] ag-01 failed: agent call failed: BadRequestError 400" in out
    assert "[2/2] ag-02 ok" in out and "SECRET" not in out
    log = run.rows("log.jsonl")
    errors = [r for r in log if r["status"] == "error"]
    assert errors == [{"ts": NOW, "id": "ag-01", "role": "agent", "status": "error",
                       "error": "BadRequestError 400", "input_tokens": None,
                       "output_tokens": None}]
    assert "SECRET" not in json.dumps(log)
    # a failed scenario is picked up again on the next run
    again = quick()
    run.go(again)
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-02", "ag-01"]


def test_three_failed_scenarios_in_a_row_stop_the_run(tmp_path, capsys):
    run = Run(tmp_path, ids=("a1", "a2", "a3", "a4", "a5"))
    boom = LLMError("BadRequestError", 400)
    client = FakeClient([boom, boom, boom, say("never used")], [])
    assert run.go(client) == 1
    assert len(client.agent_calls) == 3
    out = capsys.readouterr().out
    assert "ok: 0  failed: 3  skipped: 2" in out
    assert "stopped: 3 scenarios failed in a row" in out
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")


def test_a_success_resets_the_failure_streak(tmp_path):
    run = Run(tmp_path, ids=("a1", "a2", "a3", "a4", "a5"))
    boom = LLMError("BadRequestError", 400)
    client = FakeClient([boom, boom, say("Hi."), boom, boom], ["[END]"])
    assert run.go(client) == 0
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["a3"]


def test_rate_limit_waits_20s_and_retries_up_to_three_times(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    limited = RateLimited("RateLimitError", 429, "provider detail")
    client = FakeClient([limited, limited, say("Hi.")], ["[END]"])
    run.go(client)
    assert run.sleeps == [20, 20]
    assert len(run.rows("conversations.jsonl")) == 1
    log = run.rows("log.jsonl")
    assert [r["status"] for r in log] == ["error", "error", "ok", "ok"]
    assert log[0]["error"] == "RateLimitError 429" and "provider detail" not in json.dumps(log)

    run = Run(tmp_path / "second", ids=("ag-01",))
    run.go(FakeClient([limited] * 4), "--max-calls", "50")
    assert run.sleeps == [20, 20, 20]                      # the fourth failure is final
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")


def test_transient_error_waits_10s_and_retries_once(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    flaky = TransientError("APIConnectionError")
    run.go(FakeClient([say("Hi.")], [flaky, "[END]"]))
    assert run.sleeps == [10] and len(run.rows("conversations.jsonl")) == 1

    run = Run(tmp_path / "second", ids=("ag-01",))
    run.go(FakeClient([flaky, flaky]))
    assert run.sleeps == [10]
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")


def test_budget_counts_every_call_and_skips_the_unfinished_scenario(tmp_path, capsys):
    run = Run(tmp_path, ids=("ag-01", "ag-02"))
    client = FakeClient([say("one"), say("two"), say("three")], ["a", "b"])
    assert run.go(client, "--max-calls", "3") == 0
    out = capsys.readouterr().out
    assert "ok: 0  failed: 0  skipped: 2" in out and "calls: 3/3" in out
    assert "stopped: call budget used up" in out
    assert len(client.agent_calls) + len(client.customer_calls) == 3
    assert not (run.out_dir / "conversations.jsonl").read_text(encoding="utf-8")
    # nothing was marked done, so the next run starts them again
    run.go(quick(2))
    assert [r["id"] for r in run.rows("conversations.jsonl")] == ["ag-01", "ag-02"]


def test_budget_includes_tool_rounds_and_retries(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient([RateLimited("RateLimitError", 429), use("get_account", {}),
                         say("never")], [])
    run.go(client, "--max-calls", "2")
    assert len(client.agent_calls) == 2 and run.sleeps == [20]


# ---------------------------------------------------------------- command line

@pytest.mark.parametrize("argv", [
    [], ["--run", "../x"], ["--run", "r1", "--limit", "0"], ["--run", "r1", "--max-calls", "0"],
    ["--run", "r1", "--limit", "x"],
])
def test_bad_arguments_exit_with_code_2(tmp_path, argv):
    client = quick()
    with pytest.raises(SystemExit) as err:
        agent.main(argv, client=client, env_path=write_env(tmp_path), root=make_root(tmp_path))
    assert err.value.code == 2 and client.agent_calls == []


def test_missing_settings_are_named_without_values(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AUDITOR_BASE_URL=https://secret-host/v1\nAUDITOR_API_KEY=sk-secret\n",
                   encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        agent.main(["--run", "r1"], client=quick(), env_path=env, root=make_root(tmp_path))
    assert str(err.value) == "error: missing in .env: AGENT_MODEL, CUSTOMER_MODEL"


def test_missing_files_and_bad_scenarios(tmp_path):
    run = Run(tmp_path)
    (run.root / "prompts" / "customer_simulator.md").unlink()
    with pytest.raises(SystemExit) as err:
        run.go(quick())
    assert str(err.value).startswith("error: required file not found:")

    run = Run(tmp_path / "b")
    path = run.root / "data" / "agent" / "scenarios.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        run.go(quick())
    assert str(err.value) == f"error: bad scenario line in {path}"
    path.write_text("".join(json.dumps(make_scenario("x")) + "\n" for _ in range(2)),
                    encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        run.go(quick())
    assert str(err.value) == "error: duplicate scenario id x"


def test_real_scenarios_load():
    path = Path(__file__).resolve().parent.parent / "data" / "agent" / "scenarios.jsonl"
    scenarios, sha = agent.load_scenarios(path)
    assert len(scenarios) == 24 and len(sha) == 64
    assert all(set(s) == set(agent.SCENARIO_KEYS) for s in scenarios)  # no other field is kept


def test_agent_module_never_names_label_or_sealed_files():
    source = Path(agent.__file__).read_text(encoding="utf-8")
    for name in ("labels_key", "second_reader", "scenarios.sealed", "sealed", '"expected"'):
        assert name not in source


# ---------------------------------------------------------------- guards

VERIFY = {"full_name": "Teresa Galloway", "service_address": ADDRESS, "last_payment_amount": 60}
BLOCKED = {"blocked": "asked_a_question"}


def test_a_question_next_to_tool_calls_ends_the_turn_and_discards_the_calls():
    message = {"content": "Is that everything?", "tool_calls": [
        {"id": "a", "name": "end_contact", "arguments": json.dumps({"status": "resolved"})},
        {"id": "b", "name": "get_account", "arguments": "{}"}]}
    client = FakeClient([message])          # a second model call would fail: the script is empty
    session = AccountSession(make_account(), "ag-01")
    session.turn = 6
    text, messages, usage = agent.agent_turn(client, "m", "S", [], session)
    assert text == "Is that everything?"
    assert messages == [{"role": "assistant", "content": "Is that everything?"}]
    assert "tool_calls" not in messages[0] and len(client.agent_calls) == 1
    assert usage == USAGE and not session.ended
    assert session.calls == [
        {"after_turn": 6, "name": "end_contact", "arguments": {"status": "resolved"},
         "result": BLOCKED},
        {"after_turn": 6, "name": "get_account", "arguments": {}, "result": BLOCKED}]


def test_a_question_mark_in_a_later_round_keeps_the_earlier_text_and_results():
    client = FakeClient([
        use("get_account", {}, text="One moment.", call_id="c1"),
        use("end_contact", {"status": "resolved"}, text="Anything else?", call_id="c2")])
    session = AccountSession(make_account(), "ag-01")
    session.hear("Teresa Galloway, 412 Juniper Ct, $60")
    text, messages, _ = agent.agent_turn(client, "m", "S", [], session)
    assert text == "One moment. Anything else?"
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert messages[-1] == {"role": "assistant", "content": "Anything else?"}
    assert [c["result"] == BLOCKED for c in session.calls] == [False, True]
    assert not session.ended


def test_tool_calls_with_text_without_a_question_mark_run():
    client = FakeClient([use("end_contact", {"status": "resolved"}, text="Goodbye."), say("")])
    session = AccountSession(make_account(), "ag-01")
    text, messages, _ = agent.agent_turn(client, "m", "S", [], session)
    assert text == "Goodbye." and session.ended
    assert messages[0]["tool_calls"][0]["function"]["name"] == "end_contact"
    assert session.calls[0]["result"] == {"ended": True, "status": "resolved"}


def test_with_guards_off_a_question_next_to_tool_calls_runs_as_before():
    client = FakeClient([use("end_contact", {"status": "resolved"}, text="Anything else?"),
                         say("")])
    session = AccountSession(make_account(), "ag-01", grounding=False)
    text, messages, _ = agent.agent_turn(client, "m", "S", [], session, guards=False)
    assert text == "Anything else?" and session.ended and len(client.agent_calls) == 2
    assert messages[0]["tool_calls"] and session.calls[0]["result"]["ended"] is True


def test_guards_are_on_by_default_in_agent_turn():
    client = FakeClient([use("end_contact", {"status": "resolved"}, text="Done?")])
    session = AccountSession(make_account(), "ag-01")
    agent.agent_turn(client, "m", "S", [], session)
    assert not session.ended and session.calls[0]["result"] == BLOCKED


def test_guarded_contact_blocks_ungrounded_values_and_questions_with_calls(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient(
        [say("Hello, Marisa here."),
         use("verify_identity", VERIFY, call_id="v1"),                  # the customer said nothing yet
         say("Could you tell me your full name and address?"),
         use("verify_identity", VERIFY, call_id="v2"),
         use("end_contact", {"status": "resolved"}, text="Is that all?", call_id="e1"),
         use("end_contact", {"status": "resolved"}, text="Goodbye!", call_id="e2"),
         say("")],
        ["I want to cancel.", CALLER_ID, "Yes."])
    assert run.go(client) == 0
    [conv] = run.rows("conversations.jsonl")
    assert [t["text"] for t in conv["turns"]] == [
        "Hello, Marisa here.", "I want to cancel.",
        "Could you tell me your full name and address?", CALLER_ID, "Is that all?", "Yes.",
        "Goodbye!"]
    [calls] = run.rows("tool_calls.jsonl")
    assert calls["stop_reason"] == "end_contact" and calls["end_status"] == "resolved"
    assert [(c["after_turn"], c["name"], c["result"]) for c in calls["calls"]] == [
        (2, "verify_identity", {"result": "not_checked", "reason": "not_said_by_customer",
                                "missing": ["full_name", "service_address", "last_payment_amount"],
                                "failed_attempts": 0}),
        (4, "verify_identity", {"result": "verified_account_holder", "failed_attempts": 0}),
        (4, "end_contact", BLOCKED),
        (6, "end_contact", {"ended": True, "status": "resolved"})]
    # the blocked end_contact never reached the model as a message
    last_turn_messages = client.agent_calls[-2]["messages"]
    assert not any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "e1"
                   for m in last_turn_messages)
    assert {"role": "assistant", "content": "Is that all?"} in last_turn_messages


def test_no_guards_reproduces_the_old_behaviour(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    client = FakeClient(
        [say("Hello, Marisa here."),
         use("verify_identity", VERIFY, call_id="v1"),
         use("end_contact", {"status": "no_change"}, text="Is that all?", call_id="e1"),
         say("")],
        ["I want to cancel."])
    assert run.go(client, "--no-guards") == 0
    [calls] = run.rows("tool_calls.jsonl")
    assert calls["stop_reason"] == "end_contact" and calls["end_status"] == "no_change"
    assert [(c["after_turn"], c["name"], c["result"]) for c in calls["calls"]] == [
        (2, "verify_identity", {"result": "verified_account_holder", "failed_attempts": 0}),
        (2, "end_contact", {"ended": True, "status": "no_change"})]
    [conv] = run.rows("conversations.jsonl")
    assert conv["turns"][-1] == {"speaker": "agent", "text": "Is that all?"}
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["guards"] is False


def test_run_json_records_guards_and_the_agent_prompt_as_given(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    (run.root / "prompts" / "alt.md").write_text("ALT {channel} {today} {agent_name} {policy}",
                                                 encoding="utf-8")
    client = quick()
    run.go(client, "--agent-prompt", "prompts/alt.md")
    meta = json.loads((run.out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["agent_prompt"] == "prompts/alt.md" and meta["guards"] is True
    assert client.agent_calls[0]["messages"][0]["content"].startswith("ALT phone 2026-10-05 Marisa")


def test_missing_agent_prompt_file_exits(tmp_path):
    run = Run(tmp_path, ids=("ag-01",))
    with pytest.raises(SystemExit) as err:
        run.go(quick(), "--agent-prompt", "prompts/nope.md")
    assert "required file not found" in str(err.value)


@pytest.mark.parametrize("first,second", [
    ((), ("--no-guards",)),
    (("--no-guards",), ()),
    ((), ("--agent-prompt", "prompts/same.md")),     # same content, another path as given
])
def test_run_json_mismatch_covers_guards_and_the_agent_prompt(tmp_path, first, second):
    run = Run(tmp_path, ids=("ag-01", "ag-02"))
    (run.root / "prompts" / "same.md").write_text(AGENT_TEMPLATE, encoding="utf-8")
    run.go(quick(), "--limit", "1", *first)
    client = quick()
    with pytest.raises(SystemExit) as err:
        run.go(client, *second)
    assert str(err.value) == MISMATCH and client.agent_calls == []


def test_a_run_json_from_before_the_guards_is_refused(tmp_path):
    run = Run(tmp_path, ids=("ag-01", "ag-02"))
    run.go(quick(), "--limit", "1")
    path = run.out_dir / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["guards"], meta["agent_prompt"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        run.go(quick())
    assert str(err.value) == MISMATCH


# ---------------------------------------------------------------- load_agent_settings

def write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_agent_settings_fall_back_to_the_auditor_values(tmp_path):
    env = write(tmp_path, "AUDITOR_BASE_URL=https://a/v1\nAUDITOR_API_KEY=sk-a\n"
                          "AGENT_MODEL=am\nCUSTOMER_MODEL=cm\nAGENT_BASE_URL=\n")
    assert config.load_agent_settings(env) == ("https://a/v1", "sk-a", "am", "cm")


def test_agent_settings_prefer_the_agent_values(tmp_path):
    env = write(tmp_path, "AUDITOR_BASE_URL=https://a/v1\nAUDITOR_API_KEY=sk-a\n"
                          "AGENT_BASE_URL=https://g/v1\nAGENT_API_KEY=sk-g\n"
                          "AGENT_MODEL=am\nCUSTOMER_MODEL=cm\n")
    assert config.load_agent_settings(env) == ("https://g/v1", "sk-g", "am", "cm")


def test_agent_settings_work_without_any_auditor_values_when_agent_ones_are_set(tmp_path):
    env = write(tmp_path, "AGENT_BASE_URL=https://g/v1\nAGENT_API_KEY=sk-g\n"
                          "AGENT_MODEL=am\nCUSTOMER_MODEL=cm\n")
    assert config.load_agent_settings(env)[:2] == ("https://g/v1", "sk-g")


def test_agent_settings_name_what_is_missing_and_never_a_value(tmp_path):
    env = write(tmp_path, "AGENT_MODEL=secret-model\n")
    with pytest.raises(config.ConfigError) as err:
        config.load_agent_settings(env)
    assert str(err.value) == ("missing in .env: CUSTOMER_MODEL, AGENT_BASE_URL (or AUDITOR_BASE_URL), "
                              "AGENT_API_KEY (or AUDITOR_API_KEY)")
    assert "secret-model" not in str(err.value)


def test_agent_settings_missing_file(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load_agent_settings(tmp_path / "nope.env")
