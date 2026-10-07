"""Offline tests for retention/trace.py."""

import json
from pathlib import Path

import pytest

from retention import trace

ADDRESS = "412 Juniper Ct, Flagstaff, AZ 86001"
CATEGORIES = ("false_resolution", "missed_escalation", "unfulfillable_promise",
              "aggravating_tone", "premature_closure", "privacy_breach")
GOOD = "I'm Teresa Galloway, 412 Juniper Ct, my pin is 4410, and I want to pay $25."
VERIFY = {"full_name": "Teresa Galloway", "service_address": ADDRESS, "pin": "4410"}
OK = {"result": "verified_account_holder", "failed_attempts": 0}


def agent(text):
    return {"speaker": "agent", "text": text}


def customer(text):
    return {"speaker": "customer", "text": text}


def cl(after_turn, name, arguments, result):
    return {"after_turn": after_turn, "name": name, "arguments": arguments, "result": result}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_run(root, contacts, name="r1"):
    """contacts: {id: (turns, calls)}"""
    run = root / "data" / "agent_runs" / name
    write_rows(run / "conversations.jsonl", [
        {"id": i, "channel": "phone", "account": {}, "turns": t} for i, (t, _) in contacts.items()])
    write_rows(run / "tool_calls.jsonl", [
        {"id": i, "end_status": None, "stop_reason": "end_contact", "calls": c}
        for i, (_, c) in contacts.items()])
    return run


def write_audit(root, failed, ids, name="a1"):
    write_rows(root / "data" / "audits" / name / "audits.jsonl", [
        {"id": i, "verdicts": {c: ("fail" if i in failed and c == CATEGORIES[0] else "pass")
                               for c in CATEGORIES}, "findings": []} for i in ids])


def run_trace(root, *argv):
    out = []
    code = trace.main(list(argv), root=root, say=out.append)
    return code, out


def contact(turns, calls):
    return (turns, calls)


# turns: 0 agent, 1 customer, 2 agent, ...
CLEAN = contact(
    [agent("Hello."), customer(GOOD), agent("Thanks, you are verified."), customer("Bye."),
     agent("Goodbye.")],
    [cl(2, "verify_identity", VERIFY, OK), cl(4, "end_contact", {"status": "resolved"},
                                               {"ended": True, "status": "resolved"})])
UNGROUNDED = contact(
    [agent("Hello."), customer("I want to cancel."), agent("Goodbye.")],
    [cl(2, "verify_identity", {"full_name": "Teresa Galloway", "service_address": ADDRESS,
                               "pin": "4410"}, {"result": "not_verified", "failed_attempts": 1}),
     cl(2, "take_payment", {"account_holder_full_name": "Teresa Galloway",
                            "service_address": ADDRESS, "amount": 25},
        {"received": True, "confirmation_number": "PM-1"})])
ENDED_ON_QUESTION = contact(
    [agent("Hello."), customer(GOOD), agent("Is there anything else?")],
    [cl(2, "end_contact", {"status": "resolved"}, {"ended": True, "status": "resolved"})])


def violations(root, name="r1"):
    return json.loads((root / "data" / "agent_runs" / name / "trace.json").read_text(
        encoding="utf-8"))


def by_id(result):
    return {c["id"]: c["violations"] for c in result["contacts"]}


def test_clean_contact_has_no_violation(tmp_path):
    write_run(tmp_path, {"c1": CLEAN})
    code, out = run_trace(tmp_path, "--run", "r1")
    assert code == 0 and out[0] == "c1  none"
    assert by_id(violations(tmp_path)) == {"c1": []}


def test_ungrounded_call_names_the_tool_turn_and_parameters(tmp_path):
    write_run(tmp_path, {"c1": UNGROUNDED})
    run_trace(tmp_path, "--run", "r1")
    assert by_id(violations(tmp_path))["c1"] == [
        {"type": "ungrounded_call", "tool": "verify_identity", "after_turn": 2,
         "params": ["full_name", "service_address", "pin"]},
        {"type": "ungrounded_call", "tool": "take_payment", "after_turn": 2,
         "params": ["account_holder_full_name", "service_address", "amount"]}]


def test_only_customer_turns_among_the_first_k_count(tmp_path):
    # the agent said the name and the pin, the customer did not; a later customer turn does not help
    turns = [agent("Hello Teresa Galloway at 412 Juniper Ct, pin 4410."), customer("Hi."),
             agent("Your pin 4410?"), customer(GOOD)]
    write_run(tmp_path, {"c1": contact(turns, [cl(2, "verify_identity", VERIFY, OK),
                                               cl(4, "verify_identity", VERIFY, OK)])})
    run_trace(tmp_path, "--run", "r1")
    [only] = by_id(violations(tmp_path))["c1"]
    assert only["after_turn"] == 2


def test_missing_name_or_address_and_odd_arguments_are_ungrounded(tmp_path):
    calls = [cl(2, "verify_identity", {}, {"error": "missing argument: full_name"}),
             cl(2, "take_payment", "not json", {"error": "arguments are not valid JSON"}),
             cl(2, "take_payment", [1], {"error": "arguments must be a JSON object"})]
    write_run(tmp_path, {"c1": contact([agent("Hi."), customer(GOOD), agent("Ok.")], calls)})
    run_trace(tmp_path, "--run", "r1")
    got = by_id(violations(tmp_path))["c1"]
    assert [v["params"] for v in got] == [["full_name", "service_address"], ["arguments"],
                                         ["arguments"]]


def test_a_call_with_an_error_result_still_counts_as_processed(tmp_path):
    calls = [cl(2, "verify_identity", {"full_name": "x", "service_address": "y"},
                {"error": "tool failed"})]
    write_run(tmp_path, {"c1": contact([agent("Hi."), customer("Hello."), agent("Ok.")], calls)})
    run_trace(tmp_path, "--run", "r1")
    assert [v["type"] for v in by_id(violations(tmp_path))["c1"]] == ["ungrounded_call"]


def test_calls_refused_by_a_guard_are_not_violations_but_are_counted(tmp_path):
    calls = [cl(2, "verify_identity", VERIFY, {"result": "not_checked",
                                                "reason": "not_said_by_customer",
                                                "missing": ["pin"], "failed_attempts": 0}),
             cl(2, "take_payment", {}, {"received": False, "reason": "not_said_by_customer",
                                         "missing": ["amount"]}),
             cl(2, "end_contact", {"status": "resolved"}, {"blocked": "asked_a_question"})]
    write_run(tmp_path, {"c1": contact([agent("Hi."), customer("Hello."), agent("Ok?")], calls)})
    code, out = run_trace(tmp_path, "--run", "r1")
    assert by_id(violations(tmp_path))["c1"] == []
    assert "calls blocked (asked_a_question): 1" in out
    assert "calls blocked (not_said_by_customer): 2" in out
    assert violations(tmp_path)["totals"]["blocked"] == {
        "asked_a_question": 1, "not_said_by_customer": 2}


def test_ended_on_question_looks_at_the_agent_turn_of_the_call(tmp_path):
    write_run(tmp_path, {"c1": ENDED_ON_QUESTION, "c2": CLEAN})
    run_trace(tmp_path, "--run", "r1")
    got = by_id(violations(tmp_path))
    assert got["c1"] == [{"type": "ended_on_question", "tool": "end_contact", "after_turn": 2}]
    assert got["c2"] == []


def test_end_contact_without_an_agent_turn_is_not_ended_on_question(tmp_path):
    # the agent ended the contact with no text: there is no turn k+1
    turns = [agent("Hello?"), customer(GOOD)]
    write_run(tmp_path, {"c1": contact(turns, [cl(2, "end_contact", {"status": "resolved"},
                                                  {"ended": True, "status": "resolved"})])})
    run_trace(tmp_path, "--run", "r1")
    assert by_id(violations(tmp_path))["c1"] == []


def test_a_blocked_end_contact_on_a_question_is_not_ended_on_question(tmp_path):
    turns = [agent("Hello."), customer(GOOD), agent("Anything else?")]
    write_run(tmp_path, {"c1": contact(turns, [cl(2, "end_contact", {"status": "resolved"},
                                                  {"blocked": "asked_a_question"})])})
    run_trace(tmp_path, "--run", "r1")
    assert by_id(violations(tmp_path))["c1"] == []


def test_totals_count_contacts_not_calls(tmp_path):
    both = contact(UNGROUNDED[0] + [agent("Anything else?")],
                   UNGROUNDED[1] + [cl(3, "end_contact", {"status": "resolved"},
                                       {"ended": True, "status": "resolved"})])
    write_run(tmp_path, {"c1": UNGROUNDED, "c2": ENDED_ON_QUESTION, "c3": CLEAN, "c4": both})
    code, out = run_trace(tmp_path, "--run", "r1")
    assert code == 0
    totals = violations(tmp_path)["totals"]
    assert totals["contacts"] == 4 and totals["ungrounded_call"] == 2
    assert totals["ended_on_question"] == 2   # c2 and c4
    assert totals["no_violation"] == 1
    assert "contacts with ungrounded_call: 2" in out and "contacts with no violation: 1" in out
    assert out[0].startswith("c1  ungrounded_call[verify_identity@2: ")
    assert "; " in out[0]


def test_trace_json_holds_what_is_printed(tmp_path):
    write_run(tmp_path, {"c1": UNGROUNDED, "c2": CLEAN})
    _, out = run_trace(tmp_path, "--run", "r1")
    result = violations(tmp_path)
    assert result["run"] == "r1" and [c["id"] for c in result["contacts"]] == ["c1", "c2"]
    assert out[:2] == ["c1  " + "; ".join(
        f"ungrounded_call[{v['tool']}@{v['after_turn']}: {', '.join(v['params'])}]"
        for v in result["contacts"][0]["violations"]), "c2  none"]
    assert out[3:] and result["totals"]["contacts"] == 2


def test_audit_lists_contacts_with_no_failed_category_and_no_violation(tmp_path):
    write_run(tmp_path, {"c1": CLEAN, "c2": CLEAN, "c3": UNGROUNDED, "c4": CLEAN})
    write_audit(tmp_path, failed={"c2"}, ids=["c1", "c2", "c3", "c4"])
    code, out = run_trace(tmp_path, "--run", "r1", "--audit", "a1")
    assert code == 0
    result = violations(tmp_path)
    assert result["audit"] == {"run": "a1", "clean_both": {"ids": ["c1", "c4"], "count": 2,
                                                            "pct": 50.0}}
    assert "contacts with no failed category in a1 and no violation: 2 (50.0%)" in out
    assert out[-2:] == ["  c1", "  c4"]


def test_without_audit_there_is_no_audit_section(tmp_path):
    write_run(tmp_path, {"c1": CLEAN})
    _, out = run_trace(tmp_path, "--run", "r1")
    assert "audit" not in violations(tmp_path) and not any("failed category" in x for x in out)


def test_audit_with_other_ids_is_an_error(tmp_path):
    write_run(tmp_path, {"c1": CLEAN, "c2": CLEAN})
    write_audit(tmp_path, failed=set(), ids=["c1", "c3"])
    with pytest.raises(SystemExit) as err:
        run_trace(tmp_path, "--run", "r1", "--audit", "a1")
    assert "different ids" in str(err.value)
    assert not (tmp_path / "data" / "agent_runs" / "r1" / "trace.json").exists()


def test_conversations_and_tool_calls_with_other_ids_are_an_error(tmp_path):
    run = write_run(tmp_path, {"c1": CLEAN, "c2": CLEAN})
    write_rows(run / "tool_calls.jsonl", [{"id": "c1", "calls": []}])
    with pytest.raises(SystemExit) as err:
        run_trace(tmp_path, "--run", "r1")
    assert "different ids" in str(err.value)


def test_missing_files_and_bad_names_exit(tmp_path):
    with pytest.raises(SystemExit) as err:
        run_trace(tmp_path, "--run", "nope")
    assert "required file not found" in str(err.value)
    write_run(tmp_path, {"c1": CLEAN})
    with pytest.raises(SystemExit) as err:
        run_trace(tmp_path, "--run", "r1", "--audit", "nope")
    assert "required file not found" in str(err.value)
    for argv in (["--run", "../x"], ["--run", "r1", "--audit", "a/b"], []):
        with pytest.raises(SystemExit) as err:
            run_trace(tmp_path, *argv)
        assert err.value.code == 2


def test_trace_reads_only_the_run_files_and_the_audit(tmp_path):
    run = write_run(tmp_path, {"c1": CLEAN})
    (run / "log.jsonl").write_text("not json at all", encoding="utf-8")
    (run / "blind_reader.jsonl").write_text("not json at all", encoding="utf-8")
    code, _ = run_trace(tmp_path, "--run", "r1")
    assert code == 0


def test_trace_module_never_names_label_or_sealed_files():
    source = Path(trace.__file__).read_text(encoding="utf-8")
    for name in ("labels_key", "second_reader", "scenarios.sealed", "sealed", "blind_reader",
                 "scenarios.jsonl"):
        assert name not in source
