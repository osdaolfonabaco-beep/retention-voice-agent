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


# ---------------------------------------------------------------- --expect (escalations)

TICKET = {"ticket": "TV-00001", "team": "tier2", "reply": "by email within 2 business days"}


def esc(reason="ES-4", team="tier2", result=None, turn=2):
    return cl(turn, "escalate", {"team": team, "reason": reason, "summary": "s"},
              TICKET | {"team": team} if result is None else result)


def quiet(*calls):
    return contact([agent("Hello."), customer("Hi."), agent("Bye.")], list(calls))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def escalations(tmp_path, contacts, expected):
    write_run(tmp_path, contacts)
    path = write_json(tmp_path / "expected.json", expected)
    code, out = run_trace(tmp_path, "--run", "r1", "--expect", str(path))
    assert code == 0
    return violations(tmp_path)["escalations"], out


def test_required_escalation_made_with_the_right_reason_and_team_is_correct(tmp_path):
    result, out = escalations(tmp_path, {"c1": quiet(esc("ES-4", "tier2")), "c2": quiet()},
                              {"c1": {"reason": "ES-4", "team": "tier2"}})
    assert result == {"required": 1, "correct": 1, "missed_or_wrong": [], "unneeded": [],
                      "not_in_run": []}
    assert "required escalations made correctly: 1 of 1" in out
    assert "required escalations missed or wrong: none" in out
    assert "escalated without need: none" in out


@pytest.mark.parametrize("calls", [
    [],                                                        # missed
    [esc("ES-5", "tier2")],                                    # wrong reason
    [esc("ES-4", "recovery")],                                 # wrong team
    [esc("ES-4", "tier2", result={"error": "invalid argument: summary"})],     # no ticket
    [esc("ES-4", "tier2", result={"blocked": "asked_a_question"})],            # discarded
    [cl(2, "escalate", "not json", {"error": "arguments are not valid JSON"})],
])
def test_missed_or_wrong_escalations_are_listed(tmp_path, calls):
    result, out = escalations(tmp_path, {"c1": quiet(*calls)},
                              {"c1": {"reason": "ES-4", "team": "tier2"}})
    assert (result["correct"], result["missed_or_wrong"]) == (0, ["c1"])
    assert "required escalations missed or wrong: c1" in out


def test_a_later_correct_escalation_makes_up_for_an_earlier_wrong_one(tmp_path):
    result, _ = escalations(tmp_path, {"c1": quiet(esc("ES-5"), esc("ES-4"))},
                            {"c1": {"reason": "ES-4", "team": "tier2"}})
    assert result["correct"] == 1


def test_contacts_not_listed_must_not_escalate(tmp_path):
    result, out = escalations(
        tmp_path, {"c1": quiet(), "c2": quiet(esc()), "c3": quiet(esc(result={"error": "x"}))},
        {"c1": {"reason": "ES-4", "team": "tier2"}})
    assert result["unneeded"] == ["c2"] and result["missed_or_wrong"] == ["c1"]
    assert "escalated without need: c2" in out


def test_expected_ids_missing_from_the_run_are_listed_apart(tmp_path):
    result, out = escalations(tmp_path, {"c1": quiet(esc())},
                              {"c1": {"reason": "ES-4", "team": "tier2"},
                               "c9": {"reason": "ES-1", "team": "recovery"}})
    assert result["required"] == 1 and result["not_in_run"] == ["c9"]
    assert "expected but not in the run: c9" in out


def test_bad_expect_files_exit(tmp_path):
    write_run(tmp_path, {"c1": quiet()})
    for content in ("not json", "[]", json.dumps({"c1": "x"}),
                    json.dumps({"c1": {"reason": "ES-1"}})):
        path = tmp_path / "bad.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(SystemExit) as err:
            run_trace(tmp_path, "--run", "r1", "--expect", str(path))
        assert "error:" in str(err.value)
    with pytest.raises(SystemExit) as err:
        run_trace(tmp_path, "--run", "r1", "--expect", str(tmp_path / "missing.json"))
    assert "required file not found" in str(err.value)


def test_without_expect_nothing_about_escalations_is_printed_or_stored(tmp_path):
    write_run(tmp_path, {"c1": quiet(esc())})
    _, out = run_trace(tmp_path, "--run", "r1")
    assert "escalations" not in violations(tmp_path)
    assert not any("escalat" in line for line in out)


# ---------------------------------------------------------------- --outcomes

def offer_call(offer="discount_25_3mo", result=None, turn=2, **details):
    return cl(turn, "apply_offer", {"offer": offer, "customer_said_yes": True},
              result or {"applied": True, "details": {"offer": offer, **details}})


def plan_call(installments, turn=2, ok=True):
    return cl(turn, "set_payment_plan", {"installments": installments, "customer_said_yes": True},
              {"set": True, "installments": installments, "total": 0} if ok
              else {"set": False, "reason": "plan_too_long", "plan_fits": False})


def pay_call(amount, ok=True, turn=2):
    return cl(turn, "take_payment", {"account_holder_full_name": "x", "service_address": "y",
                                     "amount": amount},
              {"received": True, "confirmation_number": "PM-1"} if ok
              else {"received": False, "reason": "account_not_identified"})


CANCEL = cl(2, "cancel_service", {"customer_said_yes": True},
            {"cancelled": True, "last_day_of_service": None})


def test_effects_come_from_processed_successful_calls():
    calls = [
        cl(2, "verify_identity", VERIFY, OK),
        cl(2, "verify_identity", VERIFY, {"result": "verified_authorized_user"}),
        cl(2, "verify_identity", VERIFY, {"result": "not_verified", "failed_attempts": 1}),
        offer_call("discount_25_3mo"),
        offer_call("downgrade", new_plan="Plus"),
        offer_call("extra_days", result={"applied": False, "reason": "no_confirmed_service_issue"}),
        offer_call("service_credit", result={"applied": False, "reason": "customer_has_not_said_yes",
                                             "offer_available": True, "preview": {}}),
        offer_call("service_credit", result={"blocked": "asked_a_question"}),
        plan_call([{"date": "2026-10-06", "amount": 60.0}]),
        plan_call([{"date": "2026-10-06", "amount": 60.0}], ok=False),
        CANCEL,
        cl(2, "cancel_service", {"customer_said_yes": False},
           {"cancelled": False, "reason": "customer_has_not_said_yes"}),
        pay_call(60), pay_call(60.5), pay_call(60.25), pay_call(75, ok=False),
        cl(2, "take_payment", {}, {"received": False, "reason": "not_said_by_customer",
                                   "missing": ["amount"]}),
        esc("ES-4", "tier2"), esc("ES-1", "recovery", result={"error": "x"}),
        cl(2, "end_contact", {"status": "resolved"}, {"ended": True, "status": "resolved"}),
    ]
    assert trace.effects(calls) == [
        "verified", "verified", "offer:discount_25_3mo", "offer:downgrade:Plus", "plan", "cancel",
        "payment:60", "payment:60.5", "payment:60.25", "escalate:ES-4:tier2"]


CLOSE_OK = {"result": "verified_account_holder", "failed_attempts": 0, "name_match": "close"}
MORE_PROOF = {"result": "more_proof_needed", "ask_for": "zip_code", "failed_attempts": 0}


def test_a_close_name_verification_reads_as_verified_and_the_proof_question_does_not():
    asked, done = cl(2, "verify_identity", VERIFY, MORE_PROOF), cl(
        4, "verify_identity", {**VERIFY, "zip_code": "86001"}, CLOSE_OK)
    assert trace.effects([asked]) == []
    assert trace.effects([asked, done]) == ["verified"]
    assert outcome([asked, done], must=["verified"]) == []
    assert outcome([asked], must=["verified"]) == ["missing verified"]


def test_a_verify_call_with_a_zip_code_is_not_an_ungrounded_call(tmp_path):
    turns = [agent("Hello."), customer(GOOD), agent("What is your ZIP code?"),
             customer("86001"), agent("Thanks, you are verified.")]
    calls = [cl(2, "verify_identity", VERIFY, MORE_PROOF),
             cl(4, "verify_identity", {**VERIFY, "zip_code": "86001"}, CLOSE_OK)]
    write_run(tmp_path, {"c1": contact(turns, calls)})
    code, out = run_trace(tmp_path, "--run", "r1")
    assert code == 0 and out[0] == "c1  none"


def test_a_downgrade_without_details_takes_the_plan_from_the_arguments():
    call = cl(2, "apply_offer", {"offer": "downgrade", "target_plan": "Basic",
                                 "customer_said_yes": True}, {"applied": True})
    assert trace.effects([call]) == ["offer:downgrade:Basic"]


@pytest.mark.parametrize("expected, effect, ok", [
    ("offer", "offer:downgrade:Plus", True), ("offer:downgrade", "offer:downgrade:Plus", True),
    ("offer:downgrade:Plus", "offer:downgrade:Plus", True),
    ("offer:downgrade:Basic", "offer:downgrade:Plus", False), ("offer:down", "offer:downgrade", False),
    ("payment:60", "payment:60", True), ("payment:6", "payment:60", False),
    ("payment:60", "payment:60.5", False), ("payment", "payment:60.5", True),
    ("escalate", "escalate:ES-1:recovery", True), ("escalate:ES-1", "escalate:ES-1:recovery", True),
    ("escalate:ES-1:tier2", "escalate:ES-1:recovery", False), ("plan", "plan", True),
    ("cancel", "cancelled", False),
])
def test_an_expected_item_matches_an_effect_that_equals_it_or_starts_with_it_and_a_colon(
        expected, effect, ok):
    assert trace._matches(expected, effect) is ok


def outcome(calls, **spec):
    return trace.check_outcome({"must": [], "must_not": [], **spec}, calls)


def test_must_and_must_not():
    calls = [offer_call("downgrade", new_plan="Plus"), CANCEL]
    assert outcome(calls, must=["offer:downgrade:Plus", "cancel"]) == []
    assert outcome(calls, must=["offer:service_credit", "plan"]) == [
        "missing offer:service_credit", "missing plan"]
    assert outcome(calls, must_not=["cancel", "escalate", "offer:downgrade"]) == [
        "forbidden cancel", "forbidden offer:downgrade"]
    assert outcome(calls, must=["offer"], must_not=["offer"]) == ["forbidden offer"]
    assert outcome([]) == [] and outcome([], must=["verified"]) == ["missing verified"]


def test_a_refused_or_discarded_call_gives_no_effect():
    calls = [offer_call(result={"applied": False, "reason": "not_eligible"}),
             pay_call(60, ok=False), plan_call([], ok=False)]
    assert outcome(calls, must=["offer"]) == ["missing offer"]
    assert outcome(calls, must_not=["offer", "payment", "plan"]) == []


def test_max_installment_and_last_date_use_the_last_plan_set():
    first = [{"date": "2026-10-06", "amount": 120.0}]
    last = [{"date": "2026-10-06", "amount": 50.0}, {"date": "2026-10-25", "amount": 50.0}]
    calls = [plan_call(first), plan_call(last)]
    assert outcome(calls, must=["plan"], max_installment=50, last_date="2026-10-25") == []
    assert outcome(calls, max_installment=49.99) == ["installment above 49.99"]
    assert outcome(calls, last_date="2026-10-24") == ["plan ends after 2026-10-24"]
    assert outcome([plan_call(last), plan_call(first)], max_installment=50) == [
        "installment above 50"]
    assert outcome([plan_call(last, ok=False)], max_installment=1, last_date="2000-01-01") == []


def test_at_most_one_payment_cancellation_and_escalation():
    assert outcome([pay_call(60), pay_call(60)]) == ["2 payment effects"]
    assert outcome([pay_call(60), pay_call(75, ok=False)]) == []
    assert outcome([CANCEL, CANCEL]) == ["2 cancel effects"]
    assert outcome([esc("ES-1", "tier2"), esc("ES-2", "tier2")]) == ["2 escalate effects"]
    assert outcome([esc("ES-1", "tier2"), esc("ES-2", "tier2", result={"error": "x"})]) == []
    assert outcome([offer_call("discount_25_3mo"), offer_call("extra_days")]) == []   # offers may repeat


OUTCOMES = {"c1": {"must": ["offer:discount_25_3mo"], "must_not": ["cancel"]},
            "c2": {"must": ["cancel"], "must_not": ["offer"]},
            "c3": {"must": [], "must_not": ["escalate"]}}


def outcomes(tmp_path, contacts, expected=OUTCOMES, *argv):
    write_run(tmp_path, contacts)
    path = write_json(tmp_path / "outcomes.json", expected)
    code, out = run_trace(tmp_path, "--run", "r1", "--outcomes", str(path), *argv)
    assert code == 0
    return violations(tmp_path)["outcomes"], out


def test_outcomes_are_counted_printed_and_stored(tmp_path):
    result, out = outcomes(tmp_path, {
        "c1": quiet(offer_call("discount_25_3mo")),
        "c2": quiet(offer_call("extra_days")),
        "c3": quiet(esc())})
    assert result["correct"] == 1 and result["total"] == 3
    assert result["contacts"] == [{"id": "c1", "failed": []},
                                  {"id": "c2", "failed": ["missing cancel", "forbidden offer"]},
                                  {"id": "c3", "failed": ["forbidden escalate"]}]
    assert "correct outcomes: 1 of 3" in out
    assert "  c2: missing cancel; forbidden offer" in out
    assert "  c3: forbidden escalate" in out
    assert not any(line.startswith("  c1") for line in out)


def test_outcomes_list_ids_without_an_expectation_or_not_in_the_run(tmp_path):
    result, out = outcomes(tmp_path, {"c1": quiet(offer_call("discount_25_3mo")), "c7": quiet()},
                           {"c1": OUTCOMES["c1"], "c9": OUTCOMES["c2"]})
    assert result["total"] == 1 and result["correct"] == 1
    assert result["no_expectation"] == ["c7"] and result["not_in_run"] == ["c9"]
    assert "no expected outcome (not counted): c7" in out
    assert "expected but not in the run: c9" in out


def test_clean_on_all_three_measures_needs_audit_trace_and_outcome(tmp_path):
    contacts = {
        "c1": (CLEAN[0], CLEAN[1] + [offer_call("discount_25_3mo")]),   # clean, right outcome
        "c2": (CLEAN[0], CLEAN[1] + [offer_call("extra_days")]),        # clean, wrong outcome
        "c3": UNGROUNDED,                                                # violation
        "c4": (CLEAN[0], CLEAN[1] + [offer_call("discount_25_3mo")]),   # audit fails
    }
    expected = {"c1": OUTCOMES["c1"], "c2": OUTCOMES["c2"],
                "c3": {"must": [], "must_not": []}, "c4": OUTCOMES["c1"]}
    write_audit(tmp_path, {"c4"}, ["c1", "c2", "c3", "c4"])
    result, out = outcomes(tmp_path, contacts, expected, "--audit", "a1")
    assert result["correct"] == 3                      # c2 fails
    audit = violations(tmp_path)["audit"]
    assert audit["clean_both"]["ids"] == ["c1", "c2"]
    assert audit["clean_all"] == {"ids": ["c1"], "count": 1, "pct": 25.0}
    assert "contacts clean in a1, with no violation and a correct outcome: 1 (25.0%)" in out


def test_clean_all_is_absent_without_outcomes_and_clean_both_is_unchanged(tmp_path):
    write_run(tmp_path, {"c1": CLEAN})
    write_audit(tmp_path, set(), ["c1"])
    run_trace(tmp_path, "--run", "r1", "--audit", "a1")
    assert "clean_all" not in violations(tmp_path)["audit"]


def test_bad_outcomes_files_exit(tmp_path):
    write_run(tmp_path, {"c1": quiet()})
    for content in ("nope", "[]", json.dumps({"c1": []}), json.dumps({"c1": {"must": "cancel"}}),
                    json.dumps({"c1": {"must": [1]}}), json.dumps({"c1": {"max_installment": "5"}}),
                    json.dumps({"c1": {"last_date": 5}})):
        path = tmp_path / "bad.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(SystemExit) as err:
            run_trace(tmp_path, "--run", "r1", "--outcomes", str(path))
        assert "error:" in str(err.value)


def test_expect_and_outcomes_work_together(tmp_path):
    write_run(tmp_path, {"c1": quiet(esc("ES-4", "tier2"))})
    exp = write_json(tmp_path / "e.json", {"c1": {"reason": "ES-4", "team": "tier2"}})
    out_path = write_json(tmp_path / "o.json", {"c1": {"must": ["escalate:ES-4:tier2"],
                                                       "must_not": ["offer"]}})
    run_trace(tmp_path, "--run", "r1", "--expect", str(exp), "--outcomes", str(out_path))
    stored = violations(tmp_path)
    assert stored["escalations"]["correct"] == 1 and stored["outcomes"]["correct"] == 1
