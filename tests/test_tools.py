"""Offline tests for retention/tools.py."""

import json
import re
from pathlib import Path

import pytest

from retention import tools
from retention.tools import TOOL_SPECS, AccountSession

ADDRESS = "412 Juniper Ct, Flagstaff, AZ 86001"


def make_account(**over):
    account = {
        "today": "2026-10-05", "today_weekday": "Monday",
        "account_holder": "Teresa Galloway", "authorized_users": ["Lena Hoffman"],
        "service_address": ADDRESS,
        "on_file_for_verification": {"last_payment_amount": 60.0, "pin": "4410"},
        "plan": "Plus", "monthly_price": 60, "months_active": 20, "stage": "Current",
        "days_past_due": 0, "past_due_balance": 0, "late_fees_included": 0,
        "next_bill": {"due_date": "2026-10-24", "amount": 60},
        "retention_discount_in_last_12_months": False, "retention_discount_eligible": True,
        "confirmed_service_issue": None,
        "payment_plans": {"kept": 0, "broken": 0, "last_broken": None},
    }
    account.update(over)
    return account


def call(session, name, **args):
    result = session.call(name, json.dumps(args))
    json.dumps(result)  # every result must be serializable
    return result


def verified(**over):
    session = AccountSession(make_account(**over), "sc-1", grounding=False)
    result = call(session, "verify_identity", full_name="Teresa Galloway",
                  service_address=ADDRESS, pin="4410")
    assert result["result"] == "verified_account_holder"
    return session


# ---------------------------------------------------------------- verify_identity

def test_verify_account_holder_with_pin_and_with_amount():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "verify_identity", full_name="Teresa Galloway",
                service_address=ADDRESS, pin="4410") == {
        "result": "verified_account_holder", "failed_attempts": 0}
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "verify_identity", full_name="Teresa Galloway",
                service_address=ADDRESS, last_payment_amount=60)["result"] == "verified_account_holder"


def test_verify_authorized_user_ignores_case_and_extra_spaces():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = call(s, "verify_identity", full_name="  lena   HOFFMAN ",
                  service_address=ADDRESS, pin="4410")
    assert result["result"] == "verified_authorized_user"


@pytest.mark.parametrize("address", [
    "412 Juniper Ct", "412 juniper ct., apt 2, Flagstaff", "I live at 412-Juniper CT",
])
def test_address_matches_street_part_in_letters_and_digits(address):
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "verify_identity", full_name="Teresa Galloway",
                service_address=address, pin="4410")["result"] == "verified_account_holder"


@pytest.mark.parametrize("args", [
    {"full_name": "Teresa Gallo", "service_address": ADDRESS, "pin": "4410"},
    {"full_name": "Teresa Galloway", "service_address": "500 Juniper Ct", "pin": "4410"},
    {"full_name": "Teresa Galloway", "service_address": ADDRESS, "pin": "0000"},
    {"full_name": "Teresa Galloway", "service_address": ADDRESS, "last_payment_amount": 59},
    {"full_name": "Someone Else", "service_address": ADDRESS, "pin": "4410"},
])
def test_each_wrong_item_fails_and_counts_one_attempt(args):
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = call(s, "verify_identity", **args)
    assert result == {"result": "not_verified", "failed_attempts": 1}


def test_call_without_third_item_does_not_verify_or_count():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = call(s, "verify_identity", full_name="Teresa Galloway", service_address=ADDRESS)
    assert result == {"result": "not_verified", "failed_attempts": 0}
    assert call(s, "verify_identity", full_name="Teresa Galloway",
                service_address=ADDRESS, pin="4410")["result"] == "verified_account_holder"


def test_locked_after_two_failed_attempts_even_with_correct_data():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    bad = dict(full_name="Teresa Galloway", service_address=ADDRESS, pin="0000")
    assert call(s, "verify_identity", **bad) == {"result": "not_verified", "failed_attempts": 1}
    assert call(s, "verify_identity", **bad) == {"result": "not_verified", "failed_attempts": 2}
    good = dict(full_name="Teresa Galloway", service_address=ADDRESS, pin="4410")
    assert call(s, "verify_identity", **good) == {"result": "locked", "failed_attempts": 2}
    assert call(s, "get_account") == {"error": "not_verified"}


def test_verified_session_stays_verified():
    s = verified()
    again = call(s, "verify_identity", full_name="Nobody", service_address="x", pin="0")
    assert again == {"result": "verified_account_holder", "failed_attempts": 0}


def test_verify_never_returns_on_file_values():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    out = json.dumps(call(s, "verify_identity", full_name="Teresa Galloway",
                          service_address=ADDRESS, pin="1234"))
    assert "4410" not in out and "Juniper" not in out and "60" not in out


# ---------------------------------------------------------------- get_account

def test_get_account_needs_verification():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "get_account") == {"error": "not_verified"}


def test_get_account_never_returns_on_file_for_verification():
    s = verified()
    account = call(s, "get_account")
    assert "on_file_for_verification" not in account
    assert account["plan"] == "Plus" and account["next_bill"]["due_date"] == "2026-10-24"
    assert "4410" not in json.dumps(account)
    assert "on_file_for_verification" in s.account  # the original is untouched


# ---------------------------------------------------------------- apply_offer

def offer(session, name="discount_25_3mo", yes=True, **extra):
    return call(session, "apply_offer", offer=name, customer_said_yes=yes, **extra)


def test_offer_requires_verification_and_yes():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert offer(s) == {"applied": False, "reason": "not_verified"}
    s = verified()
    assert offer(s, yes=False) == {"applied": False, "reason": "customer_has_not_said_yes"}
    assert s.applied_offers == set()


def test_discount_gives_three_bills_and_the_price_after():
    s = verified()
    result = offer(s)
    assert result["applied"] is True
    details = result["details"]
    assert details["discounted_price"] == 45.0 and details["price_after"] == 60
    assert details["discounted_bills"] == [
        {"date": "2026-10-24", "amount": 45.0}, {"date": "2026-11-24", "amount": 45.0},
        {"date": "2026-12-24", "amount": 45.0}]


def test_discount_dates_clamp_to_month_end():
    s = verified(next_bill={"due_date": "2026-12-31", "amount": 60})
    dates = [b["date"] for b in offer(s)["details"]["discounted_bills"]]
    assert dates == ["2026-12-31", "2027-01-31", "2027-02-28"]


def test_discount_needs_eligibility():
    s = verified(retention_discount_eligible=False)
    assert offer(s) == {"applied": False, "reason": "not_eligible"}


def test_each_offer_type_applies_once_per_session():
    s = verified()
    assert offer(s)["applied"] is True
    assert offer(s) == {"applied": False, "reason": "already_applied"}
    assert offer(s, "downgrade", target_plan="Basic")["applied"] is True


def test_downgrade():
    s = verified()
    result = offer(s, "downgrade", target_plan="Basic")
    assert result["details"] == {"offer": "downgrade", "new_plan": "Basic", "new_price": 45,
                                 "effective_date": "2026-10-24"}
    for target in ("Plus", "Giga"):
        s = verified()
        assert offer(s, "downgrade", target_plan=target) == {
            "applied": False, "reason": "target_plan_not_cheaper"}
    assert offer(verified(), "downgrade") == {"error": "missing argument: target_plan"}
    assert offer(verified(), "downgrade", target_plan="Mega") == {
        "error": "invalid argument: target_plan"}


def test_service_credit_limits():
    issue = "Outage Sep 23 to Sep 25"
    s = verified(confirmed_service_issue=issue, months_active=8)
    assert offer(s, "service_credit", amount=15)["details"] == {
        "offer": "service_credit", "credit_amount": 15.0,
        "applies_to_bill_date": "2026-10-24", "bill_amount": 60}
    s = verified(confirmed_service_issue=issue, months_active=8)
    assert offer(s, "service_credit", amount=15.5) == {
        "applied": False, "reason": "amount_not_allowed", "max_amount": 15}
    s = verified(confirmed_service_issue=issue, months_active=12)
    assert offer(s, "service_credit", amount=30)["applied"] is True
    s = verified(confirmed_service_issue=issue, months_active=12)
    assert offer(s, "service_credit", amount=30.01)["applied"] is False
    assert offer(verified(confirmed_service_issue=issue), "service_credit", amount=0)[
        "reason"] == "amount_not_allowed"


def test_service_credit_and_extra_days_need_a_confirmed_issue():
    s = verified()
    assert offer(s, "service_credit", amount=10) == {
        "applied": False, "reason": "no_confirmed_service_issue"}
    assert offer(s, "extra_days", days=5) == {
        "applied": False, "reason": "no_confirmed_service_issue"}
    assert offer(s, "service_credit") == {"error": "missing argument: amount"}


def test_extra_days():
    issue = "Intermittent drops confirmed on Sep 29"
    s = verified(confirmed_service_issue=issue)
    details = offer(s, "extra_days", days=10)["details"]
    assert details["new_due_date"] == "2026-11-03" and details["late_fee"] == 0
    for days in (0, 11):
        s = verified(confirmed_service_issue=issue)
        assert offer(s, "extra_days", days=days) == {
            "applied": False, "reason": "days_not_allowed", "max_days": 10}
    assert offer(verified(confirmed_service_issue=issue), "extra_days", days=2.5) == {
        "error": "invalid argument: days"}


def test_offers_without_an_upcoming_bill_are_refused():
    s = verified(next_bill=None)
    assert offer(s) == {"applied": False, "reason": "no_upcoming_bill"}


def test_unknown_offer_and_non_boolean_yes_are_errors():
    s = verified()
    assert offer(s, "free_month") == {"error": "invalid argument: offer"}
    assert call(s, "apply_offer", offer="downgrade", customer_said_yes="yes") == {
        "error": "invalid argument: customer_said_yes"}


# ---------------------------------------------------------------- set_payment_plan

def early(**over):
    return verified(stage="Early", past_due_balance=120, days_past_due=12, **over)


def plan(session, installments, yes=True):
    return call(session, "set_payment_plan", installments=installments, customer_said_yes=yes)


def test_payment_plan_success_sorts_by_date():
    s = early()
    result = plan(s, [{"date": "2026-11-05", "amount": 60}, {"date": "2026-10-15", "amount": 60}])
    assert result == {"set": True, "total": 120.0, "installments": [
        {"date": "2026-10-15", "amount": 60.0}, {"date": "2026-11-05", "amount": 60.0}]}


def test_payment_plan_gates():
    s = AccountSession(make_account(stage="Early", past_due_balance=120), "sc-1", grounding=False)
    assert plan(s, [{"date": "2026-10-06", "amount": 120}]) == {"set": False, "reason": "not_verified"}
    s = early()
    assert plan(s, [{"date": "2026-10-06", "amount": 120}], yes=False) == {
        "set": False, "reason": "customer_has_not_said_yes"}
    for stage in ("Current", "Recovery"):
        s = verified(stage=stage, past_due_balance=120)
        assert plan(s, [{"date": "2026-10-06", "amount": 120}]) == {
            "set": False, "reason": "stage_not_eligible"}


@pytest.mark.parametrize("stage, first_ok, first_late, last_ok, last_late", [
    ("Early", "2026-10-15", "2026-10-16", "2027-01-05", "2027-01-06"),
    ("Mid", "2026-10-10", "2026-10-11", "2026-12-05", "2026-12-06"),
    ("Late", "2026-10-07", "2026-10-08", "2026-11-05", "2026-11-06"),
])
def test_payment_plan_limits_by_stage(stage, first_ok, first_late, last_ok, last_late):
    def attempt(first, last):
        s = verified(stage=stage, past_due_balance=100)
        return plan(s, [{"date": first, "amount": 50}, {"date": last, "amount": 50}])

    assert attempt(first_ok, last_ok)["set"] is True
    assert attempt(first_late, last_ok) == {"set": False, "reason": "first_payment_too_late"}
    if stage == "Late":  # the first date is already near the limit; test the last date alone
        s = verified(stage=stage, past_due_balance=100)
        assert plan(s, [{"date": "2026-10-06", "amount": 50}, {"date": last_late, "amount": 50}]) == {
            "set": False, "reason": "plan_too_long"}
    else:
        assert attempt(first_ok, last_late) == {"set": False, "reason": "plan_too_long"}


def test_payment_plan_amount_and_date_rules():
    s = early()
    assert plan(s, [{"date": "2026-10-04", "amount": 120}]) == {
        "set": False, "reason": "date_before_today"}
    assert plan(s, [{"date": "2026-10-06", "amount": 100}]) == {
        "set": False, "reason": "total_must_equal_past_due_balance"}
    assert plan(s, [{"date": "2026-10-06", "amount": 130}, {"date": "2026-10-08", "amount": -10}]) == {
        "set": False, "reason": "amounts_must_be_positive"}
    assert plan(s, [{"date": "2026-10-05", "amount": 120}])["set"] is True  # today is allowed


def test_payment_plan_malformed_input_is_an_error():
    s = early()
    assert plan(s, []) == {"error": "invalid argument: installments"}
    assert plan(s, [{"date": "10/06/2026", "amount": 120}]) == {"error": "invalid argument: date"}
    assert plan(s, [{"date": "2026-02-30", "amount": 120}]) == {"error": "invalid argument: date"}
    assert plan(s, [{"date": "2026-10-06"}]) == {"error": "missing argument: installments[].amount"}
    assert plan(s, [{"date": "2026-10-06", "amount": "120"}]) == {
        "error": "invalid argument: installments[].amount"}


# ---------------------------------------------------------------- cancel_service

def test_cancel_service():
    s = verified(past_due_balance=35.5)
    assert call(s, "cancel_service", customer_said_yes=True) == {
        "cancelled": True, "last_day_of_service": "2026-10-24",
        "balance_still_owed": 35.5, "confirmation_email_sent": True}
    assert call(s, "cancel_service", customer_said_yes=True) == {
        "cancelled": False, "reason": "already_cancelled"}


def test_cancel_refusals_and_no_next_bill():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "cancel_service", customer_said_yes=True) == {
        "cancelled": False, "reason": "not_verified"}
    s = verified()
    assert call(s, "cancel_service", customer_said_yes=False) == {
        "cancelled": False, "reason": "customer_has_not_said_yes"}
    s = verified(next_bill=None)
    assert call(s, "cancel_service", customer_said_yes=True)["last_day_of_service"] is None


# ---------------------------------------------------------------- take_payment

def pay(session, name="Teresa Galloway", address=ADDRESS, amount=25):
    return call(session, "take_payment", account_holder_full_name=name,
                service_address=address, amount=amount)


def test_take_payment_needs_no_verification_and_returns_no_balance():
    s = AccountSession(make_account(past_due_balance=80), "sc-1", grounding=False)
    result = pay(s, "  teresa GALLOWAY ", "412 Juniper Ct")
    assert result["received"] is True and re.fullmatch(r"PM-\d{5}", result["confirmation_number"])
    assert set(result) == {"received", "confirmation_number"}


def test_take_payment_code_is_derived_from_the_scenario_id():
    a, b = AccountSession(make_account(), "sc-1", grounding=False), AccountSession(make_account(), "sc-1", grounding=False)
    other = AccountSession(make_account(), "sc-2", grounding=False)
    first = pay(a)
    assert first == pay(b)
    assert first != pay(other)
    assert pay(a) != first  # a second payment in the same session gets its own number


@pytest.mark.parametrize("name, address", [
    ("Lena Hoffman", ADDRESS),                 # authorized user is not enough
    ("Teresa Galloway", "500 Juniper Ct"),
    ("Teresa", ADDRESS),
])
def test_take_payment_refuses_unidentified_accounts(name, address):
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert pay(s, name, address) == {"received": False, "reason": "account_not_identified"}


def test_take_payment_rejects_bad_amounts():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert pay(s, amount=0) == {"error": "invalid argument: amount"}
    assert pay(s, amount=-5) == {"error": "invalid argument: amount"}
    assert pay(s, amount="ten") == {"error": "invalid argument: amount"}


# ---------------------------------------------------------------- escalate and end_contact

def test_escalate():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = call(s, "escalate", team="tier2", reason="ES-2", summary="Duplicate charge on Sep 3.")
    assert re.fullmatch(r"TV-\d{5}", result["ticket"])
    assert result["team"] == "tier2" and result["reply"] == "by email within 2 business days"
    again = call(s, "escalate", team="recovery", reason="ES-1", summary="Over 90 days.")
    assert again["ticket"] != result["ticket"]
    same = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(same, "escalate", team="tier2", reason="ES-2", summary="x")["ticket"] == result["ticket"]


@pytest.mark.parametrize("args", [
    {"team": "tier3", "reason": "ES-2", "summary": "x"},
    {"team": "tier2", "reason": "ES-8", "summary": "x"},
    {"team": "tier2", "reason": "ES-2", "summary": "   "},
    {"team": "tier2", "reason": "ES-2"},
])
def test_escalate_refuses_bad_input(args):
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert "error" in call(s, "escalate", **args)


def test_end_contact():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert not s.ended and s.end_status is None
    assert call(s, "end_contact", status="bogus") == {"error": "invalid argument: status"}
    assert not s.ended
    assert call(s, "end_contact", status="escalated") == {"ended": True, "status": "escalated"}
    assert s.ended and s.end_status == "escalated"


# ---------------------------------------------------------------- call() itself

@pytest.mark.parametrize("name, raw", [
    ("verify_identity", "{not json"),
    ("verify_identity", "[1, 2]"),
    ("verify_identity", "null"),
    ("no_such_tool", "{}"),
    ("verify_identity", "{}"),
    ("verify_identity", '{"full_name": 5, "service_address": "x"}'),
    ("verify_identity", '{"full_name": "a", "service_address": "b", "color": 1}'),
    ("get_account", '{"extra": 1}'),
    ("end_contact", '{"status": ["resolved"]}'),
    ("take_payment", '{"account_holder_full_name": "a", "service_address": "b", "amount": 1e999}'),
])
def test_bad_arguments_return_an_error_and_never_raise(name, raw):
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = s.call(name, raw)
    assert list(result) == ["error"] and isinstance(result["error"], str)
    json.dumps(result)


def test_error_text_never_echoes_values():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = s.call("verify_identity", '{"full_name": 5, "service_address": "SECRET-ADDRESS"}')
    assert "SECRET-ADDRESS" not in json.dumps(result)


def test_empty_arguments_and_null_optionals_are_accepted():
    s = verified()
    assert "plan" in s.call("get_account", "")
    assert "plan" in s.call("get_account", "   ")
    s = AccountSession(make_account(), "sc-1", grounding=False)
    result = call(s, "verify_identity", full_name="Teresa Galloway", service_address=ADDRESS,
                  pin="4410", last_payment_amount=None)
    assert result["result"] == "verified_account_holder"


def test_every_call_is_recorded_with_its_turn():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    s.turn = 3
    s.call("get_account", "{}")
    s.turn = 5
    s.call("nope", "{bad")
    assert s.calls == [
        {"after_turn": 3, "name": "get_account", "arguments": {}, "result": {"error": "not_verified"}},
        {"after_turn": 5, "name": "nope", "arguments": "{bad",
         "result": {"error": "arguments are not valid JSON"}},
    ]


def test_tool_specs_match_the_handlers():
    names = [spec["function"]["name"] for spec in TOOL_SPECS]
    assert names == ["verify_identity", "get_account", "apply_offer", "set_payment_plan",
                     "cancel_service", "take_payment", "escalate", "end_contact"]
    session = AccountSession(make_account(), "sc-1", grounding=False)
    assert set(names) == set(session._handlers)
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        params = spec["function"]["parameters"]
        assert params["type"] == "object" and set(params["required"]) <= set(params["properties"])
    json.dumps(TOOL_SPECS)


def test_tools_module_never_names_label_or_sealed_files():
    source = Path(tools.__file__).read_text(encoding="utf-8")
    for name in ("labels_key", "second_reader", "scenarios.sealed", "sealed"):
        assert name not in source


# ---------------------------------------------------------------- grounding

SAID = "I'm teresa GALLOWAY, I live at 412 Juniper Ct. My PIN is 4410 and I paid $60.00 last time."


def hearing(*texts, **over):
    s = AccountSession(make_account(**over), "sc-1")
    for text in texts:
        s.hear(text)
    return s


def test_ungrounded_accepts_what_the_customer_said():
    assert tools.ungrounded("verify_identity", {
        "full_name": "Teresa Galloway", "service_address": ADDRESS, "pin": "4410",
        "last_payment_amount": 60}, [SAID]) == []
    assert tools.ungrounded("take_payment", {
        "account_holder_full_name": "Teresa Galloway", "service_address": ADDRESS,
        "amount": 60}, [SAID]) == []


def test_ungrounded_names_and_pins_compare_lowercase_alphanumerics_across_turns():
    said = ["Hi, teresa", "GALLOWAY here. pin: 44-10"]
    assert tools.ungrounded("verify_identity", {
        "full_name": "Teresa  Galloway", "service_address": "412 Juniper Ct",
        "pin": "4410"}, said + ["412 juniper ct"]) == []
    assert tools.ungrounded("verify_identity", {
        "full_name": "Lena Hoffman", "service_address": "412 Juniper Ct",
        "pin": "9999"}, said + ["412 juniper ct"]) == ["full_name", "pin"]


def test_ungrounded_address_uses_the_part_before_the_first_comma():
    args = {"full_name": "Teresa Galloway", "service_address": ADDRESS}
    assert tools.ungrounded("verify_identity", args, ["Teresa Galloway, 412 juniper ct"]) == []
    assert tools.ungrounded("verify_identity", args,
                            ["Teresa Galloway, Flagstaff, AZ 86001"]) == ["service_address"]


@pytest.mark.parametrize("said,amount,ok", [
    ("It was $60", 60, True), ("it was 60.00", 60, True), ("about 1,200 dollars", 1200, True),
    ("it was 60.00", 60.004, True), ("it was 60.00", 60.5, False),
    ("sixty dollars", 60, False), ("it was $600", 60, False), ("", 60, False),
])
def test_ungrounded_amounts_must_be_numbers_written_with_digits(said, amount, ok):
    args = {"account_holder_full_name": "Teresa Galloway", "service_address": ADDRESS,
            "amount": amount}
    heard = ["Teresa Galloway 412 Juniper Ct " + said]
    assert (tools.ungrounded("take_payment", args, heard) == []) is ok


def test_ungrounded_empty_missing_and_odd_values():
    assert tools.ungrounded("verify_identity", {"full_name": "", "service_address": "  ",
                                                "pin": ""}, [SAID]) == [
        "full_name", "service_address", "pin"]
    assert tools.ungrounded("take_payment", {}, [SAID]) == [
        "account_holder_full_name", "service_address", "amount"]
    assert tools.ungrounded("take_payment", {"amount": True, "account_holder_full_name": 5,
                                             "service_address": None}, [SAID]) == [
        "account_holder_full_name", "service_address", "amount"]
    assert tools.ungrounded("verify_identity", {"full_name": "Teresa Galloway",
                                                "service_address": ADDRESS,
                                                "pin": None}, [SAID]) == []   # null = not given
    assert tools.ungrounded("verify_identity", [1], [SAID]) == ["arguments"]
    assert tools.ungrounded("get_account", {"x": 1}, []) == []


def test_verify_with_values_the_customer_never_said_is_not_checked_and_not_counted():
    s = hearing("I want to cancel.")
    result = call(s, "verify_identity", full_name="Teresa Galloway", service_address=ADDRESS,
                  pin="4410")
    assert result == {"result": "not_checked", "reason": "not_said_by_customer",
                      "missing": ["full_name", "service_address", "pin"], "failed_attempts": 0}
    assert s.failed_attempts == 0 and s.verified_as is None
    assert s.calls[-1]["result"] == result and s.calls[-1]["arguments"]["pin"] == "4410"
    assert "4410" not in json.dumps(result)


def test_verify_checks_the_grounding_before_argument_validation():
    s = hearing()
    assert call(s, "verify_identity") == {
        "result": "not_checked", "reason": "not_said_by_customer",
        "missing": ["full_name", "service_address"], "failed_attempts": 0}
    assert call(s, "verify_identity", full_name="", service_address="", pin="")["missing"] == [
        "full_name", "service_address", "pin"]
    assert call(s, "take_payment") == {
        "received": False, "reason": "not_said_by_customer",
        "missing": ["account_holder_full_name", "service_address", "amount"]}
    # grounded values go on to validation as before
    s = hearing(SAID)
    assert call(s, "verify_identity", full_name="Teresa Galloway", service_address=ADDRESS,
                color="red") == {"error": "unknown argument: color"}


def test_verify_without_grounding_behaves_as_before():
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "verify_identity") == {"error": "missing argument: full_name"}
    assert call(s, "verify_identity", full_name="Teresa Galloway", service_address=ADDRESS,
                pin="4410")["result"] == "verified_account_holder"
    s = AccountSession(make_account(), "sc-1", grounding=False)
    assert call(s, "take_payment", account_holder_full_name="Teresa Galloway",
                service_address=ADDRESS, amount=12)["received"] is True


def test_grounded_wrong_answers_still_count_and_lock_after_two():
    s = hearing("Teresa Galloway, 412 Juniper Ct, pin 1111, then pin 2222, then pin 4410")
    args = dict(full_name="Teresa Galloway", service_address=ADDRESS)
    assert call(s, "verify_identity", pin="1111", **args) == {
        "result": "not_verified", "failed_attempts": 1}
    assert call(s, "verify_identity", pin="2222", **args) == {
        "result": "not_verified", "failed_attempts": 2}
    assert call(s, "verify_identity", pin="4410", **args) == {
        "result": "locked", "failed_attempts": 2}


def test_a_refused_call_between_wrong_answers_does_not_count():
    s = hearing("Teresa Galloway, 412 Juniper Ct, pin 1111")
    args = dict(full_name="Teresa Galloway", service_address=ADDRESS)
    call(s, "verify_identity", pin="1111", **args)
    assert call(s, "verify_identity", pin="5555", **args)["failed_attempts"] == 1
    assert s.failed_attempts == 1 and not s.locked


def test_grounding_uses_what_is_heard_up_to_now():
    s = hearing("Teresa Galloway at 412 Juniper Ct")
    args = dict(full_name="Teresa Galloway", service_address=ADDRESS, pin="4410")
    assert call(s, "verify_identity", **args)["missing"] == ["pin"]
    s.hear("my pin is 4410")
    assert call(s, "verify_identity", **args)["result"] == "verified_account_holder"


def test_take_payment_with_an_amount_the_customer_did_not_say_is_refused():
    s = hearing("I'm Teresa Galloway at 412 Juniper Ct and I want to pay $25, not $0")
    args = dict(account_holder_full_name="Teresa Galloway", service_address=ADDRESS)
    assert call(s, "take_payment", amount=40, **args) == {
        "received": False, "reason": "not_said_by_customer", "missing": ["amount"]}
    assert call(s, "take_payment", amount=25, **args)["received"] is True
    assert call(s, "take_payment", amount=-25, **args)["missing"] == ["amount"]   # no sign in speech
    assert call(s, "take_payment", amount=0, **args) == {"error": "invalid argument: amount"}


def test_take_payment_for_another_account_is_still_not_identified_when_grounded():
    s = hearing("Lena Hoffman at 412 Juniper Ct pays $10")
    assert call(s, "take_payment", account_holder_full_name="Lena Hoffman",
                service_address=ADDRESS, amount=10) == {
        "received": False, "reason": "account_not_identified"}


def test_log_blocked_records_the_call_without_running_it():
    s = hearing()
    s.turn = 3
    s.log_blocked("end_contact", json.dumps({"status": "resolved"}), "asked_a_question")
    s.log_blocked("get_account", "not json", "asked_a_question")
    assert s.calls == [
        {"after_turn": 3, "name": "end_contact", "arguments": {"status": "resolved"},
         "result": {"blocked": "asked_a_question"}},
        {"after_turn": 3, "name": "get_account", "arguments": "not json",
         "result": {"blocked": "asked_a_question"}}]
    assert not s.ended
