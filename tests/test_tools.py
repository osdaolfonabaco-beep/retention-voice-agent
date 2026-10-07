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
    result = offer(s, yes=False)
    assert result["reason"] == "customer_has_not_said_yes" and result["offer_available"] is True
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
        "set": False, "reason": "customer_has_not_said_yes", "plan_fits": True}
    for stage in ("Current", "Recovery"):
        s = verified(stage=stage, past_due_balance=120)
        assert plan(s, [{"date": "2026-10-06", "amount": 120}]) == {
            "set": False, "reason": "stage_not_eligible", "plan_fits": False}


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
    assert attempt(first_late, last_ok) == {
        "set": False, "reason": "first_payment_too_late", "plan_fits": False}
    if stage == "Late":  # the first date is already near the limit; test the last date alone
        s = verified(stage=stage, past_due_balance=100)
        assert plan(s, [{"date": "2026-10-06", "amount": 50}, {"date": last_late, "amount": 50}]) == {
            "set": False, "reason": "plan_too_long", "plan_fits": False}
    else:
        assert attempt(first_ok, last_late) == {
            "set": False, "reason": "plan_too_long", "plan_fits": False}


def test_payment_plan_amount_and_date_rules():
    s = early()
    assert plan(s, [{"date": "2026-10-04", "amount": 120}]) == {
        "set": False, "reason": "date_before_today", "plan_fits": False}
    assert plan(s, [{"date": "2026-10-06", "amount": 100}]) == {
        "set": False, "reason": "total_must_equal_past_due_balance", "plan_fits": False}
    assert plan(s, [{"date": "2026-10-06", "amount": 130}, {"date": "2026-10-08", "amount": -10}]) == {
        "set": False, "reason": "amounts_must_be_positive", "plan_fits": False}
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
    ("it was $600", 60, False), ("", 60, False),
])
def test_ungrounded_amounts_written_with_digits(said, amount, ok):
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


# ---------------------------------------------------------------- numbers in words

def _pay(amount):
    return {"account_holder_full_name": "Teresa Galloway", "service_address": ADDRESS,
            "amount": amount}


@pytest.mark.parametrize("said, value", [
    ("sixty", 60), ("I can pay sixty dollars", 60), ("forty-five", 45), ("forty five", 45),
    ("twelve", 12), ("nineteen bucks", 19), ("zero", 0), ("five", 5),
    ("one hundred and ten", 110), ("one hundred ten", 110), ("a hundred", 100),
    ("two hundred", 200), ("nine hundred ninety nine", 999),
    ("a thousand", 1000), ("two thousand five hundred", 2500),
    ("one thousand and twelve", 1012), ("nine thousand nine hundred and ninety-nine", 9999),
    ("SIXTY Dollars", 60),
])
def test_ungrounded_accepts_amounts_written_in_words(said, value):
    assert tools.ungrounded("take_payment", _pay(value), ["Teresa Galloway 412 Juniper Ct " + said]) == []


@pytest.mark.parametrize("said, value", [
    ("sixty dollars", 61), ("sixty dollars", 600), ("sixty dollars", 6), ("forty five", 40),
    ("ten thousand", 10000), ("hundred", 100), ("sixty sixty", 120), ("sixteen", 60),
    ("something else", 60),
])
def test_ungrounded_rejects_amounts_the_words_do_not_make(said, value):
    assert tools.ungrounded("take_payment", _pay(value), ["Teresa Galloway 412 Juniper Ct " + said]) == ["amount"]


def test_word_numbers_lists_each_number_in_order():
    assert tools._word_numbers("sixty, then forty-five and later one hundred and ten") == [
        60.0, 45.0, 110.0]
    assert tools._word_numbers("twenty twenty") == [20.0, 20.0]


def test_amounts_in_words_work_across_turns_and_with_digits_unchanged():
    said = ["Teresa Galloway at 412 Juniper Ct", "about sixty", "or maybe $45.50"]
    assert tools.ungrounded("take_payment", _pay(60), said) == []
    assert tools.ungrounded("take_payment", _pay(45.5), said) == []
    assert tools.ungrounded("take_payment", _pay(45), said) == ["amount"]


@pytest.mark.parametrize("said, pin, ok", [
    ("four four one zero", "4410", True), ("Four-Four-One-Zero", "4410", True),
    ("four four one oh", "4410", True), ("4 4 one 0", "4410", True),
    ("it is four, four, one, zero", "4410", True), ("my pin is 4410", "4410", True),
    ("four four one", "4410", False), ("four four ten", "4410", False),
    ("forty four ten", "4410", False), ("four four one two", "4410", False),
    ("four apples four one zero", "4410", False),
])
def test_ungrounded_pin_may_be_said_digit_by_digit(said, pin, ok):
    args = {"full_name": "Teresa Galloway", "service_address": ADDRESS, "pin": pin}
    heard = ["Teresa Galloway 412 Juniper Ct " + said]
    assert (tools.ungrounded("verify_identity", args, heard) == []) is ok


def test_digit_words_do_not_ground_names_or_addresses():
    args = {"full_name": "Teresa Galloway", "service_address": "4 Juniper Ct, X"}
    assert tools.ungrounded("verify_identity", args, ["Teresa Galloway, four Juniper Ct"]) == [
        "service_address"]
    args = {"full_name": "Tom 4", "service_address": "9 Elm St"}
    assert tools.ungrounded("verify_identity", args, ["Tom four, 9 Elm St"]) == ["full_name"]


def test_session_grounds_a_pin_said_in_words():
    s = AccountSession(make_account(), "sc-1", grounding=True)
    s.hear("I'm Teresa Galloway at 412 Juniper Ct, pin four four one zero")
    result = call(s, "verify_identity", full_name="Teresa Galloway",
                  service_address=ADDRESS, pin="4410")
    assert result["result"] == "verified_account_holder"


# ---------------------------------------------------------------- tools answer before the yes

def test_tools_version_and_descriptions_tell_the_model_it_can_check_first():
    assert tools.TOOLS_VERSION == 3
    descriptions = {t["function"]["name"]: t["function"]["description"] for t in tools.TOOL_SPECS}
    for name in ("apply_offer", "set_payment_plan"):
        assert "customer_said_yes false" in descriptions[name]
    assert "false" not in descriptions["cancel_service"]


def test_plan_that_fits_without_a_yes_is_not_set():
    s = early()
    first = plan(s, [{"date": "2026-10-06", "amount": 120}], yes=False)
    assert first == {"set": False, "reason": "customer_has_not_said_yes", "plan_fits": True}
    assert plan(s, [{"date": "2026-10-06", "amount": 120}])["set"] is True   # nothing was set before


def test_plan_that_breaks_a_limit_reports_its_reason_with_or_without_a_yes():
    for yes in (True, False):
        s = early()
        assert plan(s, [{"date": "2026-10-06", "amount": 100}], yes=yes) == {
            "set": False, "reason": "total_must_equal_past_due_balance", "plan_fits": False}
        assert plan(s, [{"date": "2027-06-01", "amount": 120}], yes=yes)["reason"] == (
            "first_payment_too_late")


def test_plan_still_needs_verification_first():
    s = AccountSession(make_account(stage="Early", past_due_balance=120), "sc-1", grounding=False)
    assert plan(s, [{"date": "2027-06-01", "amount": 1}], yes=False) == {
        "set": False, "reason": "not_verified"}


def test_plan_malformed_dates_are_errors_even_without_a_yes():
    assert plan(early(), [{"date": "10/06/2026", "amount": 120}], yes=False) == {
        "error": "invalid argument: date"}


def test_available_offer_without_a_yes_returns_a_preview_and_changes_nothing():
    s = verified()
    preview = offer(s, yes=False)
    assert preview["applied"] is False and preview["offer_available"] is True
    assert preview["reason"] == "customer_has_not_said_yes"
    applied = offer(s)
    assert applied["applied"] is True and preview["preview"] == applied["details"]
    assert s.applied_offers == {"discount_25_3mo"}


@pytest.mark.parametrize("name, extra, over", [
    ("downgrade", {"target_plan": "Basic"}, {}),
    ("service_credit", {"amount": 10}, {"confirmed_service_issue": "outage"}),
    ("extra_days", {"days": 5}, {"confirmed_service_issue": "outage"}),
])
def test_every_offer_previews_what_it_would_apply(name, extra, over):
    s = verified(**over)
    preview = offer(s, name, yes=False, **extra)
    assert preview["offer_available"] is True
    assert s.applied_offers == set()
    assert preview["preview"] == offer(s, name, **extra)["details"]


@pytest.mark.parametrize("name, extra, over, reason", [
    ("discount_25_3mo", {}, {"retention_discount_eligible": False}, "not_eligible"),
    ("downgrade", {"target_plan": "Giga"}, {}, "target_plan_not_cheaper"),
    ("service_credit", {"amount": 10}, {}, "no_confirmed_service_issue"),
    ("service_credit", {"amount": 99}, {"confirmed_service_issue": "outage"}, "amount_not_allowed"),
    ("extra_days", {"days": 11}, {"confirmed_service_issue": "outage"}, "days_not_allowed"),
    ("discount_25_3mo", {}, {"next_bill": None}, "no_upcoming_bill"),
])
def test_refused_offers_keep_their_reason_whatever_the_yes(name, extra, over, reason):
    for yes in (True, False):
        result = offer(verified(**over), name, yes=yes, **extra)
        assert result["applied"] is False and result["reason"] == reason
        assert "offer_available" not in result


def test_an_offer_already_applied_is_refused_without_a_yes_too():
    s = verified()
    offer(s)
    assert offer(s, yes=False) == {"applied": False, "reason": "already_applied"}


def test_offer_argument_errors_come_before_the_yes():
    assert offer(verified(), "downgrade", yes=False) == {"error": "missing argument: target_plan"}


def test_cancel_service_is_unchanged_by_the_new_order():
    s = verified()
    assert call(s, "cancel_service", customer_said_yes=False) == {
        "cancelled": False, "reason": "customer_has_not_said_yes"}
    assert call(s, "cancel_service", customer_said_yes=True)["cancelled"] is True


# ---------------------------------------------------------------- close names and the proof question

CLOSE = [
    ("Marcus Oyelaran", "Marcus Oyeleron"), ("Marcus Oyelaran", "Marcus Oyularin"),
    ("Renee Fontenot", "Rene Fontenot"), ("Simone Abara", "Simone Ibarra"),
    ("Tomasz Wrobel", "Tomas Wroble"), ("Bettina Krause", "Bettina Kraus"),
    ("Carmen Villalobos", "Carmen Villa Lobos"),
]
NOT_CLOSE = [
    ("Arthur Pennington", "Julia Pennington"), ("Meredith Vaughn", "Paul Vaughn"),
    ("Samuel Ortiz", "Diego Ortiz"), ("Eugene Marlowe", "Susan Marlowe"),
    ("Gordon Pratt", "Jordan Pratt"),
]


@pytest.mark.parametrize("on_file, given", CLOSE)
def test_names_close_accepts_names_that_sound_alike(on_file, given):
    assert tools.names_close(given, on_file) is True


@pytest.mark.parametrize("on_file, given", NOT_CLOSE)
def test_names_close_rejects_a_different_first_name(on_file, given):
    assert tools.names_close(given, on_file) is False


@pytest.mark.parametrize("given, on_file", [
    ("Teresa Galloway", "Teresa Galloway"), ("  teresa   GALLOWAY ", "Teresa Galloway"),
    ("", "Teresa Galloway"), ("1234", "Teresa Galloway"), ("Teresa", "Teresa Galloway"),
    ("Galloway Teresa", "Teresa Galloway"),
])
def test_names_close_is_false_for_equal_empty_and_unrelated_names(given, on_file):
    assert tools.names_close(given, on_file) is False


def test_names_close_ignores_everything_but_letters():
    assert tools.names_close("Marcus O'Yelaran", "Marcus Oyelaran") is True      # not equal, ratio 1
    assert tools.names_close("Marcus Oyeleron!", "Marcus Oyelaran") is True


Z_ADDRESS = "88 Larkspur Loop, Boise, ID 83702"
Z_SAID = "I'm Marcus Oyeleron at 88 Larkspur Loop, pin 4410"


def tolerant(grounding=False, tolerance=True, **over):
    account = make_account(account_holder="Marcus Oyelaran", authorized_users=["Lena Hoffman"],
                           service_address=Z_ADDRESS, **over)
    return AccountSession(account, "sc-1", grounding=grounding, name_tolerance=tolerance)


def close_verify(s, name="Marcus Oyeleron", **extra):
    args = dict(full_name=name, service_address=Z_ADDRESS, pin="4410")
    args.update(extra)
    return call(s, "verify_identity", **args)


ASK = {"result": "more_proof_needed", "ask_for": "zip_code", "failed_attempts": 0}


def test_an_exact_name_behaves_as_before_and_ignores_zip_code():
    for extra in ({}, {"zip_code": "99999"}):
        s = tolerant()
        assert close_verify(s, "Marcus Oyelaran", **extra) == {
            "result": "verified_account_holder", "failed_attempts": 0}


def test_a_close_name_without_zip_asks_for_more_proof_and_counts_no_attempt():
    s = tolerant()
    assert close_verify(s) == ASK
    assert close_verify(s) == ASK and close_verify(s) == ASK     # asking never locks
    assert s.failed_attempts == 0 and s.verified_as is None and not s.locked
    assert call(s, "get_account") == {"error": "not_verified"}


def test_a_close_name_with_the_right_zip_verifies_and_says_it_was_close():
    s = tolerant()
    assert close_verify(s) == ASK
    assert close_verify(s, zip_code="83702") == {
        "result": "verified_account_holder", "failed_attempts": 0, "name_match": "close"}
    assert "plan" in call(s, "get_account")


@pytest.mark.parametrize("zip_code", ["83702", "83702.", "8 3 7 0 2", "83702-1234",
                                      " ZIP 83702 ", "837021234"])
def test_the_zip_is_compared_on_digits_only(zip_code):
    s = tolerant()
    close_verify(s)
    assert close_verify(s, zip_code=zip_code)["result"] == "verified_account_holder"


@pytest.mark.parametrize("zip_code", ["83703", "8370", "837020", "83702-12", "83702123", "1883702",
                                      "8370212345", "boise"])
def test_a_wrong_or_malformed_zip_is_not_verified_and_counts_an_attempt(zip_code):
    s = tolerant()
    close_verify(s)
    if re.sub(r"\D", "", zip_code):
        assert close_verify(s, zip_code=zip_code) == {"result": "not_verified", "failed_attempts": 1}
    else:                                   # no digits: the same as no zip_code
        assert close_verify(s, zip_code=zip_code) == ASK


def test_two_wrong_zips_lock_the_session_even_for_the_right_one():
    s = tolerant()
    close_verify(s)
    assert close_verify(s, zip_code="11111") == {"result": "not_verified", "failed_attempts": 1}
    assert close_verify(s, zip_code="22222") == {"result": "not_verified", "failed_attempts": 2}
    assert close_verify(s, zip_code="83702") == {"result": "locked", "failed_attempts": 2}
    assert s.locked and call(s, "get_account") == {"error": "not_verified"}


def test_a_close_name_with_a_wrong_pin_or_address_is_as_before_and_gets_no_proof_question():
    for args, attempts in (({"pin": "0000"}, 1), ({"service_address": "500 Elm St"}, 1)):
        s = tolerant()
        assert close_verify(s, **args) == {"result": "not_verified", "failed_attempts": attempts}
    s = tolerant()
    assert call(s, "verify_identity", full_name="Marcus Oyeleron", service_address=Z_ADDRESS) == {
        "result": "not_verified", "failed_attempts": 0}              # no PIN or amount: not counted


def test_a_close_name_works_with_the_last_payment_instead_of_the_pin():
    s = tolerant()
    assert call(s, "verify_identity", full_name="Marcus Oyeleron", service_address=Z_ADDRESS,
                last_payment_amount=60) == ASK
    assert call(s, "verify_identity", full_name="Marcus Oyeleron", service_address=Z_ADDRESS,
                last_payment_amount=60, zip_code="83702")["name_match"] == "close"


def test_a_close_authorized_user_name_verifies_as_authorized_user():
    s = tolerant()
    assert close_verify(s, "Lena Hofman") == ASK
    assert close_verify(s, "Lena Hofman", zip_code="83702") == {
        "result": "verified_authorized_user", "failed_attempts": 0, "name_match": "close"}


def test_a_name_that_is_neither_exact_nor_close_is_as_before():
    s = tolerant()
    assert close_verify(s, "Julia Oyelaran", zip_code="83702") == {
        "result": "not_verified", "failed_attempts": 1}


def test_tolerance_off_is_exactly_the_old_behaviour():
    s = tolerant(tolerance=False)
    assert close_verify(s) == {"result": "not_verified", "failed_attempts": 1}
    assert close_verify(s, zip_code="83702") == {"result": "not_verified", "failed_attempts": 2}
    assert close_verify(s, "Marcus Oyelaran") == {"result": "locked", "failed_attempts": 2}
    assert tools.AccountSession(make_account(), "sc-1").name_tolerance is False
    s = tolerant(tolerance=False)
    assert close_verify(s, "Marcus Oyelaran") == {
        "result": "verified_account_holder", "failed_attempts": 0}


def test_a_close_name_cannot_verify_when_no_zip_is_on_file():
    s = AccountSession(make_account(account_holder="Marcus Oyelaran",
                                    service_address="88 Larkspur Loop, Boise"),
                       "sc-1", grounding=False, name_tolerance=True)
    result = call(s, "verify_identity", full_name="Marcus Oyeleron",
                  service_address="88 Larkspur Loop", pin="4410", zip_code="83702")
    assert result == {"result": "not_verified", "failed_attempts": 1}


def test_the_zip_on_file_is_the_last_five_digit_group_of_the_address():
    s = AccountSession(make_account(service_address="12345 Main St, Boise, ID 83702"), "sc-1")
    assert s._zip_on_file() == "83702"
    s = AccountSession(make_account(service_address="1234 Main St, Boise, ID 837021"), "sc-1")
    assert s._zip_on_file() is None


def test_no_result_reveals_the_name_or_the_zip_on_file():
    for step in range(3):
        s = tolerant()
        out = [close_verify(s), close_verify(s, zip_code="11111"),
               close_verify(s, zip_code="83702")][:step + 1]
        text = json.dumps(out)
        assert "83702" not in text and "Oyelaran" not in text and "Larkspur" not in text


def test_a_zip_sent_before_any_proof_question_counts_as_absent():
    s = tolerant()
    assert close_verify(s, zip_code="83702") == ASK          # nothing counted, not verified
    assert s.failed_attempts == 0 and s.verified_as is None
    assert close_verify(s, zip_code="83702")["name_match"] == "close"     # now it was asked


def test_a_zip_with_an_exact_name_does_not_open_the_proof_question():
    s = tolerant()
    close_verify(s, "Marcus Oyelaran", zip_code="83702")     # verified, ignored
    s = tolerant()
    call(s, "verify_identity", full_name="Marcus Oyelaran", service_address="500 Elm St",
         pin="4410", zip_code="83702")                       # wrong address: not_verified as before
    assert close_verify(s, zip_code="83702") == {**ASK, "failed_attempts": 1}   # never asked yet


def test_the_customer_must_say_the_zip_after_the_proof_question():
    s = tolerant(grounding=True)
    s.hear(Z_SAID + ", zip 83702")                           # said too early
    assert close_verify(s) == ASK
    assert close_verify(s, zip_code="83702") == ASK          # grounded only in earlier text: absent
    assert s.failed_attempts == 0
    s.hear("Sure, it is 83702.")
    assert close_verify(s, zip_code="83702") == {
        "result": "verified_account_holder", "failed_attempts": 0, "name_match": "close"}


def test_a_zip_the_customer_never_said_counts_as_absent_not_as_a_failed_attempt():
    s = tolerant(grounding=True)
    s.hear(Z_SAID)
    assert close_verify(s) == ASK
    s.hear("Sorry, what do you need?")
    assert close_verify(s, zip_code="83702") == ASK
    assert s.failed_attempts == 0


def test_every_new_proof_question_moves_the_position():
    s = tolerant(grounding=True)
    s.hear(Z_SAID)
    assert close_verify(s) == ASK
    assert close_verify(s, zip_code="83702") == ASK          # asks again: the position moves up
    s.hear("it is 83702")
    assert close_verify(s, zip_code="83702")["result"] == "verified_account_holder"


@pytest.mark.parametrize("said", ["8 3 7 0 2", "eight three seven oh two", "It's 83702, thanks",
                                  "83702 dash 1234"])
def test_a_zip_said_in_digits_or_words_is_grounded(said):
    s = tolerant(grounding=True)
    s.hear(Z_SAID)
    close_verify(s)
    s.hear(said)
    assert close_verify(s, zip_code="83702")["result"] == "verified_account_holder"


def test_zip_code_is_not_a_refusal_in_the_grounding_check():
    args = {"full_name": "Marcus Oyeleron", "service_address": Z_ADDRESS, "pin": "4410",
            "zip_code": "83702"}
    assert tools.ungrounded("verify_identity", args, [Z_SAID]) == []


def test_the_proof_question_is_the_same_with_grounding_off():
    s = tolerant(grounding=False)
    assert close_verify(s, zip_code="83702") == ASK
    assert close_verify(s, zip_code="83702")["name_match"] == "close"


def test_verify_identity_spec_has_zip_code_and_says_to_send_it_after_the_question():
    spec = next(t["function"] for t in TOOL_SPECS if t["function"]["name"] == "verify_identity")
    assert spec["parameters"]["properties"]["zip_code"]["type"] == "string"
    assert "zip_code" not in spec["parameters"]["required"]
    assert "only after the tool asked for more proof" in spec["description"]


def pay_close(s, name="Marcus Oyeleron", address=Z_ADDRESS):
    return call(s, "take_payment", account_holder_full_name=name, service_address=address, amount=25)


def test_take_payment_with_tolerance_accepts_a_close_holder_name_with_no_proof_question():
    s = tolerant()
    assert pay_close(s)["received"] is True
    assert s.verified_as is None and s.failed_attempts == 0


def test_take_payment_tolerance_is_off_by_default_and_needs_the_address_and_the_holder():
    assert pay_close(tolerant(tolerance=False)) == {"received": False, "reason": "account_not_identified"}
    s = tolerant()
    assert pay_close(s, address="500 Elm St")["received"] is False
    assert pay_close(s, name="Julia Oyelaran")["received"] is False
    assert pay_close(s, name="Lena Hofman")["received"] is False     # authorized users never pay
    assert pay_close(s, name="Marcus Oyelaran")["received"] is True


def test_take_payment_grounding_still_applies_with_tolerance():
    s = tolerant(grounding=True)
    assert pay_close(s)["reason"] == "not_said_by_customer"
    s.hear("This is Marcus Oyeleron at 88 Larkspur Loop and I want to pay 25 dollars")
    assert pay_close(s)["received"] is True
