"""Tools the retention agent can call, over one synthetic account.

One AccountSession holds the state of one contact. `call` never raises on bad
input: malformed arguments give {"error": ...}, business refusals give the
tool's own shape with a reason. Results never carry on-file verification values
and error text names parameters, never values.
"""

from __future__ import annotations

import calendar
import copy
import hashlib
import json
import math
import re
from datetime import date, timedelta

TOOLS_VERSION = 2       # recorded in run.json; bump when a tool's behaviour changes
PLAN_PRICES = {"Basic": 45, "Plus": 60, "Giga": 80}
OFFERS = ("discount_25_3mo", "downgrade", "service_credit", "extra_days")
# stage -> (days allowed until the first payment, months allowed until the last)
PLAN_LIMITS = {"Early": (10, 3), "Mid": (5, 2), "Late": (2, 1)}
MAX_FAILED_ATTEMPTS = 2
ESCALATION_REASONS = tuple(f"ES-{n}" for n in range(1, 8))
END_STATUSES = ("resolved", "escalated", "no_change")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CENT = 0.005


# ---------------------------------------------------------------- tool specs

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


_YES = {"type": "boolean", "description": "true only if the customer clearly said yes to this exact change"}

TOOL_SPECS = [
    _tool("verify_identity",
          "Check who the person is. Needs the full name, the service address and one more item: "
          "the account PIN or the amount of the last payment. Returns only the result.",
          {"full_name": {"type": "string"},
           "service_address": {"type": "string"},
           "last_payment_amount": {"type": "number", "description": "amount of the last payment"},
           "pin": {"type": "string", "description": "account PIN"}},
          ["full_name", "service_address"]),
    _tool("get_account",
          "Read the account: plan, price, stage, balance, next bill, eligibility, history. "
          "Only after a successful verification.", {}, []),
    _tool("apply_offer",
          "Apply one retention offer. Call it with customer_said_yes false to check whether the "
          "offer is available and to see its details before proposing it; nothing changes. Set "
          "customer_said_yes true only after the customer clearly said yes to it.",
          {"offer": {"type": "string", "enum": list(OFFERS)},
           "customer_said_yes": _YES,
           "target_plan": {"type": "string", "enum": list(PLAN_PRICES),
                           "description": "for downgrade"},
           "amount": {"type": "number", "description": "for service_credit, in dollars"},
           "days": {"type": "integer", "description": "for extra_days"}},
          ["offer", "customer_said_yes"]),
    _tool("set_payment_plan",
          "Set a payment plan for the past-due balance. Call it with customer_said_yes false to "
          "check whether a plan fits the limits before proposing it; nothing changes. Set "
          "customer_said_yes true only after the customer clearly said yes.",
          {"installments": {"type": "array", "minItems": 1, "items": {
              "type": "object",
              "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"},
                             "amount": {"type": "number"}},
              "required": ["date", "amount"]}},
           "customer_said_yes": _YES},
          ["installments", "customer_said_yes"]),
    _tool("cancel_service",
          "Cancel the service. Only after the customer clearly said yes.",
          {"customer_said_yes": _YES}, ["customer_said_yes"]),
    _tool("take_payment",
          "Take a payment from anyone who identifies the account by the account holder's full "
          "name and the service address. Never returns a balance.",
          {"account_holder_full_name": {"type": "string"},
           "service_address": {"type": "string"},
           "amount": {"type": "number", "description": "amount the payer chooses, in dollars"}},
          ["account_holder_full_name", "service_address", "amount"]),
    _tool("escalate",
          "Send the case to another team.",
          {"team": {"type": "string", "enum": ["tier2", "recovery"]},
           "reason": {"type": "string", "enum": list(ESCALATION_REASONS),
                      "description": "escalation rule that applies"},
           "summary": {"type": "string", "description": "what the next team needs to know"}},
          ["team", "reason", "summary"]),
    _tool("end_contact",
          "Mark the contact as ended, after saying goodbye.",
          {"status": {"type": "string", "enum": list(END_STATUSES)}}, ["status"]),
]
_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SPECS}


# ---------------------------------------------------------------- helpers

class _BadArguments(Exception):
    pass


def _check_value(path: str, value, rule: dict) -> None:
    kind = rule.get("type")
    ok = {
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(kind, True)
    if not ok or ("enum" in rule and value not in rule["enum"]):
        raise _BadArguments(f"invalid argument: {path}")
    if kind == "array":
        if len(value) < rule.get("minItems", 0):
            raise _BadArguments(f"invalid argument: {path}")
        for item in value:
            _check_value(f"{path}[]", item, rule["items"])
    elif kind == "object" and "properties" in rule:
        _check_object(path, value, rule, f"{path}.")


def _check_object(path: str, args: dict, schema: dict, prefix: str = "") -> dict:
    props = schema["properties"]
    for key in args:
        if key not in props:
            raise _BadArguments(f"unknown argument: {prefix}{str(key)[:40]}")
    clean = {}
    for name, rule in props.items():
        value = args.get(name)
        if value is None:
            if name in schema.get("required", ()):
                raise _BadArguments(f"missing argument: {prefix}{name}")
            continue
        _check_value(f"{prefix}{name}", value, rule)
        clean[name] = value
    return clean


def _norm_name(text: str) -> str:
    return " ".join(text.lower().split())


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z]+|\d+")
_DIGIT_WORDS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_UNITS = {w: n for w, n in _DIGIT_WORDS.items() if 1 <= n <= 9}
# tool -> (parameter, kind, required), in the order "missing" lists them
_GROUNDED = {
    "verify_identity": (("full_name", "text", True), ("service_address", "address", True),
                        ("last_payment_amount", "amount", False), ("pin", "pin", False)),
    "take_payment": (("account_holder_full_name", "text", True),
                     ("service_address", "address", True), ("amount", "amount", True)),
}


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text)]


def _below_100(words: list[str], i: int):
    """(value, next index) for "sixty", "forty five", "twelve", "zero"; None if no number."""
    word = words[i] if i < len(words) else ""
    if word in _TENS:
        if i + 1 < len(words) and words[i + 1] in _UNITS:
            return _TENS[word] + _UNITS[words[i + 1]], i + 2
        return _TENS[word], i + 1
    for table in (_TEENS, _DIGIT_WORDS):
        if word in table and word != "oh":
            return table[word], i + 1
    return None


def _multiplier(words: list[str], i: int, unit: str):
    """Index after "<1-9 or a> <unit>" at i, or None."""
    if (i + 1 < len(words) and words[i + 1] == unit
            and (words[i] in _UNITS or words[i] == "a")):
        return i + 2
    return None


def _below_1000(words: list[str], i: int):
    after = _multiplier(words, i, "hundred")
    if after is None:
        return _below_100(words, i)
    value = 100 * (1 if words[i] == "a" else _UNITS[words[i]])
    start = after + 1 if after < len(words) and words[after] == "and" else after
    rest = _below_100(words, start)
    if rest:
        return value + rest[0], rest[1]
    return value, after


def _word_numbers(text: str) -> list[float]:
    """Numbers from zero to 9,999 written in English words: "sixty", "forty-five",
    "one hundred and ten", "two thousand five hundred"."""
    words = [w for w in _WORD.findall(text.lower())]
    found, i = [], 0
    while i < len(words):
        after = _multiplier(words, i, "thousand")
        parsed = None
        if after is not None:
            value = 1000 * (1 if words[i] == "a" else _UNITS[words[i]])
            start = after + 1 if after < len(words) and words[after] == "and" else after
            rest = _below_1000(words, start) if start < len(words) else None
            parsed = (value + rest[0], rest[1]) if rest else (value, after)
        else:
            parsed = _below_1000(words, i)
        if parsed:
            found.append(float(parsed[0]))
            i = parsed[1]
        else:
            i += 1
    return found


def _spoken_digits(text: str) -> str:
    """Lowercase alphanumerics of `text` with each digit word ("four", "oh") written as a digit."""
    return "".join(str(_DIGIT_WORDS.get(w, w)) for w in _WORD.findall(text.lower()))


def ungrounded(tool: str, args, said: list[str]) -> list[str]:
    """Parameters of a verify_identity or take_payment call that the customer did not say.

    `said` holds the customer's texts so far in the contact. Names, PINs and the
    street part of an address count when their lowercase alphanumerics are a
    substring of everything said; a PIN also counts when said digit by digit in
    words ("four four one zero"). An amount counts when it is among the numbers
    written with digits or in English words (zero to 9,999). A required value that is missing or empty is ungrounded;
    an optional one (pin, last_payment_amount) only counts when it was given.
    Any other tool gives []; arguments that are not a JSON object give ["arguments"].
    """
    rules = _GROUNDED.get(tool)
    if rules is None:
        return []
    if not isinstance(args, dict):
        return ["arguments"]
    joined = "\n".join(said)
    flat, numbers = _alnum(joined), _numbers(joined) + _word_numbers(joined)
    spoken = _spoken_digits(joined)
    missing = []
    for param, kind, required in rules:
        value = args.get(param)
        if value is None and not required:
            continue
        if kind == "amount":
            ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) and any(abs(n - value) < _CENT for n in numbers))
        else:
            ok = isinstance(value, str)
            if ok:
                key = _alnum(value.split(",")[0] if kind == "address" else value)
                ok = bool(key) and (key in flat or (kind == "pin" and key in spoken))
        if not ok:
            missing.append(param)
    return missing


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _parse_date(text: str) -> date:
    if not _DATE.fullmatch(text):
        raise _BadArguments("invalid argument: date")
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise _BadArguments("invalid argument: date") from None


def _money(value: float) -> float:
    return round(float(value), 2)


# ---------------------------------------------------------------- session

class AccountSession:
    """State of one contact over one account (tools never change the account itself)."""

    def __init__(self, account: dict, scenario_id: str, grounding: bool = True):
        self.account = account
        self.scenario_id = scenario_id
        self.grounding = grounding         # refuse identity and payment values the customer never said
        self._said: list[str] = []         # customer turns so far, see `hear`
        self.calls: list[dict] = []
        self.turn = 0                       # set by the runner: turns recorded so far
        self.ended = False
        self.end_status: str | None = None
        self.verified_as: str | None = None  # "verified_account_holder" | "verified_authorized_user"
        self.failed_attempts = 0
        self.applied_offers: set[str] = set()
        self.cancelled = False
        self._counters: dict[str, int] = {}
        self._handlers = {
            "verify_identity": self._verify_identity,
            "get_account": self._get_account,
            "apply_offer": self._apply_offer,
            "set_payment_plan": self._set_payment_plan,
            "cancel_service": self._cancel_service,
            "take_payment": self._take_payment,
            "escalate": self._escalate,
            "end_contact": self._end_contact,
        }

    @property
    def locked(self) -> bool:
        return self.failed_attempts >= MAX_FAILED_ATTEMPTS and self.verified_as is None

    # -- dispatch

    def hear(self, text: str) -> None:
        """Record what the customer said. The runner calls this with every customer turn."""
        self._said.append(text)

    def log_blocked(self, name: str, arguments_json: str, reason: str) -> None:
        """Record a call that was discarded without running."""
        try:
            recorded: object = json.loads(arguments_json) if (arguments_json or "").strip() else {}
        except json.JSONDecodeError:
            recorded = arguments_json
        self.calls.append({"after_turn": self.turn, "name": name,
                           "arguments": recorded, "result": {"blocked": reason}})

    def call(self, name: str, arguments_json: str) -> dict:
        """Run one tool call. Returns a JSON-serializable dict and never raises."""
        recorded: object = arguments_json
        try:
            raw = json.loads(arguments_json) if (arguments_json or "").strip() else {}
            recorded = raw
            if not isinstance(raw, dict):
                raise _BadArguments("arguments must be a JSON object")
            handler = self._handlers.get(name)
            if handler is None:
                raise _BadArguments("unknown tool")
            missing = ungrounded(name, raw, self._said) if self.grounding else []
            if missing:     # before argument validation: empty or missing values land here too
                result = self._refuse_ungrounded(name, missing)
            else:
                result = handler(_check_object("", raw, _SCHEMAS[name]))
        except json.JSONDecodeError:
            result = {"error": "arguments are not valid JSON"}
        except _BadArguments as exc:
            result = {"error": str(exc)}
        except Exception:  # a tool must never break the contact
            result = {"error": "tool failed"}
        self.calls.append({"after_turn": self.turn, "name": name,
                           "arguments": recorded, "result": result})
        return result

    def _refuse_ungrounded(self, name: str, missing: list[str]) -> dict:
        """Nothing is compared with the account and no attempt is counted."""
        if name == "verify_identity":
            return {"result": "not_checked", "reason": "not_said_by_customer",
                    "missing": missing, "failed_attempts": self.failed_attempts}
        return {"received": False, "reason": "not_said_by_customer", "missing": missing}

    def _code(self, kind: str) -> str:
        n = self._counters.get(kind, 0)
        self._counters[kind] = n + 1
        digest = hashlib.sha256(f"{self.scenario_id}:{kind}:{n}".encode("utf-8")).hexdigest()
        return f"{int(digest, 16) % 100000:05d}"

    # -- identity

    def _address_matches(self, given: str) -> bool:
        key = _alnum(self.account["service_address"].split(",")[0])
        return bool(key) and key in _alnum(given)

    def _name_role(self, given: str) -> str | None:
        name = _norm_name(given)
        if name == _norm_name(self.account["account_holder"]):
            return "verified_account_holder"
        if name in {_norm_name(u) for u in self.account.get("authorized_users", [])}:
            return "verified_authorized_user"
        return None

    def _verify_identity(self, a: dict) -> dict:
        if self.locked:
            return {"result": "locked", "failed_attempts": self.failed_attempts}
        if self.verified_as:
            return {"result": self.verified_as, "failed_attempts": self.failed_attempts}
        on_file = self.account["on_file_for_verification"]
        has_third = "pin" in a or "last_payment_amount" in a
        third_ok = (("pin" in a and a["pin"].strip() == str(on_file["pin"]).strip())
                    or ("last_payment_amount" in a and abs(
                        a["last_payment_amount"] - float(on_file["last_payment_amount"])) < _CENT))
        role = self._name_role(a["full_name"])
        if role and self._address_matches(a["service_address"]) and third_ok:
            self.verified_as = role
            return {"result": role, "failed_attempts": self.failed_attempts}
        if has_third:
            self.failed_attempts += 1
        return {"result": "not_verified", "failed_attempts": self.failed_attempts}

    def _get_account(self, a: dict) -> dict:
        if not self.verified_as:
            return {"error": "not_verified"}
        account = copy.deepcopy(self.account)
        account.pop("on_file_for_verification", None)
        return account

    # -- changes to the account

    def _gate(self, a: dict, refusal_key: str) -> dict | None:
        """Refusal for the shared preconditions: verified, then a clear yes."""
        if not self.verified_as:
            return {refusal_key: False, "reason": "not_verified"}
        if a["customer_said_yes"] is not True:
            return {refusal_key: False, "reason": "customer_has_not_said_yes"}
        return None

    def _apply_offer(self, a: dict) -> dict:
        """Verified, then argument and eligibility checks, then the customer's yes. Without
        the yes an available offer is only previewed."""
        if not self.verified_as:
            return {"applied": False, "reason": "not_verified"}
        offer, acct = a["offer"], self.account

        def no(reason: str, **extra) -> dict:
            return {"applied": False, "reason": reason, **extra}

        if offer in self.applied_offers:
            return no("already_applied")
        bill = acct.get("next_bill")
        if offer == "downgrade" and "target_plan" not in a:
            raise _BadArguments("missing argument: target_plan")
        if offer == "service_credit" and "amount" not in a:
            raise _BadArguments("missing argument: amount")
        if offer == "extra_days" and "days" not in a:
            raise _BadArguments("missing argument: days")
        if not bill:
            return no("no_upcoming_bill")
        due = date.fromisoformat(bill["due_date"])

        if offer == "discount_25_3mo":
            if not acct.get("retention_discount_eligible"):
                return no("not_eligible")
            price = _money(acct["monthly_price"] * 0.75)
            details = {"offer": offer, "discounted_price": price,
                       "discounted_bills": [{"date": _add_months(due, i).isoformat(), "amount": price}
                                            for i in range(3)],
                       "price_after": acct["monthly_price"]}
        elif offer == "downgrade":
            current = PLAN_PRICES.get(acct["plan"], acct["monthly_price"])
            if PLAN_PRICES[a["target_plan"]] >= current:
                return no("target_plan_not_cheaper")
            details = {"offer": offer, "new_plan": a["target_plan"],
                       "new_price": PLAN_PRICES[a["target_plan"]],
                       "effective_date": bill["due_date"]}
        elif offer == "service_credit":
            if not acct.get("confirmed_service_issue"):
                return no("no_confirmed_service_issue")
            cap = 30 if acct["months_active"] >= 12 else 15
            if not 0 < a["amount"] <= cap:
                return no("amount_not_allowed", max_amount=cap)
            details = {"offer": offer, "credit_amount": _money(a["amount"]),
                       "applies_to_bill_date": bill["due_date"], "bill_amount": bill["amount"]}
        else:  # extra_days
            if not acct.get("confirmed_service_issue"):
                return no("no_confirmed_service_issue")
            if not 1 <= a["days"] <= 10:
                return no("days_not_allowed", max_days=10)
            details = {"offer": offer, "days": a["days"], "original_due_date": bill["due_date"],
                       "new_due_date": (due + timedelta(days=a["days"])).isoformat(),
                       "late_fee": 0}
        if a["customer_said_yes"] is not True:
            return {"applied": False, "reason": "customer_has_not_said_yes",
                    "offer_available": True, "preview": details}
        self.applied_offers.add(offer)
        return {"applied": True, "details": details}

    def _set_payment_plan(self, a: dict) -> dict:
        """Verified, then the plan itself, then the customer's yes."""
        if not self.verified_as:
            return {"set": False, "reason": "not_verified"}
        acct = self.account

        def no(reason: str) -> dict:
            return {"set": False, "reason": reason, "plan_fits": False}

        if acct["stage"] not in PLAN_LIMITS:
            return no("stage_not_eligible")
        max_days, max_months = PLAN_LIMITS[acct["stage"]]
        today = date.fromisoformat(acct["today"])
        plan = sorted(((_parse_date(i["date"]), i["amount"]) for i in a["installments"]),
                      key=lambda item: item[0])
        if plan[0][0] < today:
            return no("date_before_today")
        if plan[0][0] > today + timedelta(days=max_days):
            return no("first_payment_too_late")
        if plan[-1][0] > _add_months(today, max_months):
            return no("plan_too_long")
        if any(amount <= 0 for _, amount in plan):
            return no("amounts_must_be_positive")
        total = sum(amount for _, amount in plan)
        if abs(total - acct["past_due_balance"]) >= _CENT:
            return no("total_must_equal_past_due_balance")
        if a["customer_said_yes"] is not True:
            return {"set": False, "reason": "customer_has_not_said_yes", "plan_fits": True}
        return {"set": True, "installments": [{"date": d.isoformat(), "amount": _money(x)}
                                              for d, x in plan],
                "total": _money(total)}

    def _cancel_service(self, a: dict) -> dict:
        refusal = self._gate(a, "cancelled")
        if refusal:
            return refusal
        if self.cancelled:
            return {"cancelled": False, "reason": "already_cancelled"}
        self.cancelled = True
        bill = self.account.get("next_bill")
        return {"cancelled": True,
                "last_day_of_service": bill["due_date"] if bill else None,
                "balance_still_owed": self.account["past_due_balance"],
                "confirmation_email_sent": True}

    # -- no verification needed

    def _take_payment(self, a: dict) -> dict:
        if a["amount"] <= 0:
            raise _BadArguments("invalid argument: amount")
        holder = _norm_name(a["account_holder_full_name"]) == _norm_name(self.account["account_holder"])
        if not (holder and self._address_matches(a["service_address"])):
            return {"received": False, "reason": "account_not_identified"}
        return {"received": True, "confirmation_number": f"PM-{self._code('payment')}"}

    def _escalate(self, a: dict) -> dict:
        if not a["summary"].strip():
            raise _BadArguments("invalid argument: summary")
        return {"ticket": f"TV-{self._code('ticket')}", "team": a["team"],
                "reply": "by email within 2 business days"}

    def _end_contact(self, a: dict) -> dict:
        self.ended = True
        self.end_status = a["status"]
        return {"ended": True, "status": a["status"]}
