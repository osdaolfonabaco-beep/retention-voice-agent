"""Sealed scenario cards: base64 of UTF-8 JSONL, one card per line."""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

KEPT_FIELDS = ("id", "model", "channel", "account", "brief", "min_turns", "max_turns")


class CardError(Exception):
    """Invalid sealed file or card. Messages never include the brief."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check(card, index: int, seen: set[str]) -> dict:
    if not isinstance(card, dict):
        raise CardError(f"card #{index}: not a JSON object")
    cid = card.get("id")
    label = f"card {cid!r}" if isinstance(cid, str) and cid else f"card #{index}"
    if not isinstance(cid, str) or not cid:
        raise CardError(f"{label}: field 'id' must be a non-empty string")
    if cid in seen:
        raise CardError(f"{label}: field 'id' is duplicated")
    seen.add(cid)
    for field in ("model", "channel", "brief"):
        if not isinstance(card.get(field), str) or not card[field].strip():
            raise CardError(f"{label}: field '{field}' must be a non-empty string")
    if not isinstance(card.get("account"), dict):
        raise CardError(f"{label}: field 'account' must be an object")
    for field in ("min_turns", "max_turns"):
        if not _is_int(card.get(field)) or card[field] < 1:
            raise CardError(f"{label}: field '{field}' must be a positive integer")
    if card["min_turns"] > card["max_turns"]:
        raise CardError(f"{label}: field 'min_turns' is greater than 'max_turns'")
    return {k: card[k] for k in KEPT_FIELDS}


def load_cards(path) -> list[dict]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="ascii", errors="strict")
    except FileNotFoundError:
        raise CardError(f"scenario file not found: {path}") from None
    except UnicodeDecodeError:
        raise CardError(f"{path.name}: not valid base64 text") from None
    try:
        decoded = base64.b64decode("".join(raw.split()), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        raise CardError(f"{path.name}: not valid base64 of UTF-8 text") from None
    cards, seen = [], set()
    for index, line in enumerate(decoded.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            card = json.loads(line)
        except json.JSONDecodeError:
            raise CardError(f"card #{index}: invalid JSON") from None
        cards.append(_check(card, index, seen))
    return cards


def seal_cards(cards: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl = "\n".join(json.dumps(c, ensure_ascii=False) for c in cards) + "\n"
    path.write_text(base64.b64encode(jsonl.encode("utf-8")).decode("ascii") + "\n")
