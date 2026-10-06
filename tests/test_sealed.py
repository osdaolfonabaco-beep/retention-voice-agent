import base64
import json

import pytest

from retention.sealed import CardError, load_cards, seal_cards

BRIEF = "SECRET_BRIEF_TEXT"


def card(**over):
    base = {"id": "c1", "model": "m", "channel": "chat", "account": {"plan": "x"},
            "brief": BRIEF, "min_turns": 2, "max_turns": 4}
    base.update(over)
    return base


def test_round_trip_ignores_whitespace_and_extra_fields(tmp_path):
    path = tmp_path / "s.sealed"
    seal_cards([card(id="a", secret_card_field="zzz"), card(id="b")], path)
    text = path.read_text()
    path.write_text("  \n".join(text[i:i + 20] for i in range(0, len(text), 20)))
    cards = load_cards(path)
    assert [c["id"] for c in cards] == ["a", "b"]
    assert "secret_card_field" not in cards[0]
    assert cards[0]["brief"] == BRIEF


def _write_raw(path, cards):
    jsonl = "\n".join(json.dumps(c) for c in cards)
    path.write_text(base64.b64encode(jsonl.encode()).decode())


@pytest.mark.parametrize("cards,card_id,field", [
    ([card(id="")], "#1", "id"),
    ([card(), card()], "c1", "id"),
    ([card(model=3)], "c1", "model"),
    ([card(channel="")], "c1", "channel"),
    ([card(brief=" ")], "c1", "brief"),
    ([card(account="x")], "c1", "account"),
    ([card(min_turns="2")], "c1", "min_turns"),
    ([card(max_turns=True)], "c1", "max_turns"),
    ([card(min_turns=5, max_turns=3)], "c1", "min_turns"),
])
def test_card_errors_name_card_and_field_without_brief(tmp_path, cards, card_id, field):
    path = tmp_path / "s.sealed"
    _write_raw(path, cards)
    with pytest.raises(CardError) as exc:
        load_cards(path)
    msg = str(exc.value)
    assert card_id in msg and field in msg
    assert BRIEF not in msg


def test_bad_base64_and_missing_file(tmp_path):
    path = tmp_path / "s.sealed"
    path.write_text("not base64!!")
    with pytest.raises(CardError):
        load_cards(path)
    with pytest.raises(CardError):
        load_cards(tmp_path / "missing")
