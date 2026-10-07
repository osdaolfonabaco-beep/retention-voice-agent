"""Offline tests for ChatClient.complete_with_usage."""

from types import SimpleNamespace

import httpx2
import openai
import pytest

from retention.llm import ChatClient, LLMError, RateLimited, TransientError


def fake_openai(resp=None, error=None):
    def create(**kwargs):
        if error:
            raise error
        return resp

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def response(text="hi", usage=(11, 7)):
    msg = SimpleNamespace(content=text)
    u = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=u)


def status_error(cls, code):
    req = httpx2.Request("POST", "http://x")
    return cls("boom", response=httpx2.Response(code, request=req), body=None)


def test_returns_text_and_usage():
    client = ChatClient("u", "k", client=fake_openai(response("hello", (11, 7))))
    assert client.complete_with_usage("m", []) == (
        "hello", {"input_tokens": 11, "output_tokens": 7})


def test_missing_usage_is_none():
    client = ChatClient("u", "k", client=fake_openai(response("hello", None)))
    assert client.complete_with_usage("m", []) == (
        "hello", {"input_tokens": None, "output_tokens": None})


def test_empty_choices_and_none_content():
    empty = SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=0))
    client = ChatClient("u", "k", client=fake_openai(empty))
    assert client.complete_with_usage("m", []) == ("", {"input_tokens": 3, "output_tokens": 0})
    client = ChatClient("u", "k", client=fake_openai(response(None)))
    assert client.complete_with_usage("m", [])[0] == ""


def test_complete_is_unchanged():
    client = ChatClient("u", "k", client=fake_openai(response("hello")))
    assert client.complete("m", []) == "hello"


@pytest.mark.parametrize("error, expected", [
    (status_error(openai.RateLimitError, 429), RateLimited),
    (status_error(openai.InternalServerError, 503), TransientError),
    (openai.APIConnectionError(request=httpx2.Request("POST", "http://x")), TransientError),
    (status_error(openai.BadRequestError, 400), LLMError),
    (ValueError("x"), LLMError),
])
def test_error_mapping_matches_complete(error, expected):
    client = ChatClient("u", "k", client=fake_openai(error=error))
    with pytest.raises(expected) as with_usage:
        client.complete_with_usage("m", [])
    with pytest.raises(expected) as plain:
        client.complete("m", [])
    assert type(with_usage.value) is type(plain.value) is expected
    assert str(with_usage.value) == str(plain.value)
