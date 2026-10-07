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


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


def tool_response(text, calls, usage=(21, 5)):
    msg = SimpleNamespace(content=text, tool_calls=calls)
    u = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=u)


def recording_openai(resp=None, error=None):
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        if error:
            raise error
        return resp

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return fake, seen


def test_chat_with_tools_returns_text_tool_calls_and_usage():
    resp = tool_response(None, [tool_call("c1", "get_account", "{}"),
                                tool_call("c2", "end_contact", '{"status": "resolved"}')])
    fake, seen = recording_openai(resp)
    specs = [{"type": "function", "function": {"name": "get_account"}}]
    message, usage = ChatClient("u", "k", client=fake).chat_with_tools("m", [{"role": "user"}], specs)
    assert message == {"content": "", "tool_calls": [
        {"id": "c1", "name": "get_account", "arguments": "{}"},
        {"id": "c2", "name": "end_contact", "arguments": '{"status": "resolved"}'}]}
    assert usage == {"input_tokens": 21, "output_tokens": 5}
    assert seen == [{"model": "m", "messages": [{"role": "user"}], "tools": specs}]


def test_chat_with_tools_without_tools_sends_none_and_handles_plain_text():
    fake, seen = recording_openai(tool_response("hello", None))
    message, _ = ChatClient("u", "k", client=fake).chat_with_tools("m", [])
    assert message == {"content": "hello", "tool_calls": []}
    assert "tools" not in seen[0]
    fake, _ = recording_openai(response("plain", None))     # a message with no tool_calls field
    message, usage = ChatClient("u", "k", client=fake).chat_with_tools("m", [], tools=[])
    assert message == {"content": "plain", "tool_calls": []}
    assert usage == {"input_tokens": None, "output_tokens": None}


def test_chat_with_tools_sends_reasoning_effort_only_when_set():
    fake, seen = recording_openai(tool_response("hi", None))
    client = ChatClient("u", "k", client=fake)
    client.chat_with_tools("m", [], reasoning_effort="none")
    client.chat_with_tools("m", [])
    client.chat_with_tools("m", [], reasoning_effort=None)
    assert seen[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in seen[1] and "reasoning_effort" not in seen[2]


def test_complete_methods_never_send_reasoning_effort():
    fake, seen = recording_openai(response("hi"))
    client = ChatClient("u", "k", client=fake)
    client.complete("m", [])
    client.complete_with_usage("m", [])
    assert all("reasoning_effort" not in kw for kw in seen)


def test_chat_with_tools_empty_choices():
    empty = SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=0))
    message, usage = ChatClient("u", "k", client=fake_openai(empty)).chat_with_tools("m", [])
    assert message == {"content": "", "tool_calls": []}
    assert usage == {"input_tokens": 3, "output_tokens": 0}


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


@pytest.mark.parametrize("error, expected", [
    (status_error(openai.RateLimitError, 429), RateLimited),
    (status_error(openai.InternalServerError, 503), TransientError),
    (openai.APIConnectionError(request=httpx2.Request("POST", "http://x")), TransientError),
    (status_error(openai.BadRequestError, 400), LLMError),
    (ValueError("x"), LLMError),
])
def test_chat_with_tools_maps_errors_like_complete(error, expected):
    client = ChatClient("u", "k", client=fake_openai(error=error))
    with pytest.raises(expected) as with_tools:
        client.chat_with_tools("m", [], tools=[{"type": "function"}])
    with pytest.raises(expected) as plain:
        client.complete("m", [])
    assert type(with_tools.value) is type(plain.value) is expected
    assert str(with_tools.value) == str(plain.value)
