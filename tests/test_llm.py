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


# ---------------------------------------------------------------- audio

import io
import struct
import wave

from retention import llm


def header(rate=24000, channels=1, bits=16, riff_size=0xFFFFFFFF, data_size=0xFFFFFFFF,
           tag=1, extra_chunk=b""):
    width = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * width, width, bits)
    return (b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
            + fmt + extra_chunk + b"data" + struct.pack("<I", data_size))


def pcm(seconds, rate=24000, channels=1, bits=16):
    return b"\x01\x02" * int(seconds * rate * channels * bits // 16)


class Stream:
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.kwargs = chunks, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.error, Exception) and not self.chunks:
            raise self.error
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        for chunk in self.chunks:
            yield chunk
        if isinstance(self.error, Exception) and self.chunks:
            raise self.error


def audio_client(stream=None, transcription="  hello there \n", error=None):
    seen = {}

    def transcribe(**kwargs):
        seen.update(kwargs)
        if error:
            raise error
        return transcription

    ns = SimpleNamespace(audio=SimpleNamespace(
        transcriptions=SimpleNamespace(create=transcribe),
        speech=SimpleNamespace(with_streaming_response=stream)))
    return ns, seen


def test_transcribe_sends_the_wav_and_language_and_returns_stripped_text():
    fake, seen = audio_client()
    text = ChatClient("u", "k", client=fake).transcribe("stt-m", b"WAVBYTES")
    assert text == "hello there"
    assert seen["model"] == "stt-m" and seen["language"] == "en"
    assert seen["response_format"] == "text"
    assert seen["file"] == ("speech.wav", b"WAVBYTES", "audio/wav")


def test_transcribe_accepts_an_object_with_text_and_a_language_argument():
    fake, seen = audio_client(transcription=SimpleNamespace(text=" hi "))
    assert ChatClient("u", "k", client=fake).transcribe("m", b"x", language="es") == "hi"
    assert seen["language"] == "es"


ERRORS = [
    (status_error(openai.RateLimitError, 429), RateLimited),
    (status_error(openai.InternalServerError, 503), TransientError),
    (openai.APIConnectionError(request=httpx2.Request("POST", "http://x")), TransientError),
    (status_error(openai.BadRequestError, 400), LLMError),
    (ValueError("x"), LLMError),
]


@pytest.mark.parametrize("error, expected", ERRORS)
def test_transcribe_maps_errors_like_the_chat_calls(error, expected):
    fake, _ = audio_client(error=error)
    with pytest.raises(expected) as raised:
        ChatClient("u", "k", client=fake).transcribe("m", b"x")
    assert type(raised.value) is expected


@pytest.mark.parametrize("error, expected", ERRORS)
@pytest.mark.parametrize("after_data", [False, True])
def test_speak_maps_errors_like_the_chat_calls(error, expected, after_data):
    chunks = [header() + pcm(0.1)] if after_data else []
    fake, _ = audio_client(Stream(chunks, error))
    with pytest.raises(expected) as raised:
        ChatClient("u", "k", client=fake).speak("m", "v", "hi")
    assert type(raised.value) is expected


def test_speak_measures_duration_from_the_sample_data_not_the_header():
    stream = Stream([header(data_size=0xFFFFFFFF) + pcm(1.5)])   # placeholder lengths
    fake, _ = audio_client(stream)
    wav, seconds, _, _ = ChatClient("u", "k", client=fake).speak("tts-m", "coral", "Hello")
    assert seconds == pytest.approx(1.5)
    assert stream.kwargs == {"model": "tts-m", "voice": "coral", "input": "Hello",
                             "response_format": "wav"}
    with wave.open(io.BytesIO(wav)) as w:       # re-wrapped with true lengths
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)
        assert w.getnframes() == int(1.5 * 24000)


def test_speak_reads_rate_channels_and_width_from_the_header():
    stream = Stream([header(rate=16000, channels=2, bits=16) + pcm(2, 16000, 2)])
    fake, _ = audio_client(stream)
    wav, seconds, _, _ = ChatClient("u", "k", client=fake).speak("m", "v", "x")
    assert seconds == pytest.approx(2.0)
    stream = Stream([header(rate=8000, channels=1, bits=8) + b"\x80" * 4000])
    fake, _ = audio_client(stream)
    assert ChatClient("u", "k", client=fake).speak("m", "v", "x")[1] == pytest.approx(0.5)


def test_speak_ignores_wrong_length_fields_and_cuts_to_whole_frames():
    stream = Stream([header(riff_size=0, data_size=3) + pcm(1) + b"\x07"])   # odd trailing byte
    fake, _ = audio_client(stream)
    wav, seconds, _, _ = ChatClient("u", "k", client=fake).speak("m", "v", "x")
    assert seconds == pytest.approx(1.0)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnframes() == 24000


def test_speak_skips_other_chunks_before_the_data_chunk():
    stream = Stream([header(extra_chunk=b"LIST" + struct.pack("<I", 3) + b"abc\x00") + pcm(1)])
    fake, _ = audio_client(stream)
    assert ChatClient("u", "k", client=fake).speak("m", "v", "x")[1] == pytest.approx(1.0)


def test_speak_works_when_the_header_arrives_split_across_chunks():
    data = header() + pcm(1)
    stream = Stream([data[:5], b"", data[5:30], data[30:]])
    fake, _ = audio_client(stream)
    assert ChatClient("u", "k", client=fake).speak("m", "v", "x")[1] == pytest.approx(1.0)


def test_speak_times_the_first_non_empty_chunk_and_the_end():
    data = header() + pcm(1)

    class Timed(Stream):
        def iter_bytes(self):
            yield b""
            yield data[:44]
            yield data[44:]

    seq = [10.0, 10.4, 12.5]
    clock = lambda: seq.pop(0)               # noqa: E731  start, first byte, end
    fake, _ = audio_client(Timed([]))
    _, _, first, total = ChatClient("u", "k", client=fake, clock=clock).speak("m", "v", "x")
    assert first == pytest.approx(0.4) and total == pytest.approx(2.5)


@pytest.mark.parametrize("bad", [
    b"", b"not a wav file at all", header(tag=3) + pcm(1), header()[:30],
    b"RIFF\x00\x00\x00\x00WAVE" + b"data" + struct.pack("<I", 4) + b"\x00" * 4,
])
def test_speak_rejects_audio_that_is_not_pcm_wav(bad):
    fake, _ = audio_client(Stream([bad]))
    with pytest.raises(LLMError) as raised:
        ChatClient("u", "k", client=fake).speak("m", "v", "x")
    assert str(raised.value) == "bad_audio"


def test_speak_rejects_a_wav_with_no_samples():
    fake, _ = audio_client(Stream([header()]))
    with pytest.raises(LLMError):
        ChatClient("u", "k", client=fake).speak("m", "v", "x")


def test_parse_wav_accepts_extensible_format_and_odd_sized_chunks():
    data = header(tag=0xFFFE, extra_chunk=b"junk" + struct.pack("<I", 1) + b"a\x00") + pcm(1)
    assert llm.parse_wav(data)[0:3] == (24000, 1, 2)
