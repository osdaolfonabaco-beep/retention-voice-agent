"""Thin wrapper over the openai client for an OpenAI-compatible endpoint."""

from __future__ import annotations

import io
import struct
import time
import wave

import openai


class LLMError(Exception):
    """Any failure that is neither a rate limit nor transient.

    str(exc) is safe to log: exception class name and HTTP status only.
    `detail` carries the provider's message and may echo request content.
    """

    def __init__(self, kind: str, status: int | None = None, detail: str = ""):
        self.kind = kind
        self.status = status
        self.detail = detail
        super().__init__(f"{kind} {status}" if status else kind)


class RateLimited(LLMError):
    """HTTP 429."""


class TransientError(LLMError):
    """Timeout, connection failure or 5xx."""


def _map_error(exc: Exception) -> LLMError:
    kind = type(exc).__name__
    detail = str(exc)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 429:
            return RateLimited(kind, status, detail)
        if status >= 500:
            return TransientError(kind, status, detail)
        return LLMError(kind, status, detail)
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return TransientError(kind, None, detail)
    return LLMError(kind, None, detail)


def parse_wav(data: bytes) -> tuple[int, int, int, bytes]:
    """(sample rate, channels, sample width in bytes, sample data) of a WAV byte string.

    The header's length fields are ignored: a streamed response may carry placeholders
    there. The sample data is everything after the `data` chunk header, cut to whole
    frames. Raises LLMError("bad_audio") if the bytes are not 16-bit-style PCM WAV.
    """
    bad = LLMError("bad_audio")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise bad
    fmt, pos = None, 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if chunk_id == b"data":
            if fmt is None:
                raise bad
            rate, channels, width = fmt
            frame = channels * width
            samples = data[body:]
            return rate, channels, width, samples[:len(samples) - len(samples) % frame]
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise bad
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[body:body + 16])
            if tag not in (1, 0xFFFE) or channels < 1 or rate < 1 or bits < 8 or bits % 8:
                raise bad
            fmt = (rate, channels, bits // 8)
        pos = body + size + (size & 1)
    raise bad


def wrap_wav(rate: int, channels: int, width: int, samples: bytes) -> bytes:
    """A clean WAV byte string with correct length fields."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(samples)
    return buf.getvalue()


class ChatClient:
    def __init__(self, base_url: str, api_key: str, client=None, clock=time.perf_counter):
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=120, max_retries=0
        )
        self._clock = clock

    def complete(self, model: str, messages: list[dict]) -> str:
        try:
            resp = self._client.chat.completions.create(model=model, messages=messages)
        except Exception as exc:
            raise _map_error(exc) from None
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    def complete_with_usage(self, model: str, messages: list[dict]) -> tuple[str, dict]:
        """Like `complete`, also returning {"input_tokens", "output_tokens"} (None if unreported)."""
        try:
            resp = self._client.chat.completions.create(model=model, messages=messages)
        except Exception as exc:
            raise _map_error(exc) from None
        usage = getattr(resp, "usage", None)
        counts = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
        if not resp.choices:
            return "", counts
        return resp.choices[0].message.content or "", counts

    def chat_with_tools(self, model: str, messages: list[dict],
                        tools: list[dict] | None = None,
                        reasoning_effort: str | None = None) -> tuple[dict, dict]:
        """One chat call that may ask for tools.

        Returns ({"content": str, "tool_calls": [{"id", "name", "arguments"}]}, usage),
        usage as in `complete_with_usage`. `reasoning_effort` is sent only when not None.
        """
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _map_error(exc) from None
        usage = getattr(resp, "usage", None)
        counts = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
        message = {"content": "", "tool_calls": []}
        if resp.choices:
            raw = resp.choices[0].message
            message["content"] = raw.content or ""
            message["tool_calls"] = [
                {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
                for call in (getattr(raw, "tool_calls", None) or [])
            ]
        return message, counts

    def transcribe(self, model: str, wav_bytes: bytes, language: str = "en") -> str:
        """Speech to text for one WAV recording."""
        try:
            resp = self._client.audio.transcriptions.create(
                model=model, file=("speech.wav", wav_bytes, "audio/wav"),
                language=language, response_format="text")
        except Exception as exc:
            raise _map_error(exc) from None
        return (resp if isinstance(resp, str) else getattr(resp, "text", "") or "").strip()

    def speak(self, model: str, voice: str, text: str) -> tuple[bytes, float, float, float]:
        """Text to speech, streamed as WAV.

        Returns (clean WAV bytes, audio seconds, seconds to the first audio byte, total
        seconds). The duration comes from the length of the sample data, not from the header.
        """
        chunks: list[bytes] = []
        first = None
        start = self._clock()
        try:
            with self._client.audio.speech.with_streaming_response.create(
                    model=model, voice=voice, input=text, response_format="wav") as resp:
                for chunk in resp.iter_bytes():
                    if chunk:
                        if first is None:
                            first = self._clock() - start
                        chunks.append(chunk)
        except Exception as exc:
            raise _map_error(exc) from None
        total = self._clock() - start
        rate, channels, width, samples = parse_wav(b"".join(chunks))
        if not samples:
            raise LLMError("bad_audio")
        seconds = len(samples) / (rate * channels * width)
        return wrap_wav(rate, channels, width, samples), seconds, (total if first is None else first), total

    def list_models(self) -> list[str]:
        try:
            page = self._client.models.list()
        except Exception as exc:
            raise _map_error(exc) from None
        return [m.id for m in page.data]
