"""Thin wrapper over the openai client for an OpenAI-compatible endpoint."""

from __future__ import annotations

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


class ChatClient:
    def __init__(self, base_url: str, api_key: str, client=None):
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=120, max_retries=0
        )

    def complete(self, model: str, messages: list[dict]) -> str:
        try:
            resp = self._client.chat.completions.create(model=model, messages=messages)
        except Exception as exc:
            raise _map_error(exc) from None
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        try:
            page = self._client.models.list()
        except Exception as exc:
            raise _map_error(exc) from None
        return [m.id for m in page.data]
