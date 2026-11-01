"""Synchronous two-level thinking support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gemini_webapi_proxy.client.webapi import WebAPIClient
from gemini_webapi_proxy.schemas import ChatCompletionRequest, ChatMessage
from gemini_webapi_proxy.services import chat as chat_service


class _FakeNativeClient:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def generate_content(self, prompt: str, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(text="answer", images=[])


@pytest.mark.parametrize(("level", "extended"), [(1, False), (2, True)])
async def test_webapi_client_maps_thinking_level(level: int, extended: bool) -> None:
    client = object.__new__(WebAPIClient)
    native = _FakeNativeClient()
    client._client = native

    await client.generate_content("hello", files=None, model="gemini-3-flash", thinking_level=level)

    assert native.kwargs["extended_thinking"] is extended


async def test_webapi_client_defaults_to_extended_thinking() -> None:
    client = object.__new__(WebAPIClient)
    native = _FakeNativeClient()
    client._client = native

    await client.generate_content("hello", files=None, model="gemini-3-flash")

    assert native.kwargs["extended_thinking"] is True


@pytest.mark.parametrize("level", [0, 3, -1, True])
def test_chat_request_rejects_unsupported_thinking_level(level: int) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="gemini-3-flash",
            messages=[{"role": "user", "content": "hello"}],
            thinking_level=level,
        )


def test_chat_request_defaults_to_extended_thinking() -> None:
    request = ChatCompletionRequest(
        model="gemini-3-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert request.thinking_level == 2


@pytest.mark.parametrize("level", [1, 2])
async def test_chat_service_passes_thinking_level(monkeypatch, level: int) -> None:
    native = _FakeNativeClient()

    class _ProxyClient:
        async def generate_content(self, prompt: str, **kwargs):
            native.kwargs = kwargs
            return SimpleNamespace(text="answer", images=[])

    monkeypatch.setattr(chat_service, "get_client", lambda _settings: _async_value(_ProxyClient()))
    monkeypatch.setattr(chat_service.registry, "resolve_id", lambda model: model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: model)
    settings = SimpleNamespace(chat_timeout=5, http_proxy=None)

    await chat_service.create_chat_completion(
        settings,
        model="gemini-3-flash",
        messages=[ChatMessage(role="user", content="hello")],
        thinking_level=level,
    )

    assert native.kwargs["thinking_level"] == level


async def _async_value(value):
    return value
