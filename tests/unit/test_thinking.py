"""Two-level proxy thinking control mapped to the SDK boolean switch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gemini_webapi import GeminiClient as SDKGeminiClient
from pydantic import ValidationError

from gemini_webapi_proxy.client.base import thinking_level_to_extended
from gemini_webapi_proxy.client.webapi import WebAPIClient
from gemini_webapi_proxy.schemas import ChatCompletionRequest, ChatMessage
from gemini_webapi_proxy.services import chat as chat_service

SDK_KWARGS = {"files", "model", "extended_thinking"}


class _StrictNativeClient:
    """Mimics the SDK: unknown kwargs leak into the HTTP layer and raise."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def generate_content(self, prompt: str, **kwargs):
        unexpected = set(kwargs) - SDK_KWARGS
        if unexpected:
            raise TypeError(
                f"AsyncSession.request() got an unexpected keyword argument {sorted(unexpected)[0]!r}"
            )
        self.kwargs = kwargs
        return SimpleNamespace(text="answer", images=[])


@pytest.mark.parametrize(("level", "extended"), [(1, False), (2, True)])
def test_thinking_level_maps_to_sdk_boolean(level: int, extended: bool) -> None:
    assert thinking_level_to_extended(level) is extended


@pytest.mark.parametrize("level", [0, 3, -1, True, False])
def test_thinking_level_rejects_unsupported_values(level: int) -> None:
    with pytest.raises(ValueError):
        thinking_level_to_extended(level)


@pytest.mark.parametrize(("level", "extended"), [(1, False), (2, True)])
async def test_webapi_client_maps_thinking_level(level: int, extended: bool) -> None:
    client = object.__new__(WebAPIClient)
    native = _StrictNativeClient()
    client._client = native

    await client.generate_content("hello", files=None, model="gemini-3-flash", thinking_level=level)

    assert native.kwargs["extended_thinking"] is extended


async def test_webapi_client_defaults_to_extended_thinking() -> None:
    client = object.__new__(WebAPIClient)
    native = _StrictNativeClient()
    client._client = native

    await client.generate_content("hello", files=None, model="gemini-3-flash")

    assert native.kwargs["extended_thinking"] is True


@pytest.mark.parametrize("enabled", [False, True])
async def test_sdk_accepts_and_forwards_extended_thinking(monkeypatch, enabled: bool) -> None:
    client = object.__new__(SDKGeminiClient)
    client.auto_close = False
    captured: dict = {}

    async def sync_activity() -> None:
        return None

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        yield object()

    monkeypatch.setattr(client, "_sync_activity", sync_activity, raising=False)
    monkeypatch.setattr(client, "_generate", fake_generate, raising=False)

    result = await client.generate_content("offline smoke test", extended_thinking=enabled)

    assert result is not None
    assert captured["extended_thinking"] is enabled


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


@pytest.mark.parametrize(("level", "extended"), [(1, False), (2, True)])
async def test_chat_service_sends_only_sdk_known_kwargs(monkeypatch, level, extended) -> None:
    native = _StrictNativeClient()

    class _ProxyClient:
        async def generate_content(self, prompt: str, **kwargs):
            return await native.generate_content(prompt, **kwargs)

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

    assert "thinking_level" not in native.kwargs
    assert native.kwargs["extended_thinking"] is extended


async def test_chat_service_defaults_to_extended_thinking(monkeypatch) -> None:
    native = _StrictNativeClient()

    class _ProxyClient:
        async def generate_content(self, prompt: str, **kwargs):
            return await native.generate_content(prompt, **kwargs)

    monkeypatch.setattr(chat_service, "get_client", lambda _settings: _async_value(_ProxyClient()))
    monkeypatch.setattr(chat_service.registry, "resolve_id", lambda model: model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: model)
    settings = SimpleNamespace(chat_timeout=5, http_proxy=None)

    await chat_service.create_chat_completion(
        settings,
        model="gemini-3-flash",
        messages=[ChatMessage(role="user", content="hello")],
    )

    assert native.kwargs["extended_thinking"] is True


async def _async_value(value):
    return value
