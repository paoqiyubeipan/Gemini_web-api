from __future__ import annotations

from types import SimpleNamespace

from gemini_webapi_proxy.client.base import ModelDescriptor
from gemini_webapi_proxy.client.registry import ModelRegistry
from gemini_webapi_proxy.routes import chat as chat_route
from gemini_webapi_proxy.schemas import ChatCompletionRequest, ChatMessage
from gemini_webapi_proxy.services import chat as chat_service


def test_chat_request_allows_omitting_model() -> None:
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
    assert request.model is None


def test_registry_prefers_flash_chat_model_over_pro() -> None:
    registry = ModelRegistry()
    registry.sync_from_client(
        [
            ModelDescriptor(name="gemini-3-pro", display_name="Pro", description="advanced"),
            ModelDescriptor(name="gemini-3-flash", display_name="Flash", description="fast"),
        ]
    )

    assert registry.pick_chat_model() == "gemini-3-flash"


def test_registry_prefers_standard_flash_over_flash_lite() -> None:
    registry = ModelRegistry()
    registry.sync_from_client(
        [
            ModelDescriptor(
                name="gemini-flash-lite", display_name="Flash Lite", description="fast"
            ),
            ModelDescriptor(name="gemini-flash", display_name="Flash", description="fast"),
        ]
    )

    assert registry.pick_chat_model() == "gemini-flash"


def test_registry_preserves_explicit_non_flash_model() -> None:
    registry = ModelRegistry()
    registry.sync_from_client(
        [
            ModelDescriptor(name="gemini-3-pro", display_name="Pro", description="advanced"),
            ModelDescriptor(name="gemini-3-flash", display_name="Flash", description="fast"),
        ]
    )

    assert registry.pick_chat_model("gemini-3-pro") == "gemini-3-pro"


async def test_chat_route_passes_unspecified_model_to_service(monkeypatch) -> None:
    captured: dict = {}

    async def fake_create_chat_completion(settings, *, model, messages, files, thinking_level):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(chat_route, "get_settings", object)
    monkeypatch.setattr(chat_route, "create_chat_completion", fake_create_chat_completion)
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

    assert await chat_route.chat_completions(request) == "ok"
    assert captured["model"] is None


async def test_chat_service_uses_registry_default_when_model_is_omitted(monkeypatch) -> None:
    captured: dict = {}

    class FakeClient:
        async def generate_content(self, prompt, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="answer", images=[])

    async def fake_get_client(_settings):
        return FakeClient()

    def pick_chat_model(requested):
        assert requested is None
        return "gemini-3-flash"

    monkeypatch.setattr(chat_service, "get_client", fake_get_client)
    monkeypatch.setattr(chat_service.registry, "pick_chat_model", pick_chat_model)
    monkeypatch.setattr(chat_service.registry, "resolve_id", lambda model: model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: f"runtime:{model}")

    response = await chat_service.create_chat_completion(
        SimpleNamespace(chat_timeout=5, http_proxy=None),
        model=None,
        messages=[ChatMessage(role="user", content="hello")],
    )

    assert captured["model"] == "runtime:gemini-3-flash"
    assert response.model == "gemini-3-flash"
