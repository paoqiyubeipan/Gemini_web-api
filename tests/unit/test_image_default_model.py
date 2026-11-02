from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from gemini_webapi_proxy.routes import images as image_route
from gemini_webapi_proxy.schemas import ImageGenerationRequest
from gemini_webapi_proxy.services import chat as chat_service


def test_image_request_allows_omitting_model_and_keeps_explicit_choice() -> None:
    omitted = ImageGenerationRequest(prompt="draw a bird")
    explicit = ImageGenerationRequest(model="gemini-2.5-pro-image", prompt="draw a bird")

    assert omitted.model is None
    assert explicit.model == "gemini-2.5-pro-image"


async def test_image_route_passes_omitted_model_to_service(monkeypatch) -> None:
    captured: dict = {}

    async def fake_create_image_generation(
        settings, *, model, prompt, n, size, response_format, reference_images
    ):
        captured.update(model=model, reference_images=reference_images)
        return {"created": 1, "data": []}

    monkeypatch.setattr(image_route, "get_settings", object)
    monkeypatch.setattr(image_route, "create_image_generation", fake_create_image_generation)
    request = ImageGenerationRequest(prompt="draw a bird")

    await image_route.image_generations(request, SimpleNamespace())

    assert captured == {"model": None, "reference_images": None}


async def test_image_service_uses_flash_default_when_model_is_omitted(monkeypatch) -> None:
    captured: dict = {}

    class FakeClient:
        async def generate_content(self, prompt, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="", images=[object()])

    async def fake_get_client(_settings):
        return FakeClient()

    async def fake_image_to_b64(_image, _settings, _client):
        return "cG5nLWJ5dGVz"

    def pick_image_model(requested):
        assert requested is None
        captured["selected_alias"] = "gemini-2.5-flash-image"
        return "gemini-2.5-flash-image"

    monkeypatch.setattr(chat_service, "get_client", fake_get_client)
    monkeypatch.setattr(chat_service.registry, "pick_image_model", pick_image_model)
    monkeypatch.setattr(
        chat_service.registry, "resolve_runtime", lambda model: "runtime:gemini-flash"
    )
    monkeypatch.setattr(chat_service, "sanitize_gemini_client_cookies", lambda _client: None)
    monkeypatch.setattr(chat_service, "_image_to_b64", fake_image_to_b64)

    response = await chat_service.create_image_generation(
        SimpleNamespace(image_timeout=5, http_proxy=None),
        model=None,
        prompt="draw a bird",
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    assert captured["selected_alias"] == "gemini-2.5-flash-image"
    assert captured["model"] == "runtime:gemini-flash"
    assert len(response.data) == 1


async def test_image_service_preserves_explicit_model(monkeypatch) -> None:
    captured: dict = {}

    class FakeClient:
        async def generate_content(self, prompt, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="", images=[object()])

    async def fake_get_client(_settings):
        return FakeClient()

    async def fake_image_to_b64(_image, _settings, _client):
        return "cG5nLWJ5dGVz"

    def pick_image_model(requested):
        captured["requested_model"] = requested
        return requested

    monkeypatch.setattr(chat_service, "get_client", fake_get_client)
    monkeypatch.setattr(chat_service.registry, "pick_image_model", pick_image_model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: f"runtime:{model}")
    monkeypatch.setattr(chat_service, "sanitize_gemini_client_cookies", lambda _client: None)
    monkeypatch.setattr(chat_service, "_image_to_b64", fake_image_to_b64)

    await chat_service.create_image_generation(
        SimpleNamespace(image_timeout=5, http_proxy=None),
        model="gemini-2.5-pro-image",
        prompt="draw a bird",
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    assert captured["requested_model"] == "gemini-2.5-pro-image"
    assert captured["model"] == "runtime:gemini-2.5-pro-image"


async def test_generated_image_is_saved_to_local_cache_with_sdk(monkeypatch) -> None:
    saved: dict = {}
    image_bytes = b"saved-image-bytes"

    class FakeGeneratedImage:
        url = "https://lh3.googleusercontent.com/gg-dl/generated"
        proxy = "initial-proxy"

        async def save(self, *, path, filename, full_size):
            saved.update(path=path, filename=filename, full_size=full_size)
            saved_path = Path(path) / filename
            saved_path.write_bytes(image_bytes)
            saved["saved_path"] = str(saved_path)
            return str(saved_path)

    settings = SimpleNamespace(http_proxy="http://proxy.local", cdn_direct=True)
    image = FakeGeneratedImage()

    encoded = await chat_service._image_to_b64(image, settings, object())

    assert base64.b64decode(encoded) == image_bytes
    assert Path(saved["path"]).name.startswith("gemini-generated-cache-")
    assert saved["filename"] == "generated.png"
    assert saved["full_size"] is True
    assert not Path(saved["saved_path"]).exists()
    assert image.proxy is None
