from __future__ import annotations

import base64
from types import SimpleNamespace

from fastapi.testclient import TestClient

from gemini_webapi_proxy.app import app

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_image_generation_returns_get_url_and_keeps_b64_json(monkeypatch) -> None:
    from gemini_webapi_proxy.services import chat as chat_service

    class _Client:
        async def generate_content(self, prompt: str, **kwargs):
            return SimpleNamespace(text="", images=[object()])

    async def _get_client(_settings):
        return _Client()

    async def _image_to_b64(_image, _settings, _client):
        return base64.b64encode(PNG).decode("ascii")

    monkeypatch.delenv("GOP_API_KEY", raising=False)
    monkeypatch.setattr(chat_service, "get_client", _get_client)
    monkeypatch.setattr(chat_service.registry, "pick_image_model", lambda model: "gemini-3-flash")
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: "runtime:flash")
    monkeypatch.setattr(chat_service, "sanitize_gemini_client_cookies", lambda _client: None)
    monkeypatch.setattr(chat_service, "_image_to_b64", _image_to_b64)

    with TestClient(app) as client:
        response = client.post(
            "/openai/v1/images/generations",
            json={"prompt": "a red apple", "response_format": "url"},
        )
        assert response.status_code == 200, response.text
        image = response.json()["data"][0]

        assert image["b64_json"] == base64.b64encode(PNG).decode("ascii")
        assert image["url"].startswith("http://testserver/openai/v1/images/")
        download = client.get(image["url"])

    assert download.status_code == 200
    assert download.headers["content-type"] == "image/png"
    assert download.headers["content-disposition"].startswith("attachment;")
    assert download.content == PNG
