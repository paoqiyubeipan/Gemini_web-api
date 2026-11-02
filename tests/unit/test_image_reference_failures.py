"""Reference-image failures: quota, expired session, and real refusals.

These tests pin the classification that the 2026-11-02 reference-image run
needed: a spent image quota must not look like a policy refusal (429), and a
signed-out Gemini session must not look like "cannot use the reference image"
(401).  A healthy session with a genuine refusal still answers 403.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gemini_webapi.constants import AccountStatus

from gemini_webapi_proxy.config import Settings
from gemini_webapi_proxy.errors import CookieExpiredError
from gemini_webapi_proxy.routes._errors import map_api_error
from gemini_webapi_proxy.services import chat as chat_service

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

QUOTA_REPLY = (
    "You've reached your limit for generating images. Your limit will reset on 3 November."
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(image_timeout=5, http_proxy=None, cookie_source="env")


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, client) -> None:
    async def fake_get_client(_settings):
        return client

    monkeypatch.setattr(chat_service, "get_client", fake_get_client)
    monkeypatch.setattr(
        chat_service.registry, "pick_image_model", lambda requested: "gemini-2.5-flash-image"
    )
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: "gemini-flash")
    monkeypatch.setattr(chat_service, "sanitize_gemini_client_cookies", lambda _client: None)


async def _generate(monkeypatch: pytest.MonkeyPatch, client, *, refs=None):
    _patch_pipeline(monkeypatch, client)
    return await chat_service.create_image_generation(
        _settings(),
        model=None,
        prompt="restyle this mug",
        n=1,
        size="1024x1024",
        response_format="b64_json",
        reference_images=refs,
    )


async def test_expired_session_is_reported_as_401(monkeypatch) -> None:
    class FakeClient:
        account_status = AccountStatus.UNAUTHENTICATED

        async def generate_content(self, prompt, **kwargs):
            return SimpleNamespace(text="I'm not able to create that image right now.", images=[])

    with pytest.raises(CookieExpiredError) as excinfo:
        await _generate(monkeypatch, FakeClient(), refs=[PNG_DATA_URL])

    assert excinfo.value.http_status == 401
    resp = await map_api_error(excinfo.value, Settings())
    assert resp.status_code == 401
    assert b"Upstream refused" not in resp.body
    assert b"cookies" in resp.body.lower()


async def test_quota_reply_is_reported_as_429(monkeypatch) -> None:
    class FakeClient:
        account_status = AccountStatus.AVAILABLE

        async def generate_content(self, prompt, **kwargs):
            return SimpleNamespace(text=QUOTA_REPLY, images=[])

    with pytest.raises(RuntimeError) as excinfo:
        await _generate(monkeypatch, FakeClient(), refs=[PNG_DATA_URL])

    resp = await map_api_error(excinfo.value, Settings())
    assert resp.status_code == 429
    body = json.loads(resp.body)
    assert "limit" in body["error"]["message"].lower()
    assert "sign in to gemini.google.com" not in body["error"]["message"].lower()


async def test_real_refusal_still_maps_to_403(monkeypatch) -> None:
    class FakeClient:
        account_status = AccountStatus.AVAILABLE

        async def generate_content(self, prompt, **kwargs):
            return SimpleNamespace(
                text="Image creation isn't available for your account.", images=[]
            )

    with pytest.raises(RuntimeError) as excinfo:
        await _generate(monkeypatch, FakeClient(), refs=[PNG_DATA_URL])

    resp = await map_api_error(excinfo.value, Settings())
    assert resp.status_code == 403


async def test_reference_images_are_passed_as_files_and_cleaned_up(monkeypatch) -> None:
    seen: dict = {}

    class FakeClient:
        account_status = AccountStatus.AVAILABLE

        async def generate_content(self, prompt, **kwargs):
            files = kwargs.get("files") or []
            seen["files"] = [Path(f) for f in files]
            seen["exists"] = [Path(f).is_file() for f in files]
            seen["tmpdir"] = Path(files[0]).parent if files else None
            return SimpleNamespace(text="", images=[object()])

    async def fake_image_to_b64(_image, _settings, _client):
        return "cG5nLWJ5dGVz"

    monkeypatch.setattr(chat_service, "_image_to_b64", fake_image_to_b64)

    response = await _generate(monkeypatch, FakeClient(), refs=[PNG_DATA_URL])

    assert len(response.data) == 1
    assert seen["exists"] == [True]
    assert seen["files"][0].name == "ref_0.png"
    assert seen["tmpdir"].name.startswith("gemini-ref-")
    assert not seen["tmpdir"].exists()


async def test_no_image_reply_logs_upstream_text_and_reference_count(monkeypatch, caplog) -> None:
    class FakeClient:
        account_status = AccountStatus.AVAILABLE

        async def generate_content(self, prompt, **kwargs):
            return SimpleNamespace(text="I can't generate that for you.", images=[])

    with (
        caplog.at_level("WARNING", logger="gemini_webapi_proxy.services.chat"),
        pytest.raises(RuntimeError),
    ):
        await _generate(monkeypatch, FakeClient(), refs=[PNG_DATA_URL])

    logged = [r.getMessage() for r in caplog.records if "returned no image" in r.getMessage()]
    assert logged
    assert "reference_images=1" in logged[0]
    assert "I can't generate that for you." in logged[0]
    assert "UNAUTHENTICATED" not in logged[0]
