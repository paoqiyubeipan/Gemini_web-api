"""Tests for the dedicated chat file-input field."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gemini_webapi_proxy.schemas import ChatCompletionRequest, ChatMessage
from gemini_webapi_proxy.services import chat as chat_service


def test_chat_request_accepts_files_and_single_file_alias() -> None:
    url = "data:application/pdf;base64,JVBERi0xLjQ="
    canonical = ChatCompletionRequest(
        model="gemini-3-flash",
        messages=[{"role": "user", "content": "Summarize this."}],
        files=[url],
    )
    alias = ChatCompletionRequest(
        model="gemini-3-flash",
        messages=[{"role": "user", "content": "Summarize this."}],
        file=url,
    )

    assert canonical.files == [url]
    assert alias.files == url


def test_chat_request_rejects_server_local_file_paths() -> None:
    with pytest.raises(ValidationError, match=r"data URLs or HTTP\(S\) URLs"):
        ChatCompletionRequest(
            model="gemini-3-flash",
            messages=[{"role": "user", "content": "Read this."}],
            files="/etc/passwd",
        )


async def test_chat_service_materializes_dedicated_file_input(monkeypatch, tmp_path: Path) -> None:
    file_url = "data:application/pdf;base64,JVBERi0xLjQ="
    materialized = tmp_path / "ref_0.pdf"
    received_refs: list[str] = []
    native_kwargs: dict = {}

    async def materialize(refs: list[str], *, proxy: str | None = None) -> list[Path]:
        received_refs.extend(refs)
        return [materialized]

    class _ProxyClient:
        async def generate_content(self, prompt: str, **kwargs):
            native_kwargs.update(kwargs)
            return SimpleNamespace(text="Summary", images=[])

    monkeypatch.setattr(chat_service, "materialize_reference_images", materialize)
    monkeypatch.setattr(chat_service, "get_client", lambda _settings: _async_value(_ProxyClient()))
    monkeypatch.setattr(chat_service.registry, "resolve_id", lambda model: model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: model)
    settings = SimpleNamespace(chat_timeout=5, http_proxy=None)

    await chat_service.create_chat_completion(
        settings,
        model="gemini-3-flash",
        messages=[ChatMessage(role="user", content="Summarize this.")],
        files=[file_url],
    )

    assert received_refs == [file_url]
    assert native_kwargs["files"] == [materialized]


async def _async_value(value):
    return value
