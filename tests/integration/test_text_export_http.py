"""End-to-end HTTP check of text exports using a real Gemini reply sample.

The sample in ``tests/fixtures/gemini_csv_reply.txt`` was captured from an
actual Gemini Web answer (prompt: "Output only one fenced code block labeled csv
with exactly 3 fruit rows: columns fruit,price"). The SDK client is stubbed so
the test runs without cookies, but the response text is genuine model output.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gemini_webapi_proxy.app import app

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_csv_reply.txt"


def _stub_client(reply: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from gemini_webapi_proxy.services import chat as chat_service

    class _Client:
        async def generate_content(self, prompt: str, **kwargs):
            return SimpleNamespace(text=reply, images=[])

    async def _get_client(_settings):
        return _Client()

    monkeypatch.setattr(chat_service, "get_client", _get_client)
    monkeypatch.setattr(chat_service.registry, "resolve_id", lambda model: model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: model)


def test_chat_response_exports_downloadable_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = FIXTURE.read_text(encoding="utf-8")
    _stub_client(reply, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "give me a csv"}],
            },
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]

        assert "file-tag" not in message["content"]
        export = message["exports"][0]
        assert export["name"] == "generated-1.csv"
        assert export["mime_type"].startswith("text/csv")
        assert export["source"] == "assistant_text"

        download = client.get(export["download_url"])

    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/csv")
    assert 'filename="generated-1.csv"' in download.headers["content-disposition"]
    assert download.content == b"fruit,price\napple,1.50\nbanana,0.75\norange,2.00\n"


def test_unfenced_csv_reply_is_also_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = "fruit,price\napple,1.50\nbanana,0.75\norange,2.00\n"
    _stub_client(reply, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "give me a csv"}],
            },
        )
        assert response.status_code == 200, response.text
        export = response.json()["choices"][0]["message"]["exports"][0]
        download = client.get(export["download_url"])

    assert export["name"] == "generated.csv"
    assert download.content == b"fruit,price\napple,1.50\nbanana,0.75\norange,2.00\n"


def test_prose_only_reply_has_no_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_client("I created test.csv for you. Enjoy!", monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "give me a csv"}],
            },
        )

    assert response.status_code == 200, response.text
    assert "exports" not in response.json()["choices"][0]["message"]
