"""Export concrete model-emitted text as downloadable UTF-8 files."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_webapi_proxy.routes.files import router as exports_router
from gemini_webapi_proxy.services.text_exports import (
    TextExport,
    TextExportStore,
    build_text_exports,
    store_text_exports,
    strip_file_tags,
    text_export_store,
)

MARKER = "[file-tag: code-generated-file-72329e5b-b612-4e7e-b5d8]"


def test_fenced_csv_exports_only_the_actual_csv_body() -> None:
    response = f"Your CSV file is ready.\n\n```csv\nname,score\nAda,98\nKen,91\n```\n{MARKER}"

    exports = build_text_exports(response)

    assert len(exports) == 1
    assert exports[0].name == "generated-1.csv"
    assert exports[0].mime_type.startswith("text/csv")
    assert exports[0].text == "name,score\nAda,98\nKen,91\n"
    assert "file-tag" not in exports[0].text


def test_explicit_filename_in_fence_is_preserved_safely() -> None:
    response = "```csv filename=people.csv\nname,age\nMina,27\n```"

    exports = build_text_exports(response)

    assert exports[0].name == "people.csv"


def test_fenced_markdown_document_exports_as_markdown() -> None:
    response = "```markdown\n# Notes\n\n- One\n- Two\n```"

    exports = build_text_exports(response)

    assert len(exports) == 1
    assert exports[0].name == "generated-1.md"
    assert exports[0].text == "# Notes\n\n- One\n- Two\n"


def test_multiple_recognized_fences_become_separate_exports() -> None:
    response = '```csv\na,b\n1,2\n```\n\n```json\n{"ok":true}\n```'

    exports = build_text_exports(response)

    assert [item.name for item in exports] == ["generated-1.csv", "generated-2.json"]
    assert exports[1].text == '{"ok":true}\n'


def test_raw_csv_export_is_inferred_without_fence() -> None:
    exports = build_text_exports("name,score\nAda,98\nKen,91")

    assert len(exports) == 1
    assert exports[0].name == "generated.csv"
    assert exports[0].text == "name,score\nAda,98\nKen,91\n"


def test_raw_tsv_export_is_inferred_without_fence() -> None:
    exports = build_text_exports("name\tscore\nAda\t98\nKen\t91")

    assert len(exports) == 1
    assert exports[0].name == "generated.tsv"


def test_markdown_document_exports_without_fence() -> None:
    exports = build_text_exports("# A short guide\n\n- first\n- second")

    assert len(exports) == 1
    assert exports[0].name == "generated.md"
    assert exports[0].text.endswith("\n")


def test_plain_prose_that_claims_a_file_exists_is_not_exported() -> None:
    exports = build_text_exports(f"Your CSV file is ready. I created it for you. {MARKER}")

    assert exports == []


def test_marker_alone_is_never_used_as_file_content() -> None:
    assert build_text_exports(MARKER) == []
    assert strip_file_tags(f"A file was created. {MARKER}") == "A file was created."


def test_marker_cleanup_preserves_tabs_used_by_tsv() -> None:
    source = f"name\tscore\nAda\t98\nKen\t91\n{MARKER}"

    assert strip_file_tags(source) == "name\tscore\nAda\t98\nKen\t91"
    assert build_text_exports(source)[0].text == "name\tscore\nAda\t98\nKen\t91\n"


def test_unlabelled_code_fence_is_not_guessed_as_a_file() -> None:
    assert build_text_exports("```\nprint('hello')\n```") == []


def test_store_descriptors_point_to_export_endpoint() -> None:
    text_export_store.clear()

    descriptors = store_text_exports("```csv\na,b\n1,2\n```")

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor["type"] == "text"
    assert descriptor["name"] == "generated-1.csv"
    assert descriptor["source"] == "assistant_text"
    assert str(descriptor["download_url"]).startswith("/openai/v1/exports/")
    assert text_export_store.get(str(descriptor["id"])) is not None


async def test_chat_completion_exports_model_text_and_removes_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_webapi_proxy.schemas import ChatMessage
    from gemini_webapi_proxy.services import chat as chat_service

    text_export_store.clear()

    class _Client:
        async def generate_content(self, prompt: str, **kwargs):
            return SimpleNamespace(
                text=f"Here is the CSV:\n\n```csv\nname,age\nMina,27\n```\n{MARKER}",
                images=[],
            )

    async def _client(_settings):
        return _Client()

    monkeypatch.setattr(chat_service, "get_client", _client)
    monkeypatch.setattr(chat_service.registry, "resolve_id", lambda model: model)
    monkeypatch.setattr(chat_service.registry, "resolve_runtime", lambda model: model)
    settings = SimpleNamespace(chat_timeout=5, http_proxy=None)

    response = await chat_service.create_chat_completion(
        settings,
        model="gemini-3-flash",
        messages=[ChatMessage(role="user", content="make a csv")],
    )

    message = response.choices[0].message
    assert "file-tag" not in message["content"]
    assert "attachments" not in message
    assert message["exports"][0]["name"] == "generated-1.csv"


def test_export_store_roundtrip_expiry_and_size_limit() -> None:
    store = TextExportStore(ttl_seconds=30, max_export_bytes=10, max_total_bytes=20)
    record = store.put(TextExport("a.txt", "text/plain; charset=utf-8", "hello"))

    assert record is not None
    assert store.get(record.export_id).data == b"hello"

    expired = TextExportStore(ttl_seconds=0)
    expired_record = expired.put(TextExport("a.txt", "text/plain", "hello"))
    assert expired_record is not None
    assert expired.get(expired_record.export_id) is None

    assert store.put(TextExport("big.txt", "text/plain", "12345678901")) is None


def test_export_store_sanitizes_filename() -> None:
    store = TextExportStore()
    record = store.put(TextExport("../../bad\nname.csv", "text/csv", "a,b\n1,2"))

    assert record is not None
    assert record.name == "bad_name.csv"


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(exports_router)
    return TestClient(app)


def test_download_text_export_returns_bytes_and_filename(client: TestClient) -> None:
    text_export_store.clear()
    descriptors = store_text_exports("```csv\na,b\n1,2\n```")
    export_id = str(descriptors[0]["id"])

    response = client.get(f"/openai/v1/exports/{export_id}")

    assert response.status_code == 200
    assert response.content == b"a,b\n1,2\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="generated-1.csv"' in response.headers["content-disposition"]


def test_download_unknown_or_invalid_export_returns_404(client: TestClient) -> None:
    assert client.get("/openai/v1/exports/deadbeef").status_code == 404
    assert client.get("/openai/v1/exports/..%2Fsecret").status_code in {404, 405}
