"""Downloadable text exports derived from assistant response text.

This is intentionally *not* a Gemini Web attachment downloader. The proxy
exports text that is present in the model's response (for example a fenced CSV
or Markdown document). A ``[file-tag: ...]`` marker is removed from the visible
text, but is never treated as file content or as a remote file identifier.
"""

from __future__ import annotations

import csv
import io
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import PurePath

_FILE_TAG_RE = re.compile(r"\[file-tag:\s*[^\[\]]+?\s*\]", re.IGNORECASE)


def strip_file_tags(text: str) -> str:
    """Remove Gemini's internal file markers; they are not file contents."""
    return _FILE_TAG_RE.sub("", text).strip()


# Only formats where text is itself a meaningful file representation.
_LANGUAGES: dict[str, tuple[str, str]] = {
    "csv": ("csv", "text/csv; charset=utf-8"),
    "tsv": ("tsv", "text/tab-separated-values; charset=utf-8"),
    "markdown": ("md", "text/markdown; charset=utf-8"),
    "md": ("md", "text/markdown; charset=utf-8"),
    "json": ("json", "application/json; charset=utf-8"),
    "txt": ("txt", "text/plain; charset=utf-8"),
    "text": ("txt", "text/plain; charset=utf-8"),
    "html": ("html", "text/html; charset=utf-8"),
    "xml": ("xml", "application/xml; charset=utf-8"),
    "yaml": ("yaml", "application/yaml; charset=utf-8"),
    "yml": ("yaml", "application/yaml; charset=utf-8"),
    "sql": ("sql", "text/plain; charset=utf-8"),
    "python": ("py", "text/x-python; charset=utf-8"),
    "py": ("py", "text/x-python; charset=utf-8"),
    "javascript": ("js", "text/javascript; charset=utf-8"),
    "js": ("js", "text/javascript; charset=utf-8"),
    "typescript": ("ts", "text/plain; charset=utf-8"),
    "ts": ("ts", "text/plain; charset=utf-8"),
    "bash": ("sh", "text/x-shellscript; charset=utf-8"),
    "sh": ("sh", "text/x-shellscript; charset=utf-8"),
    "toml": ("toml", "application/toml; charset=utf-8"),
}
_EXTENSIONS: dict[str, str] = {
    "csv": "text/csv; charset=utf-8",
    "tsv": "text/tab-separated-values; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "markdown": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "htm": "text/html; charset=utf-8",
    "xml": "application/xml; charset=utf-8",
    "yaml": "application/yaml; charset=utf-8",
    "yml": "application/yaml; charset=utf-8",
    "sql": "text/plain; charset=utf-8",
    "py": "text/x-python; charset=utf-8",
    "js": "text/javascript; charset=utf-8",
    "ts": "text/plain; charset=utf-8",
    "sh": "text/x-shellscript; charset=utf-8",
    "toml": "application/toml; charset=utf-8",
    "ini": "text/plain; charset=utf-8",
}

_FENCE_RE = re.compile(
    r"(?m)^(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)(?:\n)?^(?P=fence)[ \t]*$",
    re.DOTALL,
)
_FILENAME_RE = re.compile(
    r"(?<![\w./-])(?P<name>[\w.-]{1,100}\.(?:csv|tsv|md|markdown|json|txt|html?|xml|yaml|yml|sql|py|js|ts|sh|toml|ini))\b",
    re.IGNORECASE,
)

_DEFAULT_TTL_SECONDS = 3600.0
_MAX_EXPORT_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 256


@dataclass(frozen=True)
class TextExport:
    """An exportable text artifact parsed from a model response."""

    name: str
    mime_type: str
    text: str


@dataclass(frozen=True)
class StoredTextExport:
    """UTF-8 bytes retained temporarily for a download request."""

    export_id: str
    data: bytes
    name: str
    mime_type: str
    expires_at: float

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass
class TextExportStore:
    """Thread-safe, bounded in-memory store for generated text exports."""

    ttl_seconds: float = _DEFAULT_TTL_SECONDS
    max_export_bytes: int = _MAX_EXPORT_BYTES
    max_total_bytes: int = _MAX_TOTAL_BYTES
    max_entries: int = _MAX_ENTRIES
    _items: dict[str, StoredTextExport] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def put(self, artifact: TextExport) -> StoredTextExport | None:
        data = artifact.text.encode("utf-8")
        if not data or len(data) > self.max_export_bytes or len(data) > self.max_total_bytes:
            return None
        record = StoredTextExport(
            export_id=uuid.uuid4().hex,
            data=data,
            name=_safe_filename(artifact.name),
            mime_type=artifact.mime_type,
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        with self._lock:
            self._purge_expired_locked()
            while self._items and (
                len(self._items) >= self.max_entries
                or sum(item.size_bytes for item in self._items.values()) + record.size_bytes
                > self.max_total_bytes
            ):
                oldest_id = min(self._items, key=lambda key: self._items[key].expires_at)
                del self._items[oldest_id]
            self._items[record.export_id] = record
        return record

    def get(self, export_id: str) -> StoredTextExport | None:
        with self._lock:
            self._purge_expired_locked()
            return self._items.get(export_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, item in self._items.items() if item.expires_at <= now]
        for key in expired:
            del self._items[key]


text_export_store = TextExportStore()


def build_text_exports(text: str) -> list[TextExport]:
    """Extract only concrete, model-emitted text artifacts from a response.

    Explicitly labelled fenced blocks are the strongest signal and may produce
    multiple files. When no such block exists, complete raw CSV/TSV data or a
    clearly structured Markdown document can still be exported. Plain prose
    that merely says a file was created is not exported as a substitute file.
    """
    visible_text = strip_file_tags(text).strip()
    if not visible_text:
        return []

    artifacts: list[TextExport] = []
    for match in _FENCE_RE.finditer(visible_text):
        info = match.group("info").strip()
        body = strip_file_tags(match.group("body")).strip("\n\r")
        if not body.strip():
            continue
        artifact = _fenced_artifact(info, body, len(artifacts) + 1)
        if artifact is not None:
            artifacts.append(artifact)
    if artifacts:
        return artifacts

    if _looks_like_delimited_text(visible_text, ","):
        return [TextExport("generated.csv", _EXTENSIONS["csv"], _with_final_newline(visible_text))]
    if _looks_like_delimited_text(visible_text, "\t"):
        return [TextExport("generated.tsv", _EXTENSIONS["tsv"], _with_final_newline(visible_text))]
    if _looks_like_markdown_document(visible_text):
        return [TextExport("generated.md", _EXTENSIONS["md"], _with_final_newline(visible_text))]
    return []


def store_text_exports(text: str) -> list[dict[str, object]]:
    """Create download records and return public descriptors for chat output."""
    descriptors: list[dict[str, object]] = []
    for artifact in build_text_exports(text):
        record = text_export_store.put(artifact)
        if record is None:
            continue
        descriptors.append(
            {
                "id": record.export_id,
                "type": "text",
                "name": record.name,
                "mime_type": record.mime_type,
                "size_bytes": record.size_bytes,
                "download_url": f"/openai/v1/exports/{record.export_id}",
                "source": "assistant_text",
            }
        )
    return descriptors


def _fenced_artifact(info: str, body: str, index: int) -> TextExport | None:
    filename_match = _FILENAME_RE.search(info)
    if filename_match:
        name = _safe_filename(filename_match.group("name"))
        extension = PurePath(name).suffix.lower().lstrip(".")
        mime_type = _EXTENSIONS.get(extension)
        if mime_type is None:
            return None
    else:
        language = info.split(maxsplit=1)[0].lower() if info else ""
        format_info = _LANGUAGES.get(language)
        if format_info is None:
            return None
        extension, mime_type = format_info
        name = f"generated-{index}.{extension}"
    cleaned = strip_file_tags(body).strip("\n\r")
    if not cleaned.strip():
        return None
    return TextExport(name=name, mime_type=mime_type, text=_with_final_newline(cleaned))


def _looks_like_delimited_text(text: str, delimiter: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or any(line.lstrip().startswith(("#", ">", "- ", "* ")) for line in lines):
        return False
    try:
        rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    except csv.Error:
        return False
    if len(rows) < 2:
        return False
    width = len(rows[0])
    return width >= 2 and all(
        len(row) == width and any(cell.strip() for cell in row) for row in rows
    )


def _looks_like_markdown_document(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if any(re.match(r"^#{1,6}\s+\S", line) for line in lines):
        return True
    if sum(bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s+)", line)) for line in lines) >= 2:
        return True
    return any("|" in line for line in lines) and any(
        re.match(r"^\|?\s*:?-{3,}", line) for line in lines
    )


def _with_final_newline(text: str) -> str:
    return text.rstrip("\r\n") + "\n"


def _safe_filename(name: str) -> str:
    # A basename prevents directory traversal; strip header-injection characters.
    cleaned = PurePath(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("._")
    return cleaned[:120] or "generated.txt"
