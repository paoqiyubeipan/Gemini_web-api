"""Temporary, bounded storage for generated images exposed by direct GET URLs."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

_DEFAULT_TTL_SECONDS = 3600.0
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_ENTRIES = 256


@dataclass(frozen=True)
class StoredGeneratedImage:
    """One generated image retained for a short-lived download URL."""

    image_id: str
    data: bytes
    mime_type: str
    expires_at: float

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass
class GeneratedImageStore:
    """Thread-safe, bounded in-memory store; entries expire after one hour."""

    ttl_seconds: float = _DEFAULT_TTL_SECONDS
    max_image_bytes: int = _MAX_IMAGE_BYTES
    max_total_bytes: int = _MAX_TOTAL_BYTES
    max_entries: int = _MAX_ENTRIES
    _items: dict[str, StoredGeneratedImage] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def put(self, data: bytes) -> StoredGeneratedImage | None:
        """Store image bytes and return a random unguessable download id."""
        if not data or len(data) > self.max_image_bytes or len(data) > self.max_total_bytes:
            return None
        record = StoredGeneratedImage(
            image_id=uuid.uuid4().hex,
            data=bytes(data),
            mime_type=_image_mime_type(data),
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
            self._items[record.image_id] = record
        return record

    def get(self, image_id: str) -> StoredGeneratedImage | None:
        """Get a stored image by its random id, if it has not expired."""
        if len(image_id) != 32 or any(char not in "0123456789abcdef" for char in image_id):
            return None
        with self._lock:
            self._purge_expired_locked()
            return self._items.get(image_id)

    def clear(self) -> None:
        """Clear stored images; primarily useful for tests and application shutdown."""
        with self._lock:
            self._items.clear()

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, item in self._items.items() if item.expires_at <= now]
        for key in expired:
            del self._items[key]


def _image_mime_type(data: bytes) -> str:
    """Determine a safe content type from common generated-image signatures."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in {b"avif", b"avis"}:
        return "image/avif"
    return "application/octet-stream"


generated_image_store = GeneratedImageStore()
