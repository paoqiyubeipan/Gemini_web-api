"""``GET /openai/v1/exports/{export_id}`` — download model-emitted text."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from gemini_webapi_proxy.auth import require_api_key_if_configured
from gemini_webapi_proxy.services.text_exports import text_export_store

router = APIRouter(
    prefix="/openai/v1",
    tags=["files"],
    dependencies=[Depends(require_api_key_if_configured)],
)

_FILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_FALLBACK_NAME = "generated-text"


def _content_disposition(name: str | None, file_id: str) -> str:
    if name:
        cleaned = re.sub(r'[\r\n"\\]', "_", name).strip() or _FALLBACK_NAME
    else:
        cleaned = f"{_FALLBACK_NAME}-{file_id}.txt"
    return f'attachment; filename="{cleaned}"'


@router.get("/exports/{export_id}", responses={200: {"content": {"text/plain": {}}}})
async def download_text_export(export_id: str) -> Response:
    if not _FILE_ID_RE.match(export_id):
        raise HTTPException(status_code=404, detail="Export not found")
    record = text_export_store.get(export_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Text export not found or expired. Exports are created only from "
                "text actually present in the assistant response."
            ),
        )
    return Response(
        content=record.data,
        media_type=record.mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": _content_disposition(record.name, export_id),
            "Content-Length": str(record.size_bytes),
            "Cache-Control": "private, max-age=0, no-store",
        },
    )
