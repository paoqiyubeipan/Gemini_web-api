from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

from gemini_webapi.constants import AccountStatus
from gemini_webapi.types.image import GeneratedImage

from gemini_webapi_proxy.client.base import thinking_level_to_extended
from gemini_webapi_proxy.client.pool import get_client, registry
from gemini_webapi_proxy.config import Settings
from gemini_webapi_proxy.downloaders.image_export import (
    _cdn_proxy,
    sanitize_gemini_client_cookies,
)
from gemini_webapi_proxy.errors import CookieExpiredError, ImageExportError
from gemini_webapi_proxy.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    ImageGenerationData,
    ImageGenerationResponse,
)
from gemini_webapi_proxy.services.generated_images import generated_image_store
from gemini_webapi_proxy.services.text_exports import store_text_exports, strip_file_tags
from gemini_webapi_proxy.utils.media import materialize_reference_images, normalize_image_field
from gemini_webapi_proxy.utils.upstream_refusal import looks_like_quota_message

logger = logging.getLogger(__name__)

QUOTA_PATTERNS = (
    r"limit resets",
    r"usage in settings",
    r"额度",
    r"重置",
    r"使用情况",
)


def _quota_message(text: str) -> bool:
    """True if the reply is a quota / usage-limit notice (see the shared helper)."""
    if looks_like_quota_message(text):
        return True
    blob = text.lower()
    return any(re.search(p, blob, re.I) for p in QUOTA_PATTERNS)


def _session_unauthenticated(client) -> bool:
    """True when the SDK has marked the Gemini session as signed out.

    Google answers such a session with short "I don't have access" style text.
    That is an authentication problem, not a policy refusal, so the caller must
    surface it as 401 instead of 403.
    """
    return getattr(client, "account_status", None) == AccountStatus.UNAUTHENTICATED


def _messages_to_prompt_and_refs(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    chunks: list[str] = []
    refs: list[str] = []
    for msg in messages:
        role = msg.role.upper()
        if isinstance(msg.content, str):
            chunks.append(f"{role}: {msg.content}")
            continue
        text_parts: list[str] = []
        for part in msg.content:
            ptype = (part.get("type") or "").lower()
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype in {"image_url", "input_image"}:
                url_part = part.get("image_url") or {}
                url = url_part.get("url") if isinstance(url_part, dict) else str(url_part)
                if url:
                    refs.append(url)
        chunks.append(f"{role}: {' '.join(text_parts)}")
    return "\n\n".join(chunks), refs


def _image_prompt(prompt: str, size: str | None, *, has_reference: bool) -> str:
    if has_reference:
        body = (
            "Use the attached reference image(s) as visual reference (character, style, composition, or subject). "
            "Generate a new image based on the reference and the prompt below. "
            "Return the generated image, not only text.\n\n"
            f"Prompt: {prompt.strip()}"
        )
    else:
        body = (
            "Generate an image from this prompt. Return the generated image, not only text.\n\n"
            f"Prompt: {prompt.strip()}"
        )
    if size:
        body += f"\nRequested size/aspect: {size}"
    return body


async def _image_to_b64(image: GeneratedImage, settings: Settings, _client) -> str:
    """Save a generated image locally through gemini-webapi and return base64 bytes."""
    if image.url:
        image.proxy = _cdn_proxy(settings, image.url)

    try:
        with tempfile.TemporaryDirectory(prefix="gemini-generated-cache-") as cache_dir:
            saved_path = await image.save(
                path=cache_dir,
                filename="generated.png",
                full_size=True,
            )
            image_bytes = Path(saved_path).read_bytes()
            if not image_bytes:
                raise ValueError("SDK saved an empty generated image")
    except Exception as exc:
        raise ImageExportError(
            f"Unable to save Gemini-generated image with gemini-webapi: {exc}",
            http_status=502,
        ) from exc

    return base64.b64encode(image_bytes).decode("ascii")


async def create_chat_completion(
    settings: Settings,
    *,
    model: str | None,
    messages: list[ChatMessage],
    files: str | list[str] | None = None,
    thinking_level: Literal[1, 2] = 2,
) -> ChatCompletionResponse:
    client = await get_client(settings)
    model = registry.pick_chat_model(model)
    resolved = registry.resolve_id(model)
    runtime = registry.resolve_runtime(resolved)
    prompt, msg_refs = _messages_to_prompt_and_refs(messages)
    file_refs = [files] if isinstance(files, str) else list(files or [])
    file_refs = [ref.strip() for ref in file_refs if ref.strip()]
    ref_paths: list[Path] = []
    all_refs = msg_refs + file_refs
    if all_refs:
        ref_paths = await materialize_reference_images(all_refs, proxy=settings.http_proxy)
    try:
        extended_thinking = thinking_level_to_extended(thinking_level)
        output = await asyncio.wait_for(
            client.generate_content(
                prompt,
                files=ref_paths or None,
                model=runtime,
                extended_thinking=extended_thinking,
            ),
            timeout=settings.chat_timeout,
        )
        text = (output.text or "").strip()
        images = output.images or []
        if _quota_message(text) and not images:
            raise RuntimeError(text)
        exports = store_text_exports(text)
        visible_text = strip_file_tags(text)
        content: str | list[dict] = visible_text
        if images and not text:
            content = text
        payload: dict = {"role": "assistant", "content": content}
        if images:
            payload["images"] = [{"url": img.url} for img in images]
        if exports:
            payload["exports"] = exports
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=resolved,
            choices=[ChatCompletionChoice(message=payload)],
        )
    finally:
        if ref_paths:
            tmp_root = ref_paths[0].parent
            if tmp_root.name.startswith("gemini-ref-"):
                for p in ref_paths:
                    p.unlink(missing_ok=True)
                tmp_root.rmdir()


async def create_image_generation(
    settings: Settings,
    *,
    model: str | None,
    prompt: str,
    n: int,
    size: str | None,
    response_format: str | None,
    reference_images: list[str] | None = None,
) -> ImageGenerationResponse:
    client = await get_client(settings)
    target = registry.pick_image_model(model)
    runtime = registry.resolve_runtime(target)
    refs = normalize_image_field(reference_images)
    ref_paths: list[Path] = []
    if refs:
        ref_paths = await materialize_reference_images(refs, proxy=settings.http_proxy)
    data: list[ImageGenerationData] = []
    attempts = 0
    max_attempts = max(n, 1) + 2
    try:
        while len(data) < n and attempts < max_attempts:
            attempts += 1
            sanitize_gemini_client_cookies(client)
            output = await asyncio.wait_for(
                client.generate_content(
                    _image_prompt(prompt, size, has_reference=bool(ref_paths)),
                    files=ref_paths or None,
                    model=runtime,
                ),
                timeout=settings.image_timeout,
            )
            text = (output.text or "").strip()
            images = output.images or []
            if not images:
                logger.warning(
                    "Gemini returned no image (model=%s, reference_images=%d, "
                    "account_status=%s): %s",
                    getattr(runtime, "model_name", runtime),
                    len(ref_paths),
                    getattr(client, "account_status", None),
                    (text or "<empty reply>")[:600],
                )
                if _quota_message(text):
                    raise RuntimeError(text)
                if _session_unauthenticated(client):
                    raise CookieExpiredError.for_source(settings.cookie_source)
                if text:
                    raise RuntimeError(text)
                raise RuntimeError(
                    "Gemini returned no image (the model path may not match; "
                    "try a different model or check GET /openai/v1/models)"
                )
            for img in images:
                if len(data) >= n:
                    break
                sanitize_gemini_client_cookies(client)
                encoded = await _image_to_b64(img, settings, client)
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
                except (ValueError, base64.binascii.Error) as exc:
                    raise RuntimeError("Generated image was not valid base64 data") from exc
                stored = generated_image_store.put(image_bytes)
                if stored is None:
                    raise RuntimeError(
                        "Generated image is empty or exceeds the download-store limit"
                    )
                data.append(
                    ImageGenerationData(
                        revised_prompt=prompt,
                        b64_json=encoded,
                        url=f"/openai/v1/images/{stored.image_id}",
                    )
                )
    finally:
        if ref_paths:
            tmp_root = ref_paths[0].parent
            if tmp_root.name.startswith("gemini-ref-"):
                for p in ref_paths:
                    p.unlink(missing_ok=True)
                tmp_root.rmdir()
    return ImageGenerationResponse(created=int(time.time()), data=data)
