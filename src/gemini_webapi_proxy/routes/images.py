"""``POST /openai/v1/images/generations`` — text-to-image + reference images."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from gemini_webapi_proxy.auth import require_api_key_if_configured
from gemini_webapi_proxy.config import get_settings
from gemini_webapi_proxy.schemas import ImageGenerationRequest, ImageGenerationResponse
from gemini_webapi_proxy.services.chat import create_image_generation
from gemini_webapi_proxy.services.generated_images import generated_image_store
from gemini_webapi_proxy.utils.media import normalize_image_field

router = APIRouter(
    prefix="/openai/v1",
    tags=["images"],
    dependencies=[Depends(require_api_key_if_configured)],
)


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def image_generations(body: ImageGenerationRequest, request: Request):
    settings = get_settings()
    refs = normalize_image_field(body.image)
    if not refs and body.extra_body:
        refs = normalize_image_field(body.extra_body.get("image"))
    try:
        result = await create_image_generation(
            settings,
            model=body.model,
            prompt=body.prompt,
            n=body.n,
            size=body.size,
            response_format=body.response_format,
            reference_images=refs or None,
        )
        for image in result.data:
            if image.url:
                image_id = image.url.rsplit("/", 1)[-1]
                image.url = str(request.url_for("download_generated_image", image_id=image_id))
        return result
    except TimeoutError:
        from gemini_webapi_proxy.routes._errors import error_json

        return error_json("Image generation timed out", 504)
    except Exception as exc:
        from gemini_webapi_proxy.routes._errors import map_api_error

        return await map_api_error(exc, settings)


@router.get("/images/{image_id}", name="download_generated_image")
async def download_generated_image(image_id: str) -> Response:
    """Return the generated image bytes at the URL included in the API response."""
    image = generated_image_store.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Generated image not found or expired")

    extension = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/avif": "avif",
    }.get(image.mime_type, "bin")
    return Response(
        content=image.data,
        media_type=image.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="generated-{image_id}.{extension}"',
            "Content-Length": str(image.size_bytes),
            "Cache-Control": "private, max-age=0, no-store",
        },
    )
