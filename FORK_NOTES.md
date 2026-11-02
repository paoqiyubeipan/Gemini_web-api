# Fork-specific changes

**Date:** 2026-11-02  
**Fork:** `paoqiyubeipan/Gemini_web-api`  
**Upstream:** [`Lutiancheng1/gemini-webapi-proxy`](https://github.com/Lutiancheng1/gemini-webapi-proxy)

This repository is a public fork. The upstream repository is unchanged.

## Changes in this fork

- **Image-model alias fallback:** when stable image aliases such as `gemini-2.5-flash-image` or `gemini-2.5-pro-image` do not match the runtime model IDs reported by the current Gemini Web session, resolve them to the best matching live Flash or Pro model. A regression test covers the current `gemini-flash` and `gemini-pro` IDs.
- **Two-level synchronous thinking switch:** `thinking_level: 1` requests standard thinking and `thinking_level: 2` requests extended thinking; the proxy maps these to the SDK's verified `extended_thinking: false/true` boolean. The current SDK does not expose arbitrary multi-level intensity. The API defaults to level `2` (extended); explicitly set `1` for standard thinking. The chat service must do this mapping itself: `GeminiClient.generate_content` forwards unknown keyword arguments into the underlying HTTP request, so passing `thinking_level` straight through produced `AsyncSession.request() got an unexpected keyword argument 'thinking_level'` and an HTTP 500.
- **Chat file input:** `POST /openai/v1/chat/completions` accepts a dedicated `files` field (or the singular alias `file`) with data URL or HTTP(S) URL references. PDF MIME types retain a `.pdf` temporary-file extension before upload. Existing `image_url` message attachments continue to work as well.
- **Honest upstream error classification:** replies that are really quota notices now answer `429` with Gemini's own wording instead of `403 "Upstream refused request"`, and a session the SDK has marked `UNAUTHENTICATED` answers `401`. When a generation returns no image, the service logs one line with the model, reference-image count, account status, and upstream text, so a reference-image failure can be told apart from an expired cookie or an exhausted quota.
- **Documentation and examples:** both READMEs and the API reference identify the fork-specific behavior and date.

The upstream `LICENSE` and `NOTICE` files, original copyright notices, and project license terms are preserved. This fork remains under GPL-3.0; see [`LICENSE`](LICENSE).
