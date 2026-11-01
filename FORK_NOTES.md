# Fork-specific changes

**Date:** 2026-11-02  
**Fork:** `paoqiyubeipan/Gemini_web-api`  
**Upstream:** [`Lutiancheng1/gemini-webapi-proxy`](https://github.com/Lutiancheng1/gemini-webapi-proxy)

This repository is a public fork. The upstream repository is unchanged.

## Changes in this fork

- **Image-model alias fallback:** when stable image aliases such as `gemini-2.5-flash-image` or `gemini-2.5-pro-image` do not match the runtime model IDs reported by the current Gemini Web session, resolve them to the best matching live Flash or Pro model. A regression test covers the current `gemini-flash` and `gemini-pro` IDs.
- **Two-level synchronous thinking switch:** `POST /openai/v1/chat/completions` accepts `thinking_level: 1` for standard thinking and `thinking_level: 2` for extended thinking. The value defaults to `1`. Value `3` is intentionally not exposed: the current synchronous client receives only an asynchronous placeholder for that mode, not the completed answer.
- **Documentation and examples:** both READMEs and the API reference identify the fork-specific behavior and date.

The upstream `LICENSE` and `NOTICE` files, original copyright notices, and project license terms are preserved. This fork remains under GPL-3.0; see [`LICENSE`](LICENSE).
