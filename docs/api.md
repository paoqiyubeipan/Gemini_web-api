# OpenAI 兼容 API

Base：`http://localhost:4982`（Studio 使用 `http://localhost:4982/openai` 作为 OpenAI Base URL）

鉴权：Header `Authorization: Bearer <任意>`（不校验）

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活检查 |
| GET | `/openai/v1/models` | 模型列表 |
| POST | `/openai/v1/chat/completions` | 对话（`stream=false`） |
| GET | `/openai/v1/exports/{export_id}` | 下载从助手回复文本生成的导出文件 |
| POST | `/openai/v1/images/generations` | 文生图 / 参考图生图 |
| GET | `/openai/v1/images/{image_id}` | 下载生成的图片 |
| POST | `/admin/probe-models` | 重新探测并刷新模型注册表 |

## 对话示例

```bash
curl -sS -X POST http://localhost:4982/openai/v1/chat/completions \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash",
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "stream": false,
    "thinking_level": 2
  }'
```

`thinking_level` 支持两档：`2`（扩展，默认）映射为 SDK 的 `extended_thinking=true`；`1`（标准）映射为 `extended_thinking=false`。省略该字段时使用扩展思考。当前 `gemini-webapi` SDK 暴露的是布尔开关，不支持任意多档强度；请求中不应使用 `thinking_level: true/false`。

## 生图示例

```bash
curl -sS -m 300 -X POST http://localhost:4982/openai/v1/images/generations \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash",
    "prompt": "a simple red apple on white background",
    "n": 1,
    "size": "1024x1024"
  }'
```

响应同时包含 `data[0].b64_json`（图片的 base64）和 `data[0].url`（代理生成的绝对下载 URL）。直接 GET 该 URL 即可下载图片；如果配置了 `GOP_API_KEY`，下载请求也需要携带相同的 Bearer key。下载文件在内存中保留 1 小时，服务重启后链接失效。Google CDN URL 不会暴露给调用方。

## 错误响应

所有错误都遵循 OpenAI 风格：

```json
{"error": {"message": "...", "type": "api_error"}}
```

| HTTP | 含义 | 触发条件 |
|------|------|----------|
| 400 | 客户端错误 | `stream=true` 暂未实现（请设 `false`） |
| 401 | Cookie 失效 | Safari 重新登录 `gemini.google.com` → 重新 sync Cookie |
| 403 | **上游拒绝** | Gemini Web 返回了策略/能力限制文案（"I cannot fulfill…", "image creation isn't available in your location" 等），而非真实答案 |
| 429 | **出图额度用尽** | 上游回复额度/配额提示（如 "You've reached your limit for generating images…"），原文照返，等重置后再试 |
| 500 | 内部错误 | 兜底；通常是上面分类没覆盖到 |
| 502 | 出图下载失败 | Google CDN 跳链全部失败；查看 `docker compose logs` |
| 504 | 超时 | 默认 `GOP_CHAT_TIMEOUT=180s` / `GOP_IMAGE_TIMEOUT=300s` |

## 参考图

请求体可增加 OpenAI 扩展字段 `image`（字符串或字符串数组），值为 `data:image/png;base64,...`。

聊天请求支持专用顶层 `files` 字段，值可以是一个 data URL / HTTP(S) URL 字符串，或多个来源组成的数组；单数 `file` 也可作为兼容别名。例如 PDF 可使用 `data:application/pdf;base64,...`，或使用响应 `Content-Type: application/pdf` 的 HTTP(S) URL。代理会保留 `.pdf` 扩展名并把文件交给上游 SDK。为安全起见，不接受服务器本地路径。现有 `image_url` 消息附件仍可使用。

Proxy 提示词包装（有参考图时）：

```text
Use the attached reference image(s) as visual reference ...
Prompt: <用户 prompt>
Requested size/aspect: <size>
```

参考图请求会先上传文件再生成，因此比纯文字多一步上游交互。若返回无图，代理会按上游原文
分类：额度提示 → `429`，会话失效（Gemini 判定 Cookie 未认证）→ `401`，其余拒绝文案 → `403`。
同时服务日志会记录一行 `Gemini returned no image (...)`，包含模型、参考图数量、账号状态和
上游原文，便于区分"参考图没被看见"和"额度/登录问题"。

## 文本内容导出

Gemini Web 可能返回 `[file-tag: code-generated-file-…]`，但该标记只是网页端文件查看器的占位符，
API 传输没有对应的文件字节或可验证的下载地址。代理不会把标记 ID 当作文件，也不会拼接猜测的 URL。

当助手回复中确实包含文本文件内容时，代理会从回复中创建临时导出项，并把描述放在
`choices[0].message.exports`。例如，模型输出带 `csv` 标签的代码块时，会导出代码块正文为 CSV：

```json
{
  "role": "assistant",
  "content": "Here is the CSV:\n```csv\nname,score\nAda,98\n```",
  "exports": [
    {
      "id": "<generated-id>",
      "type": "text",
      "name": "generated-1.csv",
      "mime_type": "text/csv; charset=utf-8",
      "size_bytes": 18,
      "download_url": "/openai/v1/exports/<generated-id>",
      "source": "assistant_text"
    }
  ]
}
```

支持明确标记的 CSV、TSV、Markdown、JSON、TXT、HTML、XML、YAML、SQL、Python、JavaScript、TypeScript、Shell 和 TOML 代码块；
也会识别明显的原始 CSV/TSV 表格或 Markdown 文档。多段符合条件的代码块会分别导出。普通文字即使声称
“文件已生成”，也不会被误当成文件内容；没有真实文本正文时，响应中不会产生 `exports`。

`GET /openai/v1/exports/{export_id}` 返回 UTF-8 文本字节，并使用 `Content-Disposition: attachment`。
导出内容只来源于模型实际输出的文本；任何 `[file-tag: ...]` 标记都会从内容中移除。导出在内存中保存，
默认 1 小时过期，单个导出最大 8 MB，进程重启后失效。

## 出图流水线（内部）

```
POST /images/generations
  → gemini-webapi generate_content（Cookie + 可选 files=附件）
  → RPC 全尺寸 URL
  → CDN 跳转链
  → Playwright Chromium GET（下载 PNG）
  → b64_json 与代理 GET 下载 URL 返回
```

CDN 域名直连，不走 `HTTP_PROXY`（避免 MITM 403）。
