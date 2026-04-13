<!-- Parent: ../AGENTS.md -->

# src

## Purpose
Core source code for the simplified web screenshot parser. URL/HTML → Playwright screenshot → LLM API → JSON.

## Key Files

| File | Description |
|------|-------------|
| `api_client.py` | LLM API clients: `ClaudeAPIClient`, `GPTAPIClient`, `GeminiAPIClient` (all implement `chat_completion(image_base64, prompt) -> str`) |
| `simple_pipeline.py` | `SimplePipeline` class + `ExtractionResult` dataclass. Core pipeline: screenshot → API → JSON with 6 fields |
| `api.py` | FastAPI server. Single endpoint: `POST /extract` accepting `{url?, html_source?}` |
| `browser.py` | `BrowserContext` class - Playwright browser management |
| `screenshot.py` | `ScreenshotCapture` class - captures screenshots from URLs |

## For AI Agents

### Architecture Flow
1. **Input**: `url` or `html_source`
2. **Screenshot**: `ScreenshotCapture.capture_sync(url)` or `SimplePipeline._render_html(html_content)`
3. **API Call**: `LLMAPIClient.chat_completion(image_base64, prompt)`
4. **Parse**: `SimplePipeline._parse_llm_response()` → `ExtractionResult`

### Output Fields
`title`, `content`, `publish_time`, `lang_type`, `country`, `city`

### Adding New LLM Providers
1. Create new class in `api_client.py` extending `LLMAPIClient`
2. Implement `chat_completion(image_base64, prompt) -> str`
3. Add to dict in `SimplePipeline.__init__`

### Dependencies
- `fastapi`, `uvicorn` - REST API
- `playwright` - Browser automation
- `anthropic` / `openai` / `google-genai` - LLM API clients
- `Pillow` - Image manipulation
- `loguru` - Logging
