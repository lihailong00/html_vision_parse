# HTML Vision Parse

使用 Playwright 截图 + 大模型 API 提取网页结构化内容。

## 功能特性

- **截图提取**: Playwright 获取网页截图
- **外部 LLM API**: 支持 Claude / GPT-4o / Gemini
- **输入灵活**: 支持 URL 或 HTML 源码
- **6字段输出**: title, content, publish_time, lang_type, country, city

## 快速开始

### 环境要求

- Python 3.10+
- API Key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY)

### 安装依赖

```bash
source .venv/bin/activate
uv pip install -e ".[playwright]"
playwright install chromium
```

### 配置

设置环境变量或直接修改 `config/settings.py`：

```bash
export ANTHROPIC_API_KEY="sk-..."
```

## 使用方式

### 1. HTTP API 服务

启动服务：

```bash
uvicorn src.api:app --host 0.0.0.0 --port 18765
```

服务地址：`http://localhost:18765`

#### 从 URL 提取

```bash
curl -X POST http://localhost:18765/extract \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

响应：

```json
{
  "title": "文章标题",
  "content": "文章内容...",
  "publish_time": "2024-03-15",
  "lang_type": "zh",
  "country": null,
  "city": null
}
```

#### 从 HTML 提取

```bash
curl -X POST http://localhost:18765/extract \
  -H "Content-Type: application/json" \
  -d '{"html_source": "<html><body><h1>Hello</h1></body></html>"}'
```

### 2. Python 直接调用

```python
from src.simple_pipeline import SimplePipeline

pipeline = SimplePipeline(api_provider="claude")
result = pipeline.extract_from_url("https://example.com/article")
print(result.title, result.content, result.lang_type)
```

## 配置说明

### config/settings.py

```python
class BrowserConfig:
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    full_page: bool = True
    wait_until: str = "networkidle"
    wait_timeout: int = 30000

class APIConfig:
    host: str = "0.0.0.0"
    port: int = 18765
    provider: str = "claude"  # "claude", "gpt", "gemini"
    api_key: str = ""         # 从环境变量读取
    model: str = ""           # 提供商默认模型
```

### 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `GOOGLE_API_KEY` | Google API Key | - |

## API 文档

启动服务后访问：

- Swagger UI: http://localhost:18765/docs
- ReDoc: http://localhost:18765/redoc

## 项目结构

```
html_vision_parse/
├── src/
│   ├── api.py              # FastAPI 服务
│   ├── api_client.py       # LLM API 客户端
│   ├── simple_pipeline.py  # 核心提取流水线
│   ├── browser.py          # Playwright 浏览器
│   └── screenshot.py       # 截图功能
├── config/
│   └── settings.py         # 配置
├── prompts/
│   └── extraction_prompt.py # 提取提示词
└── tests/
    └── test_simple_pipeline.py
```

## 输出字段说明

| 字段 | 来源 | 说明 |
|------|------|------|
| title | 视觉 | 页面标题 |
| content | 视觉 | 正文内容 |
| publish_time | 视觉 | 发布时间 |
| lang_type | 视觉+URL | 语言代码 (en, zh, ja...) |
| country | URL TLD | 国家代码 (.cn→CN, .jp→JP) |
| city | URL | 城市 (如 beijing.xxx.com) |
