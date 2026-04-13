# 配置指南

## 配置文件

项目使用 Pydantic `config/settings.py` 管理配置。

---

## 配置项详解

### browser - 浏览器配置

```python
class BrowserConfig:
    type: str = "chromium"      # 浏览器类型
    headless: bool = True       # 无头模式
    viewport_width: int = 1920   # 视口宽度
    viewport_height: int = 1080 # 视口高度
    full_page: bool = True      # 全页截图
    wait_until: str = "networkidle"  # 等待条件
    wait_timeout: int = 30000   # 超时 (毫秒)
    user_agent: str = "..."     # 用户代理
```

### api - API 配置

```python
class APIConfig:
    host: str = "0.0.0.0"      # 监听地址
    port: int = 18765          # 监听端口
    provider: str = "claude"   # LLM 提供商: "claude", "gpt", "gemini"
    api_key: str = ""         # API Key (留空从环境变量读取)
    model: str = ""           # 模型名称 (使用提供商默认值)
```

---

## 环境变量

| 环境变量 | 说明 |
|---------|------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API Key |
| `OPENAI_API_KEY` | OpenAI API Key |
| `GOOGLE_API_KEY` | Google Gemini API Key |

---

## 完整配置

```python
from config.settings import settings

# 修改配置
settings.browser.headless = True
settings.browser.viewport_width = 1920
settings.api.provider = "claude"
```

---

## LLM 提供商

| 提供商 | 环境变量 | 默认模型 |
|--------|---------|---------|
| Claude | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| GPT | `OPENAI_API_KEY` | gpt-4o |
| Gemini | `GOOGLE_API_KEY` | gemini-2.0-flash |
