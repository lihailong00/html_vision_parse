# Web Screenshot Parser 使用教程

基于 Qwen3-VL 的网页截图结构化解析工具。

## 功能

- 输入 URL → 自动打开浏览器截图 → VLM 解析
- 提取：标题(title)、正文(content)、发布时间(publish_time)
- 智能剔除：导航栏、侧边栏、广告、页脚等不相关内容
- 返回置信度评分

## 环境要求

- Python 3.10+
- NVIDIA 显卡 (4060 Ti / T4 等，16GB 显存)
- CUDA

## 安装

### 1. 克隆/进入项目目录

```bash
cd /home/longcoding/dev/project/html_vision_parse
```

### 2. 创建虚拟环境

```bash
uv venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 4. 安装 Playwright 浏览器

```bash
playwright install chromium
```

### 5. 下载模型

```bash
# 安装 HuggingFace CLI
pip install huggingface_hub

# 下载 Qwen3-VL-8B 模型
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /home/longcoding/dev/models/Qwen3-VL-8B-Instruct
```

### 6. 配置模型路径

编辑 `config/settings.py`：

```python
model:
  name: "/home/longcoding/dev/models/Qwen3-VL-8B-Instruct"  # 你的模型路径
```

---

## 使用方式

### 方式 1：命令行

#### 解析单个 URL
```bash
python main.py scrape https://news.ycombinator.com
```

#### 仅截图不解析
```bash
python main.py scrape https://example.com --screenshot-only
```

#### 批量解析 URL 列表
```bash
# 创建 url.txt，每行一个 URL
echo "https://news.ycombinator.com" > urls.txt
echo "https://www.bbc.com/news/technology" >> urls.txt

python main.py scrape-batch urls.txt -o results.json
```

#### 解析本地截图
```bash
python main.py single screenshot.png
```

---

### 方式 2：Python 代码

```python
from src.pipeline import WebPagePipeline

# 创建 pipeline
pipeline = WebPagePipeline()

# 解析 URL
result = pipeline.run_sync("https://news.ycombinator.com")

# 查看结果
print(result)
# {
#     'title': 'Hacker News',
#     'content': '1. POSSSE – Publish on your Own Site...',
#     'publish_time': None,
#     'confidence': 0.98,
#     'regions_ignored': ['顶部导航栏', '页脚', '登录按钮', '搜索框'],
#     'parse_error': None
# }
```

---

### 方式 3：API 服务

#### 启动服务
```bash
python main.py serve --port 8000
```

#### 调用接口

**解析 URL**：
```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://news.ycombinator.com"}'
```

**仅截图**：
```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "screenshot_only": true}'
```

**上传截图文件**：
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@screenshot.png"
```

---

## 配置说明

编辑 `config/settings.py`：

```python
model:
  name: "/path/to/model"           # 模型路径
  inference_framework: "transformers" # 推理框架
  quantization: "int4"             # 量化: none/int8/int4

extraction:
  min_confidence: 0.85             # 置信度阈值

browser:
  type: "chromium"                 # 浏览器
  headless: True                   # 无头模式
  viewport_width: 1920
  viewport_height: 1080
  full_page: True                  # 截整页
  wait_until: "networkidle"        # 等待策略
```

---

## 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | string | 页面标题 |
| `content` | string | 正文内容（已去除噪音） |
| `publish_time` | string | 发布时间（格式：YYYY-MM-DD HH:mm） |
| `confidence` | float | 置信度 (0.0 - 1.0) |
| `regions_ignored` | list | 被剔除的区域列表 |
| `parse_error` | string | 解析错误信息（正常为 None） |

---

## 性能

| 指标 | 数值 |
|------|------|
| 单张截图解析 | ~3-5 秒 |
| 每日处理量 | 1000+ 张 |
| 显存占用 (INT4) | ~8-10GB |

---

## 常见问题

### Q: 显存不足怎么办？
A: 尝试更激进的量化 `quantization: "int4"`，或使用更小的模型如 `Qwen3-VL-4B-Instruct`

### Q: 解析失败/置信度低怎么办？
A: 可以降低 `extraction.min_confidence` 阈值，或启用交叉验证 `enable_cross_validation: true`

### Q: 如何处理需要登录的页面？
A: 可以禁用 headless 模式 `headless: false`，手动操作登录

### Q: 页面加载慢怎么办？
A: 调整 `browser.wait_until` 为 "load" 或 "domcontentloaded"

---

## 项目结构

```
html_vision_parse/
├── main.py                 # CLI 入口
├── config/
│   └── settings.py         # 配置文件
├── src/
│   ├── browser.py          # Playwright 浏览器管理
│   ├── screenshot.py       # 截图逻辑
│   ├── pipeline.py         # 整合流程
│   ├── model_loader.py      # 模型加载
│   ├── inference.py         # 推理引擎
│   ├── extractor.py         # 内容提取
│   ├── batch_processor.py    # 批量处理
│   └── api.py               # FastAPI 服务
├── prompts/
│   └── extraction_prompt.py # 提取 Prompt
└── requirements.txt
```
