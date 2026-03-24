# HTML Vision Parse

使用视觉语言模型从网页截图中提取结构化内容（标题、内容、发布时间）。

## 功能特性

- **双模式支持**: HTTP API 接口 / Python CLI 批处理
- **灵活的提取方式**: 支持按字段选择 OCR 或 VL 模型提取
- **混合提取**: OCR + VL 模型置信度自适应切换
- **多模型支持**: Qwen3-VL-2B / InternVL3-1B
- **HTML文件支持**: 直接上传 HTML 文件进行提取

## 快速开始

### 安装依赖

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e ".[playwright]"

# 安装浏览器
playwright install chromium
```

### 配置文件

复制示例配置并修改：

```bash
cp config.yaml.example config.yaml
```

## 使用方式

### 1. HTTP API 服务

启动服务：

```bash
python -m src.api
```

服务地址：`http://localhost:8000`

#### 1.1 从 URL 提取

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

响应：

```json
{
  "success": true,
  "title": "文章标题",
  "content": "文章内容...",
  "publish_time": "2024-03-15",
  "confidence": 0.95,
  "extraction_method": "vl",
  "processing_time_ms": 2500
}
```

#### 1.2 指定提取字段和方法

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "fields": ["title", "content"],
    "methods": {"title": "vl", "content": "ocr"}
  }'
```

#### 1.3 从 HTML 文件提取

```bash
curl -X POST http://localhost:8000/api/v1/extract/html \
  -F "file=@page.html" \
  -F "fields=title,content"
```

#### 1.4 批量处理 URL

```bash
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '["https://example.com/1", "https://example.com/2"]'
```

### 2. Python CLI 批处理

#### 输入文件格式 (JSONL)

```jsonl
{"url": "https://example.com/article1"}
{"url": "https://example.com/article2"}
{"html_source": "<html><body><h1>Hello</h1></body></html>", "url": "local-page"}
```

#### 运行批处理

```bash
python -m src.batch_cli -i input.jsonl -o output.jsonl
```

#### 指定提取字段

```bash
python -m src.batch_cli -i input.jsonl -o output.jsonl --fields title,publish_time
```

#### 指定提取方法

```bash
python -m src.batch_cli -i input.jsonl -o output.jsonl \
  --methods '{"title": "vl", "content": "ocr", "publish_time": "vl"}'
```

#### 限制处理数量

```bash
python -m src.batch_cli -i input.jsonl -o output.jsonl --max-items 100
```

## 配置说明

### 配置文件 (config.yaml)

```yaml
model:
  model_type: "qwen3_vl"        # 或 "internvl3"
  name: "/path/to/model"
  inference_framework: "transformers"  # 或 "vllm"

extraction:
  extraction_method: "vl"       # "vl", "ocr", "hybrid"
  field_methods: {}             # 按字段指定方法

ocr:
  enabled: true
  min_confidence: 0.75

browser:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
```

### 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `MODEL_NAME` | 模型路径 | - |
| `EXTRACTION_METHOD` | 提取方法 | vl |
| `OCR_ENABLED` | 启用OCR | true |
| `API_PORT` | API端口 | 8000 |

### 提取方法说明

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `vl` | 视觉语言模型 | 标题、时间等语义理解任务 |
| `ocr` | 文字识别 | 纯文本内容提取，速度快 |
| `hybrid` | 混合模式 | OCR优先，低置信度时切换VL |

## API 文档

启动服务后访问：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
html_vision_parse/
├── src/
│   ├── api.py              # FastAPI 服务
│   ├── pipeline.py         # 提取流程
│   ├── html_renderer.py    # HTML 渲染
│   ├── batch_cli.py        # 批处理 CLI
│   ├── config_loader.py    # 配置加载
│   ├── flexible_extractor.py # 灵活提取
│   ├── inference.py        # 模型推理
│   ├── extractor.py        # 内容提取
│   └── browser.py         # 浏览器控制
├── config/
│   └── settings.py        # 配置定义
├── config.yaml.example    # 配置示例
└── docs/                  # 文档目录
```

## 性能参考

| 模型 | 参数量 | 推理时间 |
|------|--------|---------|
| Qwen3-VL-2B | 2B | ~3s/图 |
| InternVL3-1B | 1B | ~2.2s/图 |
| EasyOCR | - | ~2s/图 |

## 常见问题

### Q: 如何选择模型？
A: InternVL3-1B 速度更快，适合大规模处理；Qwen3-VL-2B 精度更高。

### Q: OCR 和 VL 哪个更好？
A: 标题和时间推荐用 VL（语义理解强），正文内容推荐 OCR（速度快）。

### Q: 如何提高提取精度？
A: 启用混合模式 `extraction_method: "hybrid"`，低置信度时自动切换到 VL。
