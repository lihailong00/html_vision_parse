# 配置指南

## 配置文件

项目使用 YAML 格式配置文件，可以复制示例配置：

```bash
cp config.yaml.example config.yaml
```

---

## 配置项详解

### model - 模型配置

```yaml
model:
  # 模型类型: "qwen3_vl" (2B) 或 "internvl3" (1B)
  model_type: "qwen3_vl"

  # 模型路径
  name: "/home/longcoding/dev/models/Qwen3-VL-2B-Instruct"

  # 推理设备: "cuda" 或 "cpu"
  device: "cuda"

  # 数据类型: "bfloat16", "float16", "float32"
  dtype: "bfloat16"

  # 最大批处理大小
  max_batch_size: 4

  # 量化: "none", "int8", "int4"
  quantization: "int4"

  # 推理框架: "transformers" 或 "vllm"
  inference_framework: "transformers"

  # InternVL3 模型路径 (当 model_type 为 "internvl3" 时使用)
  internvl_model_path: "/home/longcoding/dev/models/InternVL3-1B"
```

**模型选择建议**:

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| Qwen3-VL-2B | 2B | 较慢 (~3s) | 较高 | 精度优先 |
| InternVL3-1B | 1B | 较快 (~2.2s) | 中等 | 速度优先 |

---

### extraction - 提取配置

```yaml
extraction:
  # 最小置信度阈值
  min_confidence: 0.85

  # 启用交叉验证 (提高精度但降低速度)
  enable_cross_validation: false

  # 失败最大重试次数
  max_retries: 2

  # 提取超时时间 (秒)
  timeout_seconds: 30

  # 默认提取方法: "vl", "ocr", "hybrid"
  extraction_method: "vl"

  # 按字段指定提取方法 (优先级高于 extraction_method)
  # 留空则使用 extraction_method 作为默认值
  field_methods: {}
```

**提取方法说明**:

| 方法 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| `vl` | 视觉语言模型 | 语义理解强，精度高 | 速度较慢 |
| `ocr` | 文字识别 | 速度快 (~2s) | 无语义理解 |
| `hybrid` | 混合模式 | 自适应，平衡 | 配置复杂 |

**field_methods 示例**:

```yaml
# 推荐配置: OCR提取正文，VL提取标题和时间
extraction:
  field_methods:
    title: "vl"
    content: "ocr"
    publish_time: "vl"
```

---

### browser - 浏览器配置

```yaml
browser:
  # 浏览器类型: "chromium", "firefox", "webkit"
  type: "chromium"

  # 无头模式 (不显示浏览器窗口)
  headless: true

  # 视口尺寸
  viewport_width: 1920
  viewport_height: 1080

  # 截图模式: true=全页, false=可见区域
  full_page: true

  # 等待条件: "load", "domcontentloaded", "networkidle"
  wait_until: "networkidle"

  # 页面加载超时 (毫秒)
  wait_timeout: 30000

  # 用户代理
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
```

---

### api - API 服务配置

```yaml
api:
  # 监听地址
  host: "0.0.0.0"

  # 监听端口
  port: 8000

  # uvicorn worker 数量
  workers: 1

  # 最大并发请求数
  max_concurrent_requests: 10
```

---

### ocr - OCR 配置

```yaml
ocr:
  # 是否启用 OCR
  enabled: true

  # OCR 最小置信度 (低于此值切换到 VL)
  min_confidence: 0.75

  # 使用 EasyOCR (true) 或 PaddleOCR (false)
  use_easyocr: true

  # OCR 置信度低时是否 fallback 到 VL
  fallback_to_vl: true

  # OCR 处理超时 (秒)
  ocr_timeout_seconds: 10
```

---

### batch - 批处理配置

```yaml
batch:
  # 批处理大小
  batch_size: 4

  # 工作进程数
  num_workers: 4

  # 预取因子
  prefetch_factor: 2
```

---

## 环境变量

可以通过环境变量覆盖配置文件设置：

| 环境变量 | 对应配置 | 类型 | 说明 |
|----------|----------|------|------|
| `MODEL_NAME` | model.name | string | 模型路径 |
| `MODEL_TYPE` | model.model_type | string | 模型类型 |
| `INFERENCE_FRAMEWORK` | model.inference_framework | string | 推理框架 |
| `EXTRACTION_METHOD` | extraction.extraction_method | string | 默认提取方法 |
| `FIELD_METHODS` | extraction.field_methods | JSON | 字段方法映射 |
| `MIN_CONFIDENCE` | extraction.min_confidence | float | 最小置信度 |
| `BROWSER_TYPE` | browser.type | string | 浏览器类型 |
| `BROWSER_HEADLESS` | browser.headless | bool | 无头模式 |
| `VIEWPORT_WIDTH` | browser.viewport_width | int | 视口宽度 |
| `VIEWPORT_HEIGHT` | browser.viewport_height | int | 视口高度 |
| `API_HOST` | api.host | string | API 地址 |
| `API_PORT` | api.port | int | API 端口 |
| `API_WORKERS` | api.workers | int | Worker数量 |
| `OCR_ENABLED` | ocr.enabled | bool | 启用OCR |
| `OCR_MIN_CONFIDENCE` | ocr.min_confidence | float | OCR最小置信度 |
| `BATCH_SIZE` | batch.batch_size | int | 批处理大小 |

**示例**:

```bash
# 使用环境变量覆盖配置
EXTRACTION_METHOD=ocr API_PORT=9000 python -m src.api
```

---

## 完整配置示例

```yaml
model:
  model_type: "internvl3"
  name: "/home/longcoding/dev/models/InternVL3-1B"
  device: "cuda"
  dtype: "bfloat16"
  max_batch_size: 4
  quantization: "none"
  inference_framework: "transformers"

extraction:
  min_confidence: 0.85
  enable_cross_validation: false
  max_retries: 2
  timeout_seconds: 30
  extraction_method: "hybrid"
  field_methods:
    title: "vl"
    content: "ocr"
    publish_time: "vl"

browser:
  type: "chromium"
  headless: true
  viewport_width: 1920
  viewport_height: 1080
  full_page: true
  wait_until: "networkidle"
  wait_timeout: 30000

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  max_concurrent_requests: 10

ocr:
  enabled: true
  min_confidence: 0.75
  use_easyocr: true
  fallback_to_vl: true
  ocr_timeout_seconds: 10

batch:
  batch_size: 4
  num_workers: 4
  prefetch_factor: 2
```
