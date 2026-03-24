# API 参考文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API版本**: v1

## 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/extract` | 从URL提取 |
| POST | `/api/v1/extract/html` | 从HTML文件提取 |
| POST | `/api/v1/batch` | 批量提取URL |
| POST | `/extract` | 从图片文件提取 (兼容) |
| POST | `/scrape` | 抓取URL截图+提取 (兼容) |

---

## GET /health

检查服务健康状态。

**响应**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## POST /api/v1/extract

从 URL 提取内容。

**请求体**:
```json
{
  "url": "https://example.com/article",
  "fields": ["title", "content", "publish_time"],
  "methods": {"title": "vl", "content": "ocr"},
  "use_hybrid": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 目标URL |
| `fields` | string[] | 否 | 要提取的字段，不指定则提取全部 |
| `methods` | object | 否 | 每个字段的提取方法 `{"field": "vl"\|"ocr"}` |
| `use_hybrid` | boolean | 否 | 是否使用混合模式 |

**fields 可选值**: `title`, `content`, `publish_time`

**methods 可选值**: `vl`, `ocr`

**响应**:
```json
{
  "success": true,
  "title": "文章标题",
  "content": "文章正文内容...",
  "publish_time": "2024-03-15",
  "confidence": 0.95,
  "extraction_method": "vl",
  "source_url": "https://example.com/article",
  "processing_time_ms": 2500,
  "error": null
}
```

---

## POST /api/v1/extract/html

从上传的 HTML 文件提取内容。

**Content-Type**: `multipart/form-data`

**表单字段**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | HTML 文件 |
| `fields` | string | 否 | 逗号分隔的字段，如 `title,content` |
| `methods` | string | 否 | JSON 格式的方法映射 |
| `use_hybrid` | boolean | 否 | 是否使用混合模式 |

**示例**:
```bash
curl -X POST http://localhost:8000/api/v1/extract/html \
  -F "file=@article.html" \
  -F "fields=title,content" \
  -F "methods={\"title\": \"vl\", \"content\": \"ocr\"}"
```

**响应**: 同 `/api/v1/extract`

---

## POST /api/v1/batch

批量处理多个 URL。

**请求体**:
```json
["https://example.com/1", "https://example.com/2", "https://example.com/3"]
```

**响应**:
```json
[
  {
    "success": true,
    "title": "标题1",
    "content": "内容1",
    "publish_time": "2024-03-15",
    "confidence": 0.95,
    "extraction_method": "vl",
    "source_url": "https://example.com/1",
    "error": null,
    "processing_time_ms": 2500
  },
  {
    "success": false,
    "error": "Connection timeout",
    "processing_time_ms": 30000
  }
]
```

---

## POST /extract (兼容模式)

从上传的图片文件提取内容。

**表单字段**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 图片文件 (PNG/JPEG/WebP) |

**响应**:
```json
{
  "title": "文章标题",
  "content": "文章内容",
  "publish_time": "2024-03-15",
  "confidence": 0.95,
  "regions_ignored": [],
  "is_high_confidence": true,
  "parse_error": null
}
```

---

## POST /scrape (兼容模式)

抓取 URL 并提取内容。

**请求体**:
```json
{
  "url": "https://example.com/article",
  "screenshot_only": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 目标URL |
| `screenshot_only` | boolean | 否 | 仅截图不提取 |

**响应**: 同 `/extract`

---

## 错误响应

所有端点可能的错误响应：

```json
{
  "success": false,
  "error": "错误描述信息",
  "processing_time_ms": 500
}
```

**常见 HTTP 状态码**:

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |
