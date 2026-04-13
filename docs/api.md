# API 参考文档

## 基础信息

- **Base URL**: `http://localhost:18765`
- **API 版本**: v1 (简化版)

## 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/extract` | 从 URL 或 HTML 提取 |

---

## GET /health

检查服务健康状态。

**响应**:
```json
{
  "status": "ok"
}
```

---

## POST /extract

从 URL 或 HTML 源码提取内容。

**请求体**:

```json
{
  "url": "https://example.com/article"
}
```

或：

```json
{
  "html_source": "<html><body><h1>Hello</h1></body></html>"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 否* | 目标 URL |
| `html_source` | string | 否* | HTML 源码 |

*`url` 和 `html_source` 至少提供一个

**响应**:

```json
{
  "title": "文章标题",
  "content": "文章正文内容...",
  "publish_time": "2024-03-15",
  "lang_type": "zh",
  "country": null,
  "city": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | string | 页面标题 |
| `content` | string | 正文内容 |
| `publish_time` | string | 发布时间 (YYYY-MM-DD HH:mm) |
| `lang_type` | string | 语言代码 (en, zh, ja, ko...) |
| `country` | string | 国家代码 (.cn→CN, .jp→JP...) |
| `city` | string | 城市名 (从域名提取) |

---

## 错误响应

```json
{
  "detail": "错误描述信息"
}
```

**常见 HTTP 状态码**:

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 缺少 url 或 html_source |
| 500 | 服务器内部错误 (LLM API 错误等) |
