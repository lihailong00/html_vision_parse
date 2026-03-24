# CLI 使用指南

## 批处理命令

### 基本用法

```bash
python -m src.batch_cli -i input.jsonl -o output.jsonl
```

### 参数说明

| 参数 | 短参数 | 必填 | 说明 |
|------|--------|------|------|
| `--input` | `-i` | 是 | 输入 JSONL 文件路径 |
| `--output` | `-o` | 是 | 输出 JSONL 文件路径 |
| `--fields` | `-f` | 否 | 要提取的字段，逗号分隔 |
| `--methods` | `-m` | 否 | JSON 格式的字段方法映射 |
| `--max-items` | `-n` | 否 | 最大处理数量 |

---

## 输入文件格式

输入文件为 JSONL 格式（每行一个 JSON 对象）。

### URL 模式

```jsonl
{"url": "https://example.com/article1"}
{"url": "https://example.com/article2"}
{"url": "https://example.com/article3"}
```

### HTML 源码模式

```jsonl
{"html_source": "<html><body><h1>标题</h1><p>内容</p></body></html>"}
{"html_source": "<html><body><h1>Another</h1><p>More content</p></body></html>", "url": "optional-identifier"}
```

### 带标识符

```jsonl
{"url": "https://example.com/1", "identifier": "article-001"}
{"html_source": "<html>...</html>", "identifier": "local-file-001"}
```

---

## 输出文件格式

每行对应一个输入记录的处理结果：

```jsonl
{"url": "https://example.com/article1", "identifier": "article-001", "title": "标题", "content": "内容", "publish_time": "2024-03-15", "confidence": 0.95, "extraction_method": "vl", "success": true}
{"url": "https://example.com/article2", "identifier": "article-002", "title": null, "content": null, "publish_time": null, "confidence": 0, "extraction_method": null, "success": false, "error": "Connection timeout"}
```

### 输出字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `url` | string | 来源 URL（HTML模式可能为空） |
| `identifier` | string | 用户提供的标识符 |
| `title` | string | 提取的标题 |
| `content` | string | 提取的内容 |
| `publish_time` | string | 提取的发布时间 |
| `confidence` | float | 置信度 (0-1) |
| `extraction_method` | string | 使用的提取方法 |
| `success` | boolean | 是否成功 |
| `error` | string | 错误信息（失败时） |

---

## 使用示例

### 示例 1: 基本批处理

```bash
python -m src.batch_cli -i urls.jsonl -o results.jsonl
```

### 示例 2: 只提取标题和发布时间

```bash
python -m src.batch_cli -i urls.jsonl -o results.jsonl --fields title,publish_time
```

### 示例 3: 指定提取方法

```bash
python -m src.batch_cli -i urls.jsonl -o results.jsonl \
  --methods '{"title": "vl", "content": "ocr", "publish_time": "vl"}'
```

### 示例 4: 限制处理数量（测试用）

```bash
python -m src.batch_cli -i urls.jsonl -o results.jsonl --max-items 10
```

### 示例 5: 处理 HTML 文件

```bash
python -m src.batch_cli -i html_files.jsonl -o results.jsonl
```

其中 `html_files.jsonl` 内容：

```jsonl
{"html_source": "<html><body><h1>Page Title</h1><article>Content here...</article></body></html>"}
{"html_source": "<html><head><title>Another Page</title></head><body>More content...</body></html>", "identifier": "page-2"}
```

---

## 完整示例

### 准备输入文件

```bash
cat > input.jsonl << 'EOF'
{"url": "https://example.com/1"}
{"url": "https://example.com/2"}
{"html_source": "<html><body><h1>Local Page</h1><p>Some content</p></body></html>"}
EOF
```

### 运行批处理

```bash
python -m src.batch_cli -i input.jsonl -o output.jsonl --fields title,content
```

### 查看结果

```bash
cat output.jsonl | python -m json.tool
```

---

## 性能调优

### 调整批处理大小

修改 `config.yaml`：

```yaml
batch:
  batch_size: 8      # 增加批处理大小
  num_workers: 4      # 增加工作进程
```

### 选择更快模型

```yaml
model:
  model_type: "internvl3"  # 使用 InternVL3-1B (~2.2s/图)
```

### 启用 OCR 加速

```yaml
extraction:
  extraction_method: "ocr"  # 纯 OCR 模式 (~2s/图)
```

---

## 错误处理

批处理会继续处理后续记录，失败的记录会在输出中标记：

```jsonl
{"success": false, "error": "Connection timeout", "line": 5}
{"success": false, "error": "JSON parse error", "line": 12}
```

### 查看处理统计

批处理完成后会打印统计信息：

```
==================================================
Batch Processing Complete
==================================================
Total:      1000
Success:    985
Failed:     12
Skipped:    3
Time:       2567.3s
Rate:       0.4 items/sec
Output:     output.jsonl
==================================================
```

---

## 与 API 的选择

| 场景 | 推荐方式 |
|------|----------|
| 实时单次提取 | API (`/api/v1/extract`) |
| 文件上传提取 | API (`/api/v1/extract/html`) |
| 大量URL批量处理 | CLI 批处理 |
| 需要后台处理 | CLI 批处理 + nohup |
