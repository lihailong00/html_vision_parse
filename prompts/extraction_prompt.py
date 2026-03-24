"""Prompts for structured information extraction from web screenshots."""

# Main extraction prompt - identifies main content region and extracts information
EXTRACTION_PROMPT = """你是一个专业的网页内容提取助手。请仔细分析这张网页截图，提取主内容区域的信息。

## 你的任务

1. **识别主内容区域**：忽略以下内容：
   - 导航栏（顶部、侧边）
   - 广告、弹窗、banner
   - 侧边栏（左侧、右侧）
   - 页脚、版权信息
   - 推荐列表、相关链接
   - 评论区（除非是页面的主要内容）

2. **提取信息**：
   - 标题：页面的主标题（H1或最大最醒目的文字）
   - 正文：主内容区域的可读文本，去除所有噪音内容
   - 发布时间：主内容附近的日期（格式化为 YYYY-MM-DD HH:mm）

## 重要规则

- 只提取主内容区域的信息
- 正文应该连贯、可读，去除无关的干扰内容
- 如果无法确定某个字段，标记为 null
- 保持原文的语言

## 输出格式

请严格以JSON格式输出，不要包含任何其他文字：

{
  "title": "页面标题",
  "content": "正文内容摘要或完整内容...",
  "publish_time": "2026-03-23 14:30",
  "confidence": 0.95,
  "regions_ignored": ["侧边栏-推荐列表", "顶部导航", "右下角广告"]
}

confidence: 你对这次提取的自信程度（0.0-1.0）
regions_ignored: 你忽略的区域列表
"""

# Layout detection prompt - identifies different regions in the page
LAYOUT_DETECTION_PROMPT = """分析这张网页截图，标记出页面的不同区域。

请标记以下区域：
- main_content: 主内容区域（包含文章、产品等信息）
- navigation: 导航栏
- sidebar: 侧边栏
- advertisement: 广告区域
- footer: 页脚
- comments: 评论区

以JSON格式输出每个区域的位置（使用百分比坐标）：
{
  "regions": {
    "main_content": {"x": 10, "y": 15, "width": 60, "height": 70},
    "navigation": {"x": 0, "y": 0, "width": 100, "height": 8},
    ...
  }
}
"""

# Cross-validation prompt - for when we want to double-check
VALIDATION_PROMPT = """请验证以下提取结果是否正确：

标题：{title}
正文（前100字）：{content_preview}...
发布时间：{publish_time}

如果发现错误请指出，并给出修正后的结果。如果正确，回复"确认正确"。
"""


def get_extraction_prompt() -> str:
    """Returns the main extraction prompt."""
    return EXTRACTION_PROMPT


def get_layout_detection_prompt() -> str:
    """Returns the layout detection prompt."""
    return LAYOUT_DETECTION_PROMPT


def get_validation_prompt(title: str, content: str, publish_time: str) -> str:
    """Returns the validation prompt with filled data."""
    return VALIDATION_PROMPT.format(
        title=title,
        content_preview=content[:100] if content else "",
        publish_time=publish_time or "未找到"
    )
