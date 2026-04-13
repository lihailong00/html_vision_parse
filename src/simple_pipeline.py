"""Simple extraction pipeline using URL/HTML input with API-based VL extraction."""

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from PIL import Image
from loguru import logger

from .browser import BrowserContext
from .screenshot import ScreenshotCapture
from .api_client import ClaudeAPIClient, GPTAPIClient, GeminiAPIClient
from prompts.extraction_prompt import EXTRACTION_PROMPT

# TLD to country code mapping
TLD_COUNTRY_MAP = {
    ".cn": "CN",
    ".jp": "JP",
    ".kr": "KR",
    ".au": "AU",
    ".nz": "NZ",
    ".uk": "GB",
    ".de": "DE",
    ".fr": "FR",
    ".es": "ES",
    ".it": "IT",
    ".ru": "RU",
    ".br": "BR",
    ".in": "IN",
    ".sg": "SG",
    ".hk": "HK",
    ".tw": "TW",
    ".th": "TH",
    ".vn": "VN",
    ".my": "MY",
    ".id": "ID",
    ".ph": "PH",
}

# TLD to language mapping
TLD_LANG_MAP = {
    ".cn": "zh",
    ".jp": "ja",
    ".kr": "ko",
    ".au": "en",
    ".nz": "en",
    ".uk": "en",
    ".de": "de",
    ".fr": "fr",
    ".es": "es",
    ".it": "it",
    ".ru": "ru",
    ".br": "pt",
    ".in": "en",
    ".sg": "en",
    ".hk": "zh",
    ".tw": "zh",
    ".th": "th",
    ".vn": "vi",
    ".my": "en",
    ".id": "en",
    ".ph": "en",
}


@dataclass
class ExtractionResult:
    """Result of content extraction from web pages."""

    title: Optional[str] = None
    content: Optional[str] = None
    publish_time: Optional[str] = None
    lang_type: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "publish_time": self.publish_time,
            "lang_type": self.lang_type,
            "country": self.country,
            "city": self.city,
            "raw_response": self.raw_response,
            "error": self.error,
        }


def extract_country_from_url(url: str) -> Optional[str]:
    """Extract country code from URL TLD."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        for tld, country in TLD_COUNTRY_MAP.items():
            if hostname.endswith(tld):
                return country
        return None
    except Exception:
        return None


def extract_city_from_url(url: str) -> Optional[str]:
    """Extract city from domain prefix heuristic."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        if ":" in hostname:
            hostname = hostname.split(":")[0]
        parts = hostname.replace("www.", "").split(".")
        if len(parts) >= 3:
            prefix = parts[0]
            city = prefix.capitalize()
            if city.lower() not in ("news", "blog", "article", "m", "mobile", "en", "zh", "jp", "kr"):
                return city
        return None
    except Exception:
        return None


def extract_lang_from_tld(url: str) -> Optional[str]:
    """Infer language from URL TLD."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        for tld, lang in TLD_LANG_MAP.items():
            if hostname.endswith(tld):
                return lang
        return None
    except Exception:
        return None


class SimplePipeline:
    """
    Simple extraction pipeline for web content extraction.

    Supports extraction from URLs and HTML content using vision language models
    via API clients (Claude, GPT, Gemini).

    Example:
        pipeline = SimplePipeline(api_provider="claude", api_key="sk-...")
        result = pipeline.extract_from_url("https://example.com/article")
        print(result.title, result.content)
    """

    def __init__(self, api_provider: str = "claude", api_key: str = None):
        """
        Initialize the pipeline.

        Args:
            api_provider: API provider name ("claude", "gpt", "gemini")
            api_key: API key for the provider. If None, reads from environment.
        """
        self._screenshot = ScreenshotCapture()

        resolved_key = api_key
        if not resolved_key:
            env_var_map = {
                "claude": "ANTHROPIC_API_KEY",
                "gpt": "OPENAI_API_KEY",
                "gemini": "GOOGLE_API_KEY",
            }
            env_var = env_var_map.get(api_provider.lower())
            if env_var:
                resolved_key = os.environ.get(env_var)

        api_clients = {
            "claude": ClaudeAPIClient,
            "gpt": GPTAPIClient,
            "gemini": GeminiAPIClient,
        }

        provider_class = api_clients.get(api_provider.lower())
        if provider_class is None:
            raise ValueError(f"Unknown API provider: {api_provider}. Available: {list(api_clients.keys())}")

        self._api_client = provider_class(resolved_key)
        self._api_provider = api_provider.lower()

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_llm_api(self, image_base64: str, prompt: str) -> str:
        """Call LLM API with image and prompt."""
        return self._api_client.chat_completion(image_base64, prompt)

    def _parse_llm_response(self, response: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult fields."""
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            data = json.loads(cleaned.strip())

            return ExtractionResult(
                title=data.get("title"),
                content=data.get("content"),
                publish_time=data.get("publish_time"),
                lang_type=data.get("lang_type") or data.get("language"),
                raw_response=response,
            )
        except json.JSONDecodeError as e:
            logger.warning("parse_error", error=str(e), response=response[:500])
            return ExtractionResult(
                raw_response=response,
                error=f"JSON parse error: {str(e)}",
            )

    def _extract_country_and_city(self, url: str, result: ExtractionResult) -> ExtractionResult:
        """Extract country and city from URL."""
        result.country = extract_country_from_url(url)
        result.city = extract_city_from_url(url)
        return result

    def _extract_lang_type(self, url: str, result: ExtractionResult) -> ExtractionResult:
        """Extract lang_type from LLM response or URL TLD fallback."""
        if not result.lang_type:
            result.lang_type = extract_lang_from_tld(url)
        return result

    def _extract(self, image: Image.Image, url: str = None) -> ExtractionResult:
        """Core extraction logic: image to ExtractionResult."""
        try:
            image_base64 = self._image_to_base64(image)
            response = self._call_llm_api(image_base64, EXTRACTION_PROMPT)
            result = self._parse_llm_response(response)

            if result.error:
                return result

            if url:
                result = self._extract_country_and_city(url, result)
                result = self._extract_lang_type(url, result)

            return result

        except Exception as e:
            logger.error("extraction_failed", error=str(e))
            return ExtractionResult(error=str(e))

    def extract_from_url(self, url: str) -> ExtractionResult:
        """Extract content from a URL."""
        try:
            image = self._screenshot.capture_sync(url)
            return self._extract(image, url)
        except Exception as e:
            logger.error("url_capture_failed", url=url, error=str(e))
            return ExtractionResult(error=f"Failed to capture URL: {str(e)}")

    def _render_html(self, html_content: str) -> Image.Image:
        """Render HTML content to PIL Image using BrowserContext."""
        data_url = f"data:text/html;base64,{base64.b64encode(html_content.encode('utf-8')).decode()}"

        async def _capture():
            async with BrowserContext() as page:
                await page.goto(
                    data_url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                await page.wait_for_timeout(1000)
                screenshot_bytes = await page.screenshot(type="png")
                return Image.open(BytesIO(screenshot_bytes)).convert("RGB")

        return asyncio.run(_capture())

    def extract_from_html(self, html_content: str, url: str = None) -> ExtractionResult:
        """Extract content from HTML content."""
        try:
            image = self._render_html(html_content)
            return self._extract(image, url)
        except Exception as e:
            logger.error("html_render_failed", error=str(e))
            return ExtractionResult(error=f"Failed to render HTML: {str(e)}")
