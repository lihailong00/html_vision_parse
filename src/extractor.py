"""Content extraction from web screenshots."""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from PIL import Image
import structlog

from .inference import InferenceEngine
from prompts.extraction_prompt import (
    get_extraction_prompt,
    get_layout_detection_prompt,
    get_validation_prompt,
)
from config.settings import settings

logger = structlog.get_logger()


class ExtractionResult:
    """Result of content extraction."""

    def __init__(
        self,
        title: Optional[str] = None,
        content: Optional[str] = None,
        publish_time: Optional[str] = None,
        confidence: float = 0.0,
        regions_ignored: Optional[List[str]] = None,
        raw_response: Optional[str] = None,
        parse_error: Optional[str] = None,
        source_url: Optional[str] = None,
        extraction_method: Optional[str] = None,
    ):
        self.title = title
        self.content = content
        self.publish_time = publish_time
        self.confidence = confidence
        self.regions_ignored = regions_ignored or []
        self.raw_response = raw_response
        self.parse_error = parse_error
        self.source_url = source_url
        self.extraction_method = extraction_method  # "ocr" or "vl"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "publish_time": self.publish_time,
            "confidence": self.confidence,
            "regions_ignored": self.regions_ignored,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "source_url": self.source_url,
            "extraction_method": self.extraction_method,
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if result meets confidence threshold."""
        return self.confidence >= settings.extraction.min_confidence

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(title={self.title!r}, "
            f"confidence={self.confidence:.2f}, "
            f"is_valid={self.is_high_confidence})"
        )


class ContentExtractor:
    """Extracts structured content from web screenshots using Qwen3-VL."""

    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
        self.extraction_prompt = get_extraction_prompt()
        self.layout_prompt = get_layout_detection_prompt()

    def _parse_response(self, response: str) -> ExtractionResult:
        """Parse the model's JSON response, handling truncated JSON."""
        try:
            # Clean markdown code blocks
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
                confidence=data.get("confidence", 0.0),
                regions_ignored=data.get("regions_ignored", []),
                raw_response=response,
            )

        except json.JSONDecodeError as e:
            # Try to extract partial data from truncated JSON
            partial_result = self._extract_partial_json(response)
            if partial_result.title or partial_result.content:
                logger.info("partial_parse_success", title=partial_result.title)
                partial_result.raw_response = response
                return partial_result

            logger.warning("parse_error", error=str(e), response=response[:500])
            return ExtractionResult(
                confidence=0.0,
                raw_response=response,
                parse_error=str(e),
            )

    def _extract_partial_json(self, response: str) -> ExtractionResult:
        """Extract fields from potentially truncated JSON response."""
        result = ExtractionResult(confidence=0.5)  # Partial confidence

        # Try to extract title
        title_match = re.search(r'"title"\s*:\s*"([^"]*)"', response)
        if title_match:
            result.title = title_match.group(1)

        # Try to extract publish_time
        time_match = re.search(r'"publish_time"\s*:\s*"([^"]*)"', response)
        if time_match:
            result.publish_time = time_match.group(1)

        # Try to extract content (might be truncated)
        # Look for content after "content": " and before the next field or end
        content_match = re.search(r'"content"\s*:\s*"(.*)', response)
        if content_match:
            content = content_match.group(1)
            # Clean up escaped characters
            content = content.replace('\\n', '\n').replace('\\"', '"')
            # Truncate at common truncation markers or JSON end
            for marker in ['\\n"', '\\n    }', '\\n  }', '...']:
                if marker in content:
                    content = content.split(marker)[0]
                    break
            result.content = content

        return result

    def _parse_time(self, time_str: Optional[str]) -> Optional[str]:
        """
        Normalize time string to YYYY-MM-DD HH:mm format.

        Args:
            time_str: Time string in various formats

        Returns:
            Normalized time string or None
        """
        if not time_str:
            return None

        # Common patterns
        patterns = [
            # 2026-03-23 14:30
            (r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", "%Y-%m-%d %H:%M"),
            # 2026-03-23 14:30:00
            (r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})", "%Y-%m-%d %H:%M:%S"),
            # 2026/03/23 14:30
            (r"(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})", "%Y/%m/%d %H:%M"),
            # 2026年03月23日 14:30
            (r"(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})", "%Y年%m月%d日 %H:%M"),
            # Relative times like "2小时前", "昨天" are harder to normalize
            # without current date context, so we leave them as-is
        ]

        for pattern, _ in patterns:
            match = re.search(pattern, time_str)
            if match:
                try:
                    # Try common formats
                    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y年%m月%d日 %H:%M"]:
                        try:
                            dt = datetime.strptime(time_str.strip(), fmt)
                            return dt.strftime("%Y-%m-%d %H:%M")
                        except ValueError:
                            continue
                except Exception:
                    pass

        # Return as-is if we can't parse it
        return time_str

    def _is_truncated(self, response: str) -> bool:
        """Check if response appears truncated (JSON incomplete)."""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        # Also remove trailing markdown fences
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Check if ends with closing braces (complete JSON)
        # If doesn't end with }, it's likely truncated
        if not cleaned.strip().endswith("}"):
            return True

        # Check if content field is truncated (ends with \n instead of ")
        if '"content": "' in response:
            # Get the portion after content
            content_start = response.find('"content": "')
            if content_start != -1:
                after_content = response[content_start + 11:]
                # If it ends with \n... or ends with " but no closing structure, likely truncated
                if after_content.rstrip().endswith("\\n") or after_content.rstrip().endswith("\n"):
                    return True

        # Try parsing - if it fails, it's truncated
        try:
            json.loads(cleaned)
            return False  # Valid JSON, not truncated
        except json.JSONDecodeError:
            return True

    def extract(self, image: Image.Image) -> ExtractionResult:
        """
        Extract content from a single image with auto-expand for long content.

        Args:
            image: PIL Image of the web screenshot

        Returns:
            ExtractionResult with extracted content
        """
        logger.info("extracting_content", image_size=image.size)

        # Start with moderate tokens
        max_tokens = 4096
        response = self.engine.process_image(
            image=image,
            prompt=self.extraction_prompt,
            max_tokens=max_tokens,
        )

        # Auto-expand if truncated
        attempts = 0
        max_attempts = 3
        while self._is_truncated(response) and attempts < max_attempts:
            attempts += 1
            max_tokens *= 2
            logger.info("content_truncated_expanding", attempt=attempts, max_tokens=max_tokens)

            response = self.engine.process_image(
                image=image,
                prompt=self.extraction_prompt,
                max_tokens=max_tokens,
            )

        if attempts > 0:
            logger.info("content_extracted_after_expand", total_tokens=max_tokens)

        # Parse response
        result = self._parse_response(response)

        # Normalize time
        if result.publish_time and result.publish_time != "null":
            result.publish_time = self._parse_time(result.publish_time)

        logger.info(
            "extraction_completed",
            title=result.title[:50] if result.title else None,
            confidence=result.confidence,
            has_content=bool(result.content),
        )

        return result

    def extract_with_retry(self, image: Image.Image) -> ExtractionResult:
        """
        Extract content with retry on failure.

        Args:
            image: PIL Image of the web screenshot

        Returns:
            ExtractionResult with extracted content
        """
        max_retries = settings.extraction.max_retries

        for attempt in range(max_retries):
            result = self.extract(image)

            if result.parse_error is None or attempt == max_retries - 1:
                return result

            logger.warning(
                "extraction_retry",
                attempt=attempt + 1,
                error=result.parse_error,
            )

        return result

    def extract_batch(self, images: List[Image.Image]) -> List[ExtractionResult]:
        """
        Extract content from multiple images.

        Args:
            images: List of PIL Images

        Returns:
            List of ExtractionResults
        """
        logger.info("batch_extraction_started", count=len(images))

        # Process in batches
        batch_size = settings.model.max_batch_size
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            logger.info("processing_batch", batch_num=i // batch_size + 1, size=len(batch))

            # Batch inference
            responses = self.engine.process_batch(
                images=batch,
                prompt=self.extraction_prompt,
            )

            # Parse each response
            for resp in responses:
                result = self._parse_response(resp)
                if result.publish_time and result.publish_time != "null":
                    result.publish_time = self._parse_time(result.publish_time)
                results.append(result)

        logger.info("batch_extraction_completed", total=len(results))
        return results

    def validate(self, result: ExtractionResult) -> bool:
        """
        Validate an extraction result using cross-validation.

        Args:
            result: The extraction result to validate

        Returns:
            True if result seems correct
        """
        if not settings.extraction.enable_cross_validation:
            return True

        # Simple validation: check if we have basic fields
        if result.parse_error:
            return False

        # Could implement more sophisticated validation here
        # e.g., comparing with a second model pass
        return result.confidence >= settings.extraction.min_confidence
