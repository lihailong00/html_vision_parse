"""Flexible extraction with configurable per-field OCR/VL method."""

import json
import re
from typing import Optional, Dict, Any, List, Literal
from PIL import Image
import structlog

from .inference import InferenceEngine
from .ocr_extractor import OCRExtractor
from .extractor import ExtractionResult
from prompts.extraction_prompt import get_extraction_prompt
from config.settings import settings

logger = structlog.get_logger()


class FlexibleExtractor:
    """
    Flexible extractor that allows per-field OCR/VL method selection.

    Usage:
        extractor = FlexibleExtractor(ocr_extractor, vl_extractor)

        # Extract with specific methods per field
        result = extractor.extract_fields(
            image,
            fields=["title", "content", "publish_time"],
            methods={"title": "vl", "content": "ocr", "publish_time": "vl"}
        )

        # Or use default method from settings
        result = extractor.extract_fields(image, fields=["title", "content"])
    """

    def __init__(
        self,
        ocr_extractor: OCRExtractor,
        vl_engine: InferenceEngine,
    ):
        self.ocr = ocr_extractor
        self.vl = vl_engine
        self.extraction_prompt = get_extraction_prompt()

    def extract_fields(
        self,
        image: Image.Image,
        fields: List[str] = None,
        methods: Dict[str, Literal["vl", "ocr"]] = None,
    ) -> ExtractionResult:
        """
        Extract specific fields using specified or default methods.

        Args:
            image: PIL Image of the screenshot
            fields: List of fields to extract ["title", "content", "publish_time"]
                   If None, extracts all fields.
            methods: Dict mapping field -> method {"vl" or "ocr"}
                   If None, uses settings.extraction.field_methods or extraction_method

        Returns:
            ExtractionResult with extracted fields
        """
        if fields is None:
            fields = ["title", "content", "publish_time"]

        if methods is None:
            methods = self._get_default_methods()

        result = ExtractionResult()

        # Extract each field using specified method
        for field in fields:
            method = methods.get(field, settings.extraction.extraction_method)
            logger.info("extracting_field", field=field, method=method)

            if method == "ocr":
                self._extract_field_ocr(image, field, result)
            else:  # vl
                self._extract_field_vl(image, field, result)

        return result

    def _get_default_methods(self) -> Dict[str, str]:
        """Get default extraction methods for each field."""
        if settings.extraction.field_methods:
            return settings.extraction.field_methods

        method = settings.extraction.extraction_method
        if method == "hybrid":
            # Hybrid: OCR for content, VL for title and time
            return {
                "title": "vl",
                "content": "ocr",
                "publish_time": "vl",
            }
        elif method == "ocr":
            return {
                "title": "ocr",
                "content": "ocr",
                "publish_time": "ocr",
            }
        else:
            return {
                "title": "vl",
                "content": "vl",
                "publish_time": "vl",
            }

    def _extract_field_ocr(
        self,
        image: Image.Image,
        field: str,
        result: ExtractionResult,
    ):
        """Extract a single field using OCR."""
        ocr_result = self.ocr.extract_ocr(image)

        if field == "title":
            title = self.ocr.extract_title(ocr_result.blocks, image.height)
            result.title = title
            logger.info("ocr_title_extracted", title=title[:50] if title else None)

        elif field == "content":
            result.content = ocr_result.full_text
            result.confidence = ocr_result.confidence
            logger.info("ocr_content_extracted", length=len(ocr_result.full_text))

        elif field == "publish_time":
            time = self.ocr.extract_time(ocr_result.blocks)
            result.publish_time = time
            logger.info("ocr_time_extracted", time=time)

    def _extract_field_vl(
        self,
        image: Image.Image,
        field: str,
        result: ExtractionResult,
    ):
        """Extract a single field using VL model."""
        # Build field-specific prompt
        prompt = self._build_field_prompt(field)

        response = self.vl.process_image(
            image=image,
            prompt=prompt,
            max_tokens=500,
        )

        # Parse the response
        field_result = self._parse_field_response(response, field)

        if field == "title" and field_result:
            result.title = field_result.get("title") or field_result.get("name")
        elif field == "content" and field_result:
            result.content = field_result.get("content")
        elif field == "publish_time" and field_result:
            result.publish_time = field_result.get("publish_time")

        # Update confidence if VL succeeded
        if field_result and field_result.get("confidence"):
            conf = float(field_result.get("confidence", 0))
            result.confidence = max(result.confidence, conf)

        logger.info("vl_field_extracted", field=field, success=bool(field_result))

    def _build_field_prompt(self, field: str) -> str:
        """Build prompt for specific field extraction."""
        base = "You are an expert at analyzing web page screenshots.\n"

        if field == "title":
            return base + "Extract ONLY the main title/headline of this webpage. Respond with JSON: {\"title\": \"...\"}"
        elif field == "content":
            return base + "Extract ONLY the main body text content (not navigation, ads, footer). Respond with JSON: {\"content\": \"...\"}"
        elif field == "publish_time":
            return base + "Extract ONLY the publish date/time if visible. Respond with JSON: {\"publish_time\": \"...\"}"
        else:
            return base + f"Extract the {field}. Respond with JSON: {{{field}: \"...\"}}"

    def _parse_field_response(
        self, response: str, field: str
    ) -> Optional[Dict[str, Any]]:
        """Parse VL response for a single field."""
        try:
            # Clean markdown
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            logger.warning("field_parse_failed", field=field, response=response[:200])
            return None

    def extract_all_hybrid(self, image: Image.Image) -> ExtractionResult:
        """
        Extract all fields using hybrid approach:
        - title: VL (better semantic understanding)
        - content: OCR (faster, text completeness)
        - publish_time: VL (better at finding in context)
        """
        return self.extract_fields(
            image,
            fields=["title", "content", "publish_time"],
            methods={
                "title": "vl",
                "content": "ocr",
                "publish_time": "vl",
            }
        )

    def extract_all_vl(self, image: Image.Image) -> ExtractionResult:
        """Extract all fields using VL model."""
        return self.extract_fields(
            image,
            fields=["title", "content", "publish_time"],
            methods={"title": "vl", "content": "vl", "publish_time": "vl"}
        )

    def extract_all_ocr(self, image: Image.Image) -> ExtractionResult:
        """Extract all fields using OCR."""
        return self.extract_fields(
            image,
            fields=["title", "content", "publish_time"],
            methods={"title": "ocr", "content": "ocr", "publish_time": "ocr"}
        )
