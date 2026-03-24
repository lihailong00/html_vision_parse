"""Pipeline integrating browser, screenshot, and extraction."""

from typing import Optional, Dict, Any, List, Literal
import structlog

from .browser import BrowserContext
from .screenshot import ScreenshotCapture
from .model_loader import ModelLoader
from .inference import InferenceEngine
from .vllm_engine import VLLMEngine
from .extractor import ContentExtractor, ExtractionResult
from .ocr_extractor import OCRExtractor
from .flexible_extractor import FlexibleExtractor
from config.settings import settings

logger = structlog.get_logger()


class WebPagePipeline:
    """
    End-to-end pipeline: URL -> Screenshot -> Extraction.

    Example:
        pipeline = WebPagePipeline()
        result = await pipeline.run("https://example.com/article")
        print(result.title, result.content, result.publish_time)

    Flexible extraction:
        # Extract only specific fields with specified methods
        result = await pipeline.extract_from_url(
            "https://example.com",
            fields=["title", "publish_time"],
            methods={"title": "vl", "publish_time": "ocr"}
        )

        # Use hybrid extraction (OCR for content, VL for title/time)
        result = await pipeline.extract_from_url(
            "https://example.com",
            use_hybrid=True
        )
    """

    def __init__(self, use_hybrid: bool = True):
        self._screenshot = ScreenshotCapture()
        self._model_loader = None
        self._vllm_engine = None
        self._engine = None
        self._extractor = None
        self._flexible_extractor = None
        self._ocr_extractor = None
        self._use_hybrid = use_hybrid and settings.ocr.enabled

    def _ensure_model_loaded(self):
        """Lazy load the model based on inference framework setting."""
        if self._engine is None:
            if settings.model.inference_framework == "vllm":
                self._vllm_engine = VLLMEngine(settings.model.name)
                self._vllm_engine.load()
                self._engine = self._vllm_engine
            else:
                self._model_loader = ModelLoader()
                self._model_loader.load()
                self._engine = InferenceEngine(self._model_loader)
            self._extractor = ContentExtractor(self._engine)
            # Initialize OCR extractor with VL engine for fallback
            if self._use_hybrid:
                self._ocr_extractor = OCRExtractor(inference_engine=self._engine)
                self._flexible_extractor = FlexibleExtractor(
                    ocr_extractor=self._ocr_extractor,
                    vl_engine=self._engine
                )

    def _release_model(self):
        """Release model resources."""
        if settings.model.inference_framework == "vllm":
            if self._vllm_engine:
                self._vllm_engine.unload()
                self._vllm_engine = None
        else:
            if self._model_loader:
                self._model_loader.unload()
                self._model_loader = None
        self._engine = None
        self._extractor = None
        self._ocr_extractor = None

    async def screenshot_only(self, url: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Just capture screenshot without extraction.

        Args:
            url: Target URL
            save_path: Optional path to save screenshot

        Returns:
            Dict with image info
        """
        logger.info("screenshot_only", url=url)

        image = await self._screenshot.capture(url, screenshot_path=save_path)

        return {
            "url": url,
            "width": image.width,
            "height": image.height,
            "saved_to": save_path,
        }

    async def extract_from_url(
        self,
        url: str,
        use_hybrid: bool = None,
        fields: List[str] = None,
        methods: Dict[str, Literal["vl", "ocr"]] = None,
    ) -> ExtractionResult:
        """
        Extract content from a URL (screenshot + extract).

        Args:
            url: Target URL
            use_hybrid: Override hybrid mode (None = use pipeline default)
            fields: List of fields to extract ["title", "content", "publish_time"]
                   If None, extracts all fields.
            methods: Dict mapping field -> method {"vl" or "ocr"}
                   If None, uses settings.extraction.field_methods

        Returns:
            ExtractionResult with title, content, publish_time
        """
        logger.info("extracting_from_url", url=url, fields=fields, methods=methods)

        # Ensure model is loaded
        self._ensure_model_loaded()

        # Capture screenshot
        image = await self._screenshot.capture(url)

        # Use flexible extraction if methods specified or fields specified
        if methods is not None or fields is not None:
            result = self._flexible_extractor.extract_fields(
                image, fields=fields, methods=methods
            )
            result.source_url = url
            return result

        # Use hybrid OCR+VL extraction if enabled
        should_use_hybrid = use_hybrid if use_hybrid is not None else self._use_hybrid

        if should_use_hybrid and self._ocr_extractor is not None:
            # Try OCR first, fallback to VL if confidence is low
            ocr_result, used_vl = self._ocr_extractor.extract_with_fallback(
                image,
                min_confidence=settings.ocr.min_confidence
            )

            if not used_vl:
                # OCR succeeded - convert OCRResult to ExtractionResult
                result = ExtractionResult(
                    title=self._ocr_extractor.extract_title(
                        ocr_result.blocks, image.height
                    ),
                    content=ocr_result.full_text,
                    publish_time=self._ocr_extractor.extract_time(
                        ocr_result.blocks
                    ),
                    confidence=ocr_result.confidence,
                    extraction_method="ocr",
                )
                logger.info(
                    "ocr_extraction_done",
                    confidence=result.confidence,
                    title=result.title[:50] if result.title else None,
                )
                result.source_url = url
                return result
            else:
                # VL fallback was used
                result = self._extractor.extract(image)
                result.extraction_method = "vl"
                result.source_url = url
                logger.info(
                    "vl_fallback_extraction_done",
                    confidence=result.confidence,
                    title=result.title[:50] if result.title else None,
                )
                return result
        else:
            # Pure VL extraction
            result = self._extractor.extract(image)
            result.extraction_method = "vl"
            result.source_url = url

            logger.info(
                "vl_extraction_complete",
                url=url,
                title=result.title[:50] if result.title else None,
                confidence=result.confidence,
            )
            return result

    async def run(self, url: str, screenshot_only: bool = False) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            url: Target URL
            screenshot_only: If True, only capture screenshot

        Returns:
            Dict with extraction results or screenshot info
        """
        if screenshot_only:
            return await self.screenshot_only(url)

        result = await self.extract_from_url(url)
        return result.to_dict()

    def run_sync(self, url: str, screenshot_only: bool = False) -> Dict[str, Any]:
        """
        Synchronous wrapper for async pipeline.

        Note: This blocks the event loop. Use async run() in production.
        """
        import asyncio
        return asyncio.run(self.run(url, screenshot_only))

    def __del__(self):
        """Cleanup on deletion."""
        self._release_model()


class BatchWebPipeline:
    """
    Batch processing pipeline for multiple URLs.

    Example:
        pipeline = BatchWebPipeline()
        results = await pipeline.process_urls([
            "https://example.com/1",
            "https://example.com/2",
        ])
    """

    def __init__(self):
        self._pipeline = WebPagePipeline()
        self._results = []

    async def process_urls(self, urls: list) -> list:
        """
        Process multiple URLs.

        Args:
            urls: List of target URLs

        Returns:
            List of extraction results
        """
        logger.info("batch_processing_urls", count=len(urls))

        results = []
        for url in urls:
            try:
                result = await self._pipeline.extract_from_url(url)
                results.append(result)
            except Exception as e:
                logger.error("url_processing_failed", url=url, error=str(e))
                results.append(ExtractionResult(parse_error=str(e)))

        return results

    def process_urls_sync(self, urls: list) -> list:
        """
        Synchronous wrapper for batch processing.
        """
        import asyncio
        return asyncio.run(self.process_urls(urls))
