"""Pipeline integrating browser, screenshot, and extraction."""

import os
from typing import Optional, Dict, Any, List, Literal
from loguru import logger

from .browser import BrowserContext
from .screenshot import ScreenshotCapture
from .model_loader import ModelLoader
from .inference import InferenceEngine
from .vllm_engine import VLLMEngine
from .extractor import ContentExtractor, ExtractionResult
from .ocr_extractor import OCRExtractor
from .flexible_extractor import FlexibleExtractor
from config.settings import settings


def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs."""
    if settings.gpu.visible_devices:
        return settings.gpu.visible_devices

    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass

    return [0]  # Default to GPU 0


def set_gpu_device(device_id: int):
    """Set the current process to use a specific GPU."""
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            logger.info("gpu_assigned", device_id=device_id, device_name=torch.cuda.get_device_name(device_id))
    except Exception as e:
        logger.warning("failed_to_set_gpu", device_id=device_id, error=str(e))


class WebPagePipeline:
    """
    End-to-end pipeline: URL -> Screenshot -> Extraction.

    Example:
        pipeline = WebPagePipeline()
        result = await pipeline.run("https://example.com/article")
        print(result.title, result.content, result.publish_time)

    GPU Selection:
        # By default, uses first available GPU (or GPU 0)
        pipeline = WebPagePipeline(gpu_id=1)  # Use GPU 1

    Extraction Methods:
        # Use OCR for all fields
        pipeline = WebPagePipeline(extraction_method="ocr")

        # Use VL for all fields
        pipeline = WebPagePipeline(extraction_method="vl")

        # Per-field method selection
        result = await pipeline.extract_from_url(
            "https://example.com",
            fields=["title", "content"],
            methods={"title": "vl", "content": "ocr"}
        )
    """

    def __init__(
        self,
        gpu_id: int = None,
        extraction_method: Literal["vl", "ocr"] = None,
    ):
        """
        Initialize pipeline.

        Args:
            gpu_id: GPU device ID to use. None = auto-select based on worker ID.
            extraction_method: "vl" or "ocr". None = use settings default.
        """
        self._screenshot = ScreenshotCapture()
        self._model_loader = None
        self._vllm_engine = None
        self._engine = None
        self._extractor = None
        self._ocr_extractor = None
        self._flexible_extractor = None

        # GPU assignment
        available_gpus = get_available_gpus()
        if gpu_id is None:
            # Default: use first available GPU
            self._gpu_id = available_gpus[0] if available_gpus else 0
        else:
            if gpu_id not in available_gpus:
                logger.warning("gpu_id_not_available", requested=gpu_id, available=available_gpus)
                self._gpu_id = available_gpus[0] if available_gpus else 0
            else:
                self._gpu_id = gpu_id

        # Extraction method
        self._extraction_method = extraction_method or settings.extraction.extraction_method

        # Initialize extractors
        self._init_ocr_extractor()

    def _init_ocr_extractor(self):
        """Initialize OCR extractor (always available)."""
        self._ocr_extractor = OCRExtractor(inference_engine=None)

    def _ensure_model_loaded(self):
        """Lazy load the VL model if needed."""
        if self._engine is not None:
            return

        # Set GPU before loading model
        set_gpu_device(self._gpu_id)

        if settings.model.inference_framework == "vllm":
            self._vllm_engine = VLLMEngine(settings.model.name)
            self._vllm_engine.load()
            self._engine = self._vllm_engine
        else:
            self._model_loader = ModelLoader()
            self._model_loader.load()
            self._engine = InferenceEngine(self._model_loader)

        self._extractor = ContentExtractor(self._engine)
        self._flexible_extractor = FlexibleExtractor(
            ocr_extractor=self._ocr_extractor,
            vl_engine=self._engine
        )

        logger.info("model_loaded", gpu_id=self._gpu_id, method=self._extraction_method)

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
        self._flexible_extractor = None

        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
        fields: List[str] = None,
        methods: Dict[str, Literal["vl", "ocr"]] = None,
    ) -> ExtractionResult:
        """
        Extract content from a URL (screenshot + extract).

        Args:
            url: Target URL
            fields: List of fields to extract ["title", "content", "publish_time"]
                   If None, extracts all fields.
            methods: Dict mapping field -> method {"vl" or "ocr"}
                   If None, uses settings.extraction.field_methods or extraction_method

        Returns:
            ExtractionResult with title, content, publish_time
        """
        logger.info("extracting_from_url", url=url, fields=fields, methods=methods, gpu_id=self._gpu_id)

        # Capture screenshot
        image = await self._screenshot.capture(url)

        # If OCR method specified, use OCR directly (no VL model needed)
        if self._extraction_method == "ocr" and methods is None:
            return self._extract_with_ocr(image, url)

        # If VL method or custom methods, need VL model
        if methods is not None:
            # Check if any method is VL
            use_vl = any(m == "vl" for m in methods.values())
        else:
            use_vl = self._extraction_method == "vl"

        if use_vl:
            self._ensure_model_loaded()
            result = self._flexible_extractor.extract_fields(
                image, fields=fields, methods=methods
            )
            result.source_url = url
            result.extraction_method = methods.get("content", self._extraction_method) if methods else self._extraction_method
            return result

        # Pure OCR
        return self._extract_with_ocr(image, url)

    def _extract_with_ocr(self, image, url: str) -> ExtractionResult:
        """Extract using OCR only (no VL model)."""
        ocr_result = self._ocr_extractor.extract_ocr(image)

        result = ExtractionResult(
            title=self._ocr_extractor.extract_title(ocr_result.blocks, image.height),
            content=ocr_result.full_text,
            publish_time=self._ocr_extractor.extract_time(ocr_result.blocks),
            confidence=ocr_result.confidence,
            extraction_method="ocr",
        )
        result.source_url = url

        logger.info(
            "ocr_extraction_done",
            gpu_id=self._gpu_id,
            confidence=result.confidence,
            title=result.title[:50] if result.title else None,
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
