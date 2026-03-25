"""Batch processing for web screenshots."""

import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from PIL import Image
from tqdm import tqdm

from .model_loader import ModelLoader
from .inference import InferenceEngine
from .vllm_engine import VLLMEngine
from .extractor import ContentExtractor, ExtractionResult
from config.settings import settings




class BatchProcessor:
    """
    Handles batch processing of web screenshots.

    Supports:
    - Parallel processing
    - Progress tracking
    - Result caching
    - Error recovery
    - Both Transformers and vLLM backends
    """

    def __init__(self, use_vllm: bool = None):
        """
        Initialize batch processor.

        Args:
            use_vllm: Override inference framework. None uses settings.default.
        """
        self.use_vllm = use_vllm if use_vllm is not None else (
            settings.model.inference_framework == "vllm"
        )
        self.model_loader = ModelLoader()
        self.vllm_engine = None
        self._cache: dict = {}
        self._stats = {"processed": 0, "failed": 0, "cached": 0}

    def _get_engine(self):
        """Get the inference engine based on settings."""
        if self.use_vllm:
            if self.vllm_engine is None:
                self.vllm_engine = VLLMEngine(settings.model.name)
                self.vllm_engine.load()
            return self.vllm_engine
        else:
            if not self.model_loader.is_loaded:
                self.model_loader.load()
            return InferenceEngine(self.model_loader)

    def _release_engine(self):
        """Release the engine resources."""
        if self.use_vllm and self.vllm_engine:
            self.vllm_engine.unload()
            self.vllm_engine = None
        elif not self.use_vllm and self.model_loader.is_loaded:
            self.model_loader.unload()

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key from image path."""
        return hashlib.md5(Path(image_path).read_bytes()).hexdigest()

    def _load_image(self, path: str) -> Image.Image:
        """Load an image from path."""
        return Image.open(path).convert("RGB")

    def process_directory(
        self,
        directory: str,
        output_path: Optional[str] = None,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".webp"),
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ExtractionResult]:
        """
        Process all images in a directory.

        Args:
            directory: Path to directory containing images
            output_path: Optional path to save results as JSON
            extensions: File extensions to process
            progress_callback: Optional callback(processed, total)

        Returns:
            List of extraction results
        """
        directory = Path(directory)
        image_files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]

        logger.info("processing_directory", path=str(directory), count=len(image_files), use_vllm=self.use_vllm)

        # Load model
        engine = self._get_engine()
        extractor = ContentExtractor(engine)

        results = []
        failed = []

        try:
            for i, img_path in enumerate(tqdm(image_files, desc="Processing")):
                try:
                    # Check cache
                    cache_key = self._get_cache_key(str(img_path))
                    if cache_key in self._cache:
                        results.append(self._cache[cache_key])
                        self._stats["cached"] += 1
                        continue

                    # Process image
                    image = self._load_image(str(img_path))
                    result = extractor.extract_with_retry(image)
                    result.source_path = str(img_path)

                    if result.parse_error is None:
                        results.append(result)
                        self._cache[cache_key] = result
                        self._stats["processed"] += 1
                    else:
                        failed.append((img_path, result))
                        self._stats["failed"] += 1

                    if progress_callback:
                        progress_callback(i + 1, len(image_files))

                except Exception as e:
                    logger.error("image_processing_failed", path=str(img_path), error=str(e))
                    self._stats["failed"] += 1

        finally:
            self._release_engine()

        # Save results if output path specified
        if output_path:
            self._save_results(results, output_path)

        logger.info(
            "batch_processing_complete",
            successful=len(results),
            failed=len(failed),
            cached=self._stats["cached"],
        )

        return results

    def process_list(
        self,
        image_paths: List[str],
        parallel: bool = True,
    ) -> List[ExtractionResult]:
        """
        Process a list of image paths.

        Args:
            image_paths: List of image file paths
            parallel: Whether to process in parallel

        Returns:
            List of extraction results
        """
        if parallel:
            return self._process_parallel(image_paths)
        else:
            return self._process_sequential(image_paths)

    def _process_sequential(self, image_paths: List[str]) -> List[ExtractionResult]:
        """Process images sequentially."""
        engine = self._get_engine()
        extractor = ContentExtractor(engine)

        results = []
        try:
            for path in tqdm(image_paths, desc="Processing"):
                try:
                    image = self._load_image(path)
                    result = extractor.extract_with_retry(image)
                    result.source_path = path
                    results.append(result)
                except Exception as e:
                    logger.error("processing_failed", path=path, error=str(e))
                    results.append(ExtractionResult(parse_error=str(e)))
        finally:
            self._release_engine()

        return results

    def _process_parallel(self, image_paths: List[str]) -> List[ExtractionResult]:
        """Process images in parallel using thread pool."""
        # For GPU inference, we use sequential to avoid memory issues
        # but could use async for I/O-bound operations
        return self._process_sequential(image_paths)

    def _save_results(self, results: List[ExtractionResult], output_path: str) -> None:
        """Save results to JSON file."""
        import json

        output_data = [
            {
                "source_path": getattr(r, "source_path", None),
                **r.to_dict(),
            }
            for r in results
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info("results_saved", path=output_path, count=len(results))

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self._stats.copy()

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        logger.info("cache_cleared")
