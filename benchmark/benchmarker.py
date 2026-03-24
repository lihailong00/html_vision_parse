"""
Performance benchmarking for web screenshot parsing.

Measures:
- Cold start time (model loading)
- Single image processing time
- End-to-end latency (URL -> screenshot -> extraction)
- Throughput (images per second)
- Memory usage
"""

import asyncio
import gc
import sys
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pathlib import Path

import structlog
from PIL import Image

# Add parent to path for imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import settings
from src.model_loader import ModelLoader
from src.inference import InferenceEngine
from src.extractor import ContentExtractor, ExtractionResult
from src.ocr_extractor import OCRExtractor

logger = structlog.get_logger()


@dataclass
class TimingResult:
    """Single timing measurement."""
    name: str
    duration_ms: float


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    num_runs: int
    warmup_runs: int

    # Timing statistics (in milliseconds)
    timings: List[float] = field(default_factory=list)

    # Memory stats (in MB)
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0

    # Additional metrics
    success_count: int = 0
    failure_count: int = 0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.timings) if self.timings else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.timings) if self.timings else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.timings:
            return 0.0
        sorted_timings = sorted(self.timings)
        idx = int(len(sorted_timings) * 0.95)
        return sorted_timings[min(idx, len(sorted_timings) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.timings:
            return 0.0
        sorted_timings = sorted(self.timings)
        idx = int(len(sorted_timings) * 0.99)
        return sorted_timings[min(idx, len(sorted_timings) - 1)]

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.timings) if len(self.timings) > 1 else 0.0

    @property
    def throughput(self) -> float:
        """Images per second."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Runs: {len(self.timings)}/{self.num_runs} (warmup: {self.warmup_runs})\n"
            f"  Mean: {self.mean_ms:.1f}ms | Median: {self.median_ms:.1f}ms\n"
            f"  P95: {self.p95_ms:.1f}ms | P99: {self.p99_ms:.1f}ms\n"
            f"  StdDev: {self.std_ms:.1f}ms\n"
            f"  Throughput: {self.throughput:.2f} img/s\n"
            f"  Success: {self.success_count} | Failed: {self.failure_count}\n"
            f"  GPU Memory: {self.avg_gpu_memory_mb:.0f}MB avg | {self.peak_gpu_memory_mb:.0f}MB peak"
        )


class Benchmark:
    """
    Benchmark runner for measuring inference performance.

    Usage:
        benchmark = Benchmark()
        result = benchmark.run_screenshot_benchmark(
            urls=["https://example.com"],
            warmup=3,
            runs=10,
        )
        print(result)
    """

    def __init__(self):
        from src.screenshot import ScreenshotCaptureLite
        self.screenshot_capture = ScreenshotCaptureLite()  # Use sync version
        self.model_loader: Optional[ModelLoader] = None
        self.engine: Optional[InferenceEngine] = None
        self.extractor: Optional[ContentExtractor] = None

    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            pass
        return 0.0

    def _load_model(self):
        """Load model if not already loaded."""
        if self.model_loader is None:
            self.model_loader = ModelLoader()
            self.model_loader.load()
            self.engine = InferenceEngine(self.model_loader)
            self.extractor = ContentExtractor(self.engine)

    def _unload_model(self):
        """Unload model to measure cold start."""
        if self.model_loader:
            self.model_loader.unload()
            self.model_loader = None
            self.engine = None
            self.extractor = None
            gc.collect()

    def measure_cold_start(self, runs: int = 3) -> BenchmarkResult:
        """
        Measure model cold start time (load + first inference).

        This measures the time from scratch to ready for inference.
        """
        timings = []
        screenshot = None
        image = None

        # Pre-download a screenshot to measure pure model load time
        logger.info("Preparing screenshot for cold start benchmark...")
        try:
            image = self.screenshot_capture.capture("https://example.com")
        except Exception as e:
            logger.warning("Screenshot failed", error=str(e))
            return BenchmarkResult(
                name="cold_start",
                num_runs=runs,
                warmup_runs=0,
                failure_count=runs,
            )

        for i in range(runs):
            logger.info("Cold start run", run=i+1, total=runs)

            # Unload first
            self._unload_model()

            # Measure load + first inference
            start = time.perf_counter()
            self._load_model()

            # First inference (cold)
            result = self.extractor.extract(image)
            duration = (time.perf_counter() - start) * 1000
            timings.append(duration)

            logger.info("Cold start complete", run=i+1, duration_ms=duration)

        self._unload_model()

        return BenchmarkResult(
            name="cold_start",
            num_runs=runs,
            warmup_runs=0,
            timings=timings,
            success_count=len(timings),
            failure_count=runs - len(timings),
            peak_gpu_memory_mb=self._get_gpu_memory_mb(),
        )

    def measure_screenshot_only(self, urls: List[str], warmup: int = 1, runs: int = 5) -> BenchmarkResult:
        """
        Measure screenshot capture performance only.
        """
        timings = []
        failures = 0

        # Warmup
        for _ in range(warmup):
            try:
                self.screenshot_capture.capture(urls[0])
            except Exception:
                pass

        for url in urls:
            for _ in range(runs):
                try:
                    start = time.perf_counter()
                    self.screenshot_capture.capture(url)
                    duration = (time.perf_counter() - start) * 1000
                    timings.append(duration)
                except Exception as e:
                    logger.warning("Screenshot failed", url=url, error=str(e))
                    failures += 1

        return BenchmarkResult(
            name="screenshot_only",
            num_runs=runs * len(urls),
            warmup_runs=warmup,
            timings=timings,
            success_count=len(timings),
            failure_count=failures,
        )

    def measure_inference_only(self, images: List[Image.Image], warmup: int = 1, runs: int = 5) -> BenchmarkResult:
        """
        Measure inference performance only (no screenshot).
        Assumes model is already loaded.
        """
        self._load_model()

        timings = []
        peak_memory = 0.0
        failures = 0

        # Warmup
        for _ in range(warmup):
            try:
                self.extractor.extract(images[0])
            except Exception:
                pass

        gc.collect()
        if hasattr(self, '_get_gpu_memory_mb'):
            peak_memory = self._get_gpu_memory_mb()

        for img in images:
            for _ in range(runs):
                try:
                    start = time.perf_counter()
                    self.extractor.extract(img)
                    duration = (time.perf_counter() - start) * 1000
                    timings.append(duration)

                    # Track peak memory
                    mem = self._get_gpu_memory_mb()
                    if mem > peak_memory:
                        peak_memory = mem
                except Exception as e:
                    logger.warning("Inference failed", error=str(e))
                    failures += 1

        self._unload_model()

        return BenchmarkResult(
            name="inference_only",
            num_runs=runs * len(images),
            warmup_runs=warmup,
            timings=timings,
            success_count=len(timings),
            failure_count=failures,
            peak_gpu_memory_mb=peak_memory,
        )

    async def measure_end_to_end(self, urls: List[str], warmup: int = 1, runs: int = 5) -> BenchmarkResult:
        """
        Measure end-to-end performance: URL -> screenshot -> extraction.

        This is the most realistic benchmark for actual usage.
        """
        timings = []
        peak_memory = 0.0
        failures = 0

        # Warmup
        for _ in range(warmup):
            try:
                self._load_model()
                await self.screenshot_capture.capture(urls[0])
                self.extractor.extract(await self.screenshot_capture.capture(urls[0]))
            except Exception:
                pass

        gc.collect()

        for url in urls:
            for _ in range(runs):
                try:
                    self._load_model()

                    start = time.perf_counter()
                    image = await self.screenshot_capture.capture(url)
                    result = self.extractor.extract(image)
                    duration = (time.perf_counter() - start) * 1000
                    timings.append(duration)

                    # Track peak memory
                    mem = self._get_gpu_memory_mb()
                    if mem > peak_memory:
                        peak_memory = mem

                    self._unload_model()
                    gc.collect()

                except Exception as e:
                    logger.warning("End-to-end failed", url=url, error=str(e))
                    failures += 1

        return BenchmarkResult(
            name="end_to_end",
            num_runs=runs * len(urls),
            warmup_runs=warmup,
            timings=timings,
            success_count=len(timings),
            failure_count=failures,
            peak_gpu_memory_mb=peak_memory,
            avg_gpu_memory_mb=peak_memory / 2 if peak_memory > 0 else 0,
        )

    def run_screenshot_benchmark(
        self,
        urls: List[str],
        warmup: int = 2,
        runs: int = 10,
        category: str = "benchmark",
    ) -> BenchmarkResult:
        """
        Run a complete screenshot + extraction benchmark.

        Args:
            urls: List of URLs to test
            warmup: Number of warmup runs (not counted)
            runs: Number of measured runs per URL
            category: Name for this benchmark

        Returns:
            BenchmarkResult with statistics
        """
        logger.info("Starting benchmark", urls=len(urls), warmup=warmup, runs=runs)

        # Pre-capture all screenshots
        logger.info("Pre-capturing screenshots...")
        images = []
        for url in urls:
            try:
                img = self.screenshot_capture.capture(url)
                images.append(img)
                logger.info("Screenshot captured", url=url, size=img.size)
            except Exception as e:
                logger.warning("Screenshot failed, skipping", url=url, error=str(e))

        if not images:
            logger.error("No screenshots captured, benchmark aborted")
            return BenchmarkResult(
                name=category,
                num_runs=runs * len(urls),
                warmup_runs=warmup,
                failure_count=runs * len(urls),
            )

        # Measure inference
        result = self.measure_inference_only(images, warmup=warmup, runs=runs)
        result.name = category

        logger.info("Benchmark complete", result=str(result))
        return result

    def compare_quantization(self, image: Image.Image, quantizations: List[str] = None) -> dict:
        """
        Compare performance across different quantization settings.

        Note: This requires restarting with different settings.
        """
        if quantizations is None:
            quantizations = ["none", "int8", "int4"]

        results = {}
        for quant in quantizations:
            logger.info("Testing quantization", quant=quant)
            # In practice, you'd need to reload with different settings
            # This is a placeholder for the comparison logic
            results[quant] = {"note": "Requires model reload to test"}
        return results

    def measure_ocr_only(self, images: List[Image.Image], warmup: int = 1, runs: int = 5) -> BenchmarkResult:
        """
        Measure OCR-only performance (without VL model).

        This measures the fast OCR path that filters noise by position.
        """
        ocr_extractor = OCRExtractor(inference_engine=None)  # No VL fallback

        timings = []
        confidences = []
        failures = 0

        # Warmup
        for _ in range(warmup):
            try:
                ocr_extractor.extract_ocr(images[0])
            except Exception:
                pass

        for img in images:
            for _ in range(runs):
                try:
                    start = time.perf_counter()
                    result = ocr_extractor.extract_ocr(img)
                    duration = (time.perf_counter() - start) * 1000
                    timings.append(duration)
                    confidences.append(result.confidence)
                except Exception as e:
                    logger.warning("OCR failed", error=str(e))
                    failures += 1

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return BenchmarkResult(
            name="ocr_only",
            num_runs=runs * len(images),
            warmup_runs=warmup,
            timings=timings,
            success_count=len(timings),
            failure_count=failures,
            peak_gpu_memory_mb=0.0,  # OCR doesn't use GPU
            avg_gpu_memory_mb=0.0,
        )

    def measure_hybrid_comparison(
        self,
        images: List[Image.Image],
        warmup: int = 1,
        runs: int = 5,
    ) -> dict:
        """
        Compare OCR vs VL vs Hybrid extraction methods.

        Returns dict with results for each method.
        """
        self._load_model()
        ocr_extractor = OCRExtractor(inference_engine=self.engine)

        results = {}

        # OCR only
        logger.info("Benchmarking OCR-only...")
        ocr_timings = []
        ocr_confs = []
        for img in images:
            for _ in range(warmup):
                ocr_extractor.extract_ocr(img)
            for _ in range(runs):
                start = time.perf_counter()
                result = ocr_extractor.extract_ocr(img)
                ocr_timings.append((time.perf_counter() - start) * 1000)
                ocr_confs.append(result.confidence)

        results["ocr_only"] = {
            "timings": ocr_timings,
            "mean_ms": statistics.mean(ocr_timings) if ocr_timings else 0,
            "avg_confidence": statistics.mean(ocr_confs) if ocr_confs else 0,
        }

        # VL only (pure vision model)
        logger.info("Benchmarking VL-only...")
        vl_timings = []
        vl_confs = []
        for img in images:
            for _ in range(warmup):
                self.extractor.extract(img)
            for _ in range(runs):
                start = time.perf_counter()
                result = self.extractor.extract(img)
                vl_timings.append((time.perf_counter() - start) * 1000)
                vl_confs.append(result.confidence)

        results["vl_only"] = {
            "timings": vl_timings,
            "mean_ms": statistics.mean(vl_timings) if vl_timings else 0,
            "avg_confidence": statistics.mean(vl_confs) if vl_confs else 0,
        }

        # Hybrid (OCR with VL fallback)
        logger.info("Benchmarking Hybrid...")
        hybrid_timings = []
        hybrid_vl_fallbacks = 0
        hybrid_confs = []

        for img in images:
            for _ in range(warmup):
                ocr_extractor.extract_with_fallback(img)
            for _ in range(runs):
                start = time.perf_counter()
                result, used_vl = ocr_extractor.extract_with_fallback(img)
                hybrid_timings.append((time.perf_counter() - start) * 1000)
                hybrid_confs.append(result.confidence)
                if used_vl:
                    hybrid_vl_fallbacks += 1

        results["hybrid"] = {
            "timings": hybrid_timings,
            "mean_ms": statistics.mean(hybrid_timings) if hybrid_timings else 0,
            "avg_confidence": statistics.mean(hybrid_confs) if hybrid_confs else 0,
            "vl_fallbacks": hybrid_vl_fallbacks,
            "fallback_rate": hybrid_vl_fallbacks / len(hybrid_timings) if hybrid_timings else 0,
        }

        self._unload_model()

        return results


def print_benchmark_results(results: List[BenchmarkResult]):
    """Pretty print multiple benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\n{result}")
        print("-" * 70)

    # Summary table
    print("\nSUMMARY TABLE")
    print("-" * 70)
    print(f"{'Benchmark':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<15} {'Success'}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<20} {r.mean_ms:<12.1f} {r.p95_ms:<12.1f} {r.throughput:<15.2f} {r.success_count}/{r.num_runs}")
    print("=" * 70)
