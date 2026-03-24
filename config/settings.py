"""Configuration settings for the web screenshot parser."""

from pathlib import Path
from pydantic import BaseModel
from typing import Literal


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# Model configuration
class ModelConfig(BaseModel):
    # Model type: "qwen3_vl" (2B, 4B, 8B) or "internvl3" (1B)
    model_type: Literal["qwen3_vl", "internvl3"] = "qwen3_vl"
    name: str = "/home/longcoding/dev/models/Qwen3-VL-2B-Instruct"
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: str = "bfloat16"
    max_batch_size: int = 4  # Adjust based on VRAM
    quantization: Literal["none", "int8", "int4"] = "int4"  # INT4 for more speed
    trust_remote_code: bool = True
    # Inference framework: "transformers" or "vllm"
    inference_framework: Literal["transformers", "vllm"] = "transformers"
    # InternVL model path (when model_type is "internvl3")
    internvl_model_path: str = "/home/longcoding/dev/models/InternVL3-1B"


# Extraction configuration
class ExtractionConfig(BaseModel):
    min_confidence: float = 0.85
    enable_cross_validation: bool = False  # Enable for higher accuracy
    max_retries: int = 2
    timeout_seconds: int = 30
    # Field extraction method: "vl" (vision language) or "ocr"
    # Can be set globally or per-field
    # Global default: "vl" or "ocr" or "hybrid"
    extraction_method: Literal["vl", "ocr", "hybrid"] = "vl"
    # Per-field method override: {"title": "vl", "content": "ocr", "publish_time": "vl"}
    # If not set, uses extraction_method as default
    field_methods: dict = {}


# Browser configuration
class BrowserConfig(BaseModel):
    type: Literal["chromium", "firefox", "webkit"] = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    full_page: bool = True
    wait_until: Literal["load", "domcontentloaded", "networkidle"] = "networkidle"
    wait_timeout: int = 30000  # ms
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# API configuration
class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_concurrent_requests: int = 10


# OCR configuration
class OCRConfig(BaseModel):
    enabled: bool = True
    min_confidence: float = 0.75  # Fallback to VL below this
    use_easyocr: bool = True  # Use EasyOCR vs PaddleOCR
    fallback_to_vl: bool = True  # Use VL when OCR confidence is low
    ocr_timeout_seconds: int = 10


# Batch processing configuration
class BatchConfig(BaseModel):
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 2


# All configurations
class Settings(BaseModel):
    model: ModelConfig = ModelConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    browser: BrowserConfig = BrowserConfig()
    api: APIConfig = APIConfig()
    batch: BatchConfig = BatchConfig()
    ocr: OCRConfig = OCRConfig()


# Global settings instance
settings = Settings()
