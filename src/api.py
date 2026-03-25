"""FastAPI service for web screenshot parsing with multi-GPU support."""

import os
import io
import json
import time
import uuid
import multiprocessing
from pathlib import Path
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

from loguru import logger
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from .model_loader import ModelLoader
from .inference import InferenceEngine
from .extractor import ContentExtractor, ExtractionResult
from .pipeline import WebPagePipeline, get_available_gpus
from .html_renderer import HTMLRenderer
from config.settings import settings


def get_worker_id() -> int:
    """
    Get worker ID from environment.
    Uvicorn sets WORKER_TIMEOUT env var when using multiple workers.
    Also check for commonly used worker ID environment variables.
    """
    # Uvicorn master process
    if "UVICORN_WORKER_ID" in os.environ:
        return int(os.environ["UVICORN_WORKER_ID"])

    # Gunicorn workers
    if "WORKER" in os.environ:
        return int(os.environ["WORKER"])

    # Custom: worker ID passed via environment
    if "WORKER_ID" in os.environ:
        return int(os.environ["WORKER_ID"])

    # Fallback: try to detect from process info
    return 0


def assign_gpu_to_worker(worker_id: int) -> int:
    """
    Assign a GPU to a worker based on worker ID.
    Round-robin assignment across available GPUs.
    """
    available_gpus = get_available_gpus()

    if not available_gpus:
        logger.warning("no_gpus_available_using_cpu")
        return -1  # CPU

    # Round-robin assignment
    gpu_id = available_gpus[worker_id % len(available_gpus)]

    logger.info("worker_gpu_assigned", worker_id=worker_id, gpu_id=gpu_id, available_gpus=available_gpus)

    return gpu_id


# Global pipeline instance (lazy loaded per worker)
_pipeline: Optional[WebPagePipeline] = None
_html_renderer: Optional[HTMLRenderer] = None
_worker_gpu_id: int = 0


def get_pipeline() -> WebPagePipeline:
    """Get or create the global pipeline instance for this worker."""
    global _pipeline, _worker_gpu_id

    if _pipeline is None:
        # Assign GPU based on worker ID
        worker_id = get_worker_id()
        _worker_gpu_id = assign_gpu_to_worker(worker_id)

        _pipeline = WebPagePipeline(
            gpu_id=_worker_gpu_id,
            extraction_method=settings.extraction.extraction_method
        )
        logger.info("pipeline_initialized", worker_id=worker_id, gpu_id=_worker_gpu_id)

    return _pipeline


def get_html_renderer() -> HTMLRenderer:
    """Get or create the global HTML renderer instance."""
    global _html_renderer
    if _html_renderer is None:
        _html_renderer = HTMLRenderer()
    return _html_renderer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _pipeline, _html_renderer

    worker_id = get_worker_id()
    logger.info("api_starting", host=settings.api.host, port=settings.api.port, worker_id=worker_id, gpu_id=_worker_gpu_id)

    yield

    # Shutdown
    logger.info("api_shutting_down", worker_id=worker_id)
    if _pipeline is not None:
        _pipeline._release_model()
        _pipeline = None
    _html_renderer = None
    logger.info("api_shutdown_complete", worker_id=worker_id)


app = FastAPI(
    title="HTML Vision Parse API",
    description="Extract structured content from web screenshots using VL models",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models
class ExtractRequest(BaseModel):
    """Request for URL-based extraction."""
    url: str = Field(..., description="Target URL to extract from")
    fields: Optional[List[str]] = Field(default=None, description="Fields to extract: title, content, publish_time")
    methods: Optional[dict] = Field(default=None, description="Per-field method: {\"title\": \"vl\", \"content\": \"ocr\"}")


class ExtractResponse(BaseModel):
    """Response for extraction."""
    success: bool
    title: Optional[str] = None
    content: Optional[str] = None
    publish_time: Optional[str] = None
    confidence: Optional[float] = None
    extraction_method: Optional[str] = None
    source_url: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
    gpu_id: Optional[int] = None


class ExtractionResponse(BaseModel):
    """Single extraction response (legacy compatibility)."""
    title: Optional[str] = None
    content: Optional[str] = None
    publish_time: Optional[str] = None
    confidence: float = 0.0
    regions_ignored: List[str] = Field(default_factory=list)
    is_high_confidence: bool = False
    parse_error: Optional[str] = None


class BatchExtractionResponse(BaseModel):
    """Batch extraction response."""
    request_id: str
    total: int
    results: List[ExtractionResponse]
    stats: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"
    gpu_id: int = -1
    available_gpus: List[int] = Field(default_factory=list)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline._engine is not None,
        gpu_id=_worker_gpu_id,
        available_gpus=get_available_gpus(),
    )


@app.get("/info")
async def get_info():
    """Get worker and GPU information."""
    return {
        "worker_id": get_worker_id(),
        "gpu_id": _worker_gpu_id,
        "available_gpus": get_available_gpus(),
        "extraction_method": settings.extraction.extraction_method,
        "model_name": settings.model.name,
        "inference_framework": settings.model.inference_framework,
    }


# ============ API v1 Endpoints ============

@app.post("/api/v1/extract", response_model=ExtractResponse)
async def extract_from_url(request: ExtractRequest):
    """
    Extract content from a URL.

    Supports:
    - Full extraction (title, content, publish_time)
    - Field-specific extraction with methods
    """
    start_time = time.time()

    try:
        pipeline = get_pipeline()

        result = await pipeline.extract_from_url(
            url=request.url,
            fields=request.fields,
            methods=request.methods,
        )

        processing_time = (time.time() - start_time) * 1000

        return ExtractResponse(
            success=True,
            title=result.title,
            content=result.content,
            publish_time=result.publish_time,
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            source_url=result.source_url,
            processing_time_ms=processing_time,
            gpu_id=_worker_gpu_id,
        )

    except Exception as e:
        logger.error("extraction_failed", url=request.url, error=str(e), gpu_id=_worker_gpu_id)
        processing_time = (time.time() - start_time) * 1000
        return ExtractResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
            gpu_id=_worker_gpu_id,
        )


@app.post("/api/v1/extract/html", response_model=ExtractResponse)
async def extract_from_html(
    file: UploadFile = File(..., description="HTML file to extract from"),
    fields: Optional[str] = Form(default=None, description="Comma-separated fields: title,content,publish_time"),
    methods: Optional[str] = Form(default=None, description="JSON dict: {\"title\": \"vl\", \"content\": \"ocr\"}"),
):
    """
    Extract content from uploaded HTML file.
    """
    start_time = time.time()

    try:
        # Read uploaded file
        html_content = await file.read()
        html_text = html_content.decode('utf-8')

        # Parse fields
        field_list = None
        if fields:
            field_list = [f.strip() for f in fields.split(',')]

        # Parse methods
        method_dict = None
        if methods:
            try:
                method_dict = json.loads(methods)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid methods JSON")

        # Render HTML to image
        renderer = get_html_renderer()
        image = await renderer.render_from_html(html_text)

        # Extract using pipeline
        pipeline = get_pipeline()
        result = await pipeline.extract_from_url(
            url="html://upload",
            fields=field_list,
            methods=method_dict,
        )

        processing_time = (time.time() - start_time) * 1000

        return ExtractResponse(
            success=True,
            title=result.title,
            content=result.content,
            publish_time=result.publish_time,
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            processing_time_ms=processing_time,
            gpu_id=_worker_gpu_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("html_extraction_failed", error=str(e), gpu_id=_worker_gpu_id)
        processing_time = (time.time() - start_time) * 1000
        return ExtractResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
            gpu_id=_worker_gpu_id,
        )


@app.post("/api/v1/batch", response_model=List[ExtractResponse])
async def batch_extract(urls: List[str]):
    """
    Batch extract from multiple URLs.

    Note: This processes URLs sequentially. For high throughput,
    use the batch CLI instead.
    """
    start_time = time.time()
    results = []

    pipeline = get_pipeline()

    for url in urls:
        try:
            result = await pipeline.extract_from_url(url)
            results.append(ExtractResponse(
                success=True,
                title=result.title,
                content=result.content,
                publish_time=result.publish_time,
                confidence=result.confidence,
                extraction_method=result.extraction_method,
                source_url=result.source_url,
                gpu_id=_worker_gpu_id,
            ))
        except Exception as e:
            logger.error("batch_url_failed", url=url, error=str(e))
            results.append(ExtractResponse(success=False, error=str(e), gpu_id=_worker_gpu_id))

    processing_time = (time.time() - start_time) * 1000
    logger.info("batch_complete", count=len(urls), total_time_ms=processing_time, gpu_id=_worker_gpu_id)

    return results


# ============ Legacy Endpoints (for backward compatibility) ============

@app.post("/extract", response_model=ExtractionResponse)
async def extract_content(file: UploadFile = File(...)):
    """Extract from uploaded image file."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_types = {"image/png", "image/jpeg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        pipeline = get_pipeline()
        pipeline._ensure_model_loaded()
        result = pipeline._extractor.extract(image)

        return ExtractionResponse(
            title=result.title,
            content=result.content,
            publish_time=result.publish_time,
            confidence=result.confidence,
            regions_ignored=result.regions_ignored,
            is_high_confidence=result.is_high_confidence,
            parse_error=result.parse_error,
        )

    except Exception as e:
        logger.error("extraction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape", response_model=ExtractionResponse)
async def scrape_url(request: dict):
    """Scrape URL and extract content."""
    url = request.get("url")
    screenshot_only = request.get("screenshot_only", False)

    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' field")

    logger.info("scrape_request", url=url, screenshot_only=screenshot_only)

    try:
        pipeline = get_pipeline()

        if screenshot_only:
            result = await pipeline.screenshot_only(url)
            return {"url": url, "width": result["width"], "height": result["height"]}
        else:
            result = await pipeline.extract_from_url(url)
            return ExtractionResponse(
                title=result.title,
                content=result.content,
                publish_time=result.publish_time,
                confidence=result.confidence,
                regions_ignored=result.regions_ignored,
                is_high_confidence=result.is_high_confidence,
                parse_error=result.parse_error,
            )

    except Exception as e:
        logger.error("scrape_failed", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    """Run the FastAPI server."""
    import uvicorn

    logger.info("starting_server",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        available_gpus=get_available_gpus(),
    )

    uvicorn.run(
        "src.api:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
    )


if __name__ == "__main__":
    run_server()
