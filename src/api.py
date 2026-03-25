"""FastAPI service for web screenshot parsing."""

import io
import json
import time
import uuid
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
from .pipeline import WebPagePipeline
from .html_renderer import HTMLRenderer
from config.settings import settings



# Global pipeline instances (lazy loaded)
_pipeline: Optional[WebPagePipeline] = None
_html_renderer: Optional[HTMLRenderer] = None


def get_pipeline() -> WebPagePipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = WebPagePipeline()
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

    # Startup
    logger.info("api_starting", host=settings.api.host, port=settings.api.port)

    yield

    # Shutdown
    logger.info("api_shutting_down")
    if _pipeline is not None:
        _pipeline._release_model()
        _pipeline = None
    _html_renderer = None
    logger.info("api_shutdown_complete")


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
    use_hybrid: Optional[bool] = Field(default=None, description="Use hybrid OCR+VL extraction")


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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline._engine is not None,
    )


# ============ New API v1 Endpoints ============

@app.post("/api/v1/extract", response_model=ExtractResponse)
async def extract_from_url(request: ExtractRequest):
    """
    Extract content from a URL.

    Supports:
    - Full extraction (title, content, publish_time)
    - Field-specific extraction with methods
    - Hybrid OCR+VL mode
    """
    start_time = time.time()

    try:
        pipeline = get_pipeline()

        result = await pipeline.extract_from_url(
            url=request.url,
            fields=request.fields,
            methods=request.methods,
            use_hybrid=request.use_hybrid,
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
        )

    except Exception as e:
        logger.error("extraction_failed", url=request.url, error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return ExtractResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
        )


@app.post("/api/v1/extract/html", response_model=ExtractResponse)
async def extract_from_html(
    file: UploadFile = File(..., description="HTML file to extract from"),
    fields: Optional[str] = Form(default=None, description="Comma-separated fields: title,content,publish_time"),
    methods: Optional[str] = Form(default=None, description="JSON dict: {\"title\": \"vl\", \"content\": \"ocr\"}"),
    use_hybrid: Optional[bool] = Form(default=None),
):
    """
    Extract content from uploaded HTML file.

    Upload an HTML file and extract structured content from it.
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
        pipeline._ensure_model_loaded()

        if method_dict or field_list:
            result = pipeline._flexible_extractor.extract_fields(
                image, fields=field_list, methods=method_dict
            )
        elif use_hybrid or (use_hybrid is None and pipeline._use_hybrid):
            ocr_result, used_vl = pipeline._ocr_extractor.extract_with_fallback(
                image,
                min_confidence=settings.ocr.min_confidence
            )
            if not used_vl:
                result = ExtractionResult(
                    title=pipeline._ocr_extractor.extract_title(ocr_result.blocks, image.height),
                    content=ocr_result.full_text,
                    publish_time=pipeline._ocr_extractor.extract_time(ocr_result.blocks),
                    confidence=ocr_result.confidence,
                    extraction_method="ocr",
                )
            else:
                result = pipeline._extractor.extract(image)
                result.extraction_method = "vl"
        else:
            result = pipeline._extractor.extract(image)
            result.extraction_method = "vl"

        processing_time = (time.time() - start_time) * 1000

        return ExtractResponse(
            success=True,
            title=result.title,
            content=result.content,
            publish_time=result.publish_time,
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("html_extraction_failed", error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return ExtractResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
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
            ))
        except Exception as e:
            logger.error("batch_url_failed", url=url, error=str(e))
            results.append(ExtractResponse(success=False, error=str(e)))

    processing_time = (time.time() - start_time) * 1000
    logger.info("batch_complete", count=len(urls), total_time_ms=processing_time)

    return results


@app.post("/extract", response_model=ExtractionResponse)
async def extract_content(file: UploadFile = File(...)):
    """
    Extract content from a single screenshot.

    Upload an image file and receive structured extraction results.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}",
        )

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Extract content using pipeline
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


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Extract content from multiple screenshots.

    Upload multiple image files and receive extraction results.
    Note: For large batches, consider using the batch processing CLI instead.
    """
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per batch. For larger batches, use batch CLI.",
        )

    pipeline = get_pipeline()
    pipeline._ensure_model_loaded()

    request_id = str(uuid.uuid4())
    results = []
    failed = 0

    for file in files:
        try:
            # Validate file type
            allowed_types = {"image/png", "image/jpeg", "image/webp"}
            if file.content_type not in allowed_types:
                results.append(ExtractionResponse(
                    title=None,
                    confidence=0.0,
                    parse_error=f"Invalid file type: {file.content_type}",
                ))
                failed += 1
                continue

            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Extract content
            result = pipeline._extractor.extract(image)

            results.append(ExtractionResponse(
                title=result.title,
                content=result.content,
                publish_time=result.publish_time,
                confidence=result.confidence,
                regions_ignored=result.regions_ignored,
                is_high_confidence=result.is_high_confidence,
                parse_error=result.parse_error,
            ))

        except Exception as e:
            logger.error("batch_extraction_failed", file=file.filename, error=str(e))
            results.append(ExtractionResponse(
                title=None,
                confidence=0.0,
                parse_error=str(e),
            ))
            failed += 1

    return BatchExtractionResponse(
        request_id=request_id,
        total=len(files),
        results=results,
        stats={
            "successful": len(files) - failed,
            "failed": failed,
        },
    )


@app.post("/extract/base64")
async def extract_from_base64(image_data: dict):
    """
    Extract content from a base64-encoded image.

    Body: {"image": "base64_encoded_image_data", "format": "png/jpeg"}
    """
    import base64

    if "image" not in image_data:
        raise HTTPException(status_code=400, detail="Missing 'image' field")

    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Extract content using pipeline
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
        logger.error("base64_extraction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class ScrapeRequest(BaseModel):
    """Request model for scrape endpoint."""
    url: str
    screenshot_only: bool = False


class ScreenshotResponse(BaseModel):
    """Response model for screenshot-only scrape."""
    url: str
    width: int
    height: int
    saved_to: Optional[str] = None


@app.post("/scrape", response_model=ExtractionResponse)
async def scrape_url(request: ScrapeRequest):
    """
    Scrape a URL: open in browser, capture screenshot, and extract content.

    Provide a URL and optionally just capture a screenshot without extraction.
    """
    logger.info("scrape_request", url=request.url, screenshot_only=request.screenshot_only)

    try:
        pipeline = get_pipeline()

        if request.screenshot_only:
            # Screenshot only
            result = await pipeline.run(request.url, screenshot_only=True)
            return ScreenshotResponse(**result)
        else:
            # Full extraction
            result = await pipeline.extract_from_url(request.url)

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
        logger.error("scrape_failed", url=request.url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        workers=settings.api.workers,
    )
