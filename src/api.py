"""Simple FastAPI application for HTML Vision Parse."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .simple_pipeline import SimplePipeline


app = FastAPI(title="HTML Vision Parse API")

# Lazy pipeline instance
_pipeline = None


def get_pipeline() -> SimplePipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SimplePipeline()
    return _pipeline


class ExtractRequest(BaseModel):
    url: Optional[str] = None
    html_source: Optional[str] = None


class ExtractResponse(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    publish_time: Optional[str] = None
    lang_type: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """
    Extract content from a URL or HTML source.

    Either `url` or `html_source` must be provided.
    """
    if not request.url and not request.html_source:
        raise HTTPException(
            status_code=400,
            detail="Either 'url' or 'html_source' must be provided"
        )

    pipeline = get_pipeline()

    if request.url:
        result = pipeline.extract_from_url(request.url)
    else:
        result = pipeline.extract_from_html(request.html_source)

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    return ExtractResponse(
        title=result.title,
        content=result.content,
        publish_time=result.publish_time,
        lang_type=result.lang_type,
        country=result.country,
        city=result.city,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
