#!/usr/bin/env python3
"""
Web Screenshot Parser - CLI Entry Point

Extract structured content (title, content, publish_time) from web screenshots
using Qwen3-VL model.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.batch_processor import BatchProcessor
from config.settings import settings

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)




def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from src.api import app

    logger.info("starting_server", host=args.host, port=args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
    )


def cmd_process(args):
    """Process a directory of screenshots."""
    processor = BatchProcessor()

    output_path = args.output or f"results_{Path(args.directory).name}.json"

    results = processor.process_directory(
        directory=args.directory,
        output_path=output_path,
    )

    stats = processor.get_stats()
    logger.info(
        "processing_complete",
        results=len(results),
        cached=stats["cached"],
        failed=stats["failed"],
        output=output_path,
    )

    # Print summary
    print(f"\n✅ Processing complete!")
    print(f"   Results: {len(results)}")
    print(f"   Cached: {stats['cached']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Output: {output_path}")

    # List high/low confidence results
    high_conf = [r for r in results if r.is_high_confidence]
    low_conf = [r for r in results if not r.is_high_confidence]

    print(f"\n   High confidence: {len(high_conf)} ({len(high_conf)/len(results)*100:.1f}%)")
    print(f"   Low confidence: {len(low_conf)} ({len(low_conf)/len(results)*100:.1f}%)")

    if low_conf:
        print(f"\n⚠️  {len(low_conf)} results need review (confidence < {settings.extraction.min_confidence})")


def cmd_single(args):
    """Process a single screenshot."""
    from PIL import Image
    from src.model_loader import ModelLoader
    from src.inference import InferenceEngine
    from src.extractor import ContentExtractor

    logger.info("processing_single", path=args.image)

    # Load model
    loader = ModelLoader()
    loader.load()
    engine = InferenceEngine(loader)
    extractor = ContentExtractor(engine)

    try:
        image = Image.open(args.image).convert("RGB")
        result = extractor.extract(image)

        print("\n" + "=" * 60)
        print("EXTRACTION RESULT")
        print("=" * 60)

        if result.parse_error:
            print(f"❌ Error: {result.parse_error}")
            print(f"\nRaw response:\n{result.raw_response}")
        else:
            print(f"📌 Title: {result.title or 'N/A'}")
            print(f"📅 Publish Time: {result.publish_time or 'N/A'}")
            print(f"⭐ Confidence: {result.confidence:.2%}")
            print(f"🚫 Ignored Regions: {', '.join(result.regions_ignored) or 'None'}")
            print(f"\n📄 Content Preview:")
            print("-" * 40)
            content = result.content or "N/A"
            print(content[:500] + "..." if len(content) > 500 else content)

            if result.is_high_confidence:
                print("\n✅ High confidence result")
            else:
                print(f"\n⚠️  Low confidence (threshold: {settings.extraction.min_confidence})")

        print("=" * 60)

    finally:
        loader.unload()


def cmd_info(args):
    """Show model and configuration info."""
    print("Web Screenshot Parser Configuration")
    print("=" * 50)
    print(f"\nModel: {settings.model.name}")
    print(f"Device: {settings.model.device}")
    print(f"Quantization: {settings.model.quantization}")
    print(f"Max Batch Size: {settings.model.max_batch_size}")
    print(f"\nConfidence Threshold: {settings.extraction.min_confidence}")
    print(f"Cross Validation: {settings.extraction.enable_cross_validation}")
    print(f"Max Retries: {settings.extraction.max_retries}")
    print(f"\nBrowser: {settings.browser.type} (headless={settings.browser.headless})")
    print(f"Viewport: {settings.browser.viewport_width}x{settings.browser.viewport_height}")
    print(f"\nAPI Server: {settings.api.host}:{settings.api.port}")


def cmd_scrape(args):
    """Scrape a URL: open in browser, screenshot, and extract."""
    from src.pipeline import WebPagePipeline

    method = getattr(args, 'method', 'hybrid')
    logger.info("scraping_url", url=args.url, screenshot_only=args.screenshot_only, method=method)

    # Configure pipeline based on method
    if method == "ocr":
        pipeline = WebPagePipeline(use_hybrid=False)
        # For pure OCR, we need a different approach since OCR extractor
        # is only initialized when use_hybrid=True
        use_ocr_only = True
    else:
        pipeline = WebPagePipeline(use_hybrid=(method == "hybrid"))
        use_ocr_only = False

    if args.screenshot_only:
        # Only capture screenshot
        result = pipeline.run_sync(args.url, screenshot_only=True)
        print("\n" + "=" * 60)
        print("SCREENSHOT CAPTURED")
        print("=" * 60)
        print(f"URL: {args.url}")
        print(f"Width: {result.get('width')}px")
        print(f"Height: {result.get('height')}px")
        if result.get('saved_to'):
            print(f"Saved to: {result.get('saved_to')}")
        print("=" * 60)
    else:
        # Full extraction
        import asyncio

        if use_ocr_only:
            # Pure OCR extraction (no VL model needed)
            from src.ocr_extractor import OCRExtractor

            # Capture screenshot first
            image = asyncio.run(pipeline._screenshot.capture(args.url))

            # Use OCR directly
            ocr = OCRExtractor()
            ocr_result = ocr.extract_ocr(image)

            # Build result
            from src.extractor import ExtractionResult
            result = ExtractionResult(
                title=ocr.extract_title(ocr_result.blocks, image.height),
                content=ocr_result.full_text,
                publish_time=ocr.extract_time(ocr_result.blocks),
                confidence=ocr_result.confidence,
                extraction_method="ocr",
                source_url=args.url,
            )
        else:
            # VL or Hybrid extraction
            result = asyncio.run(pipeline.extract_from_url(args.url))

        print("\n" + "=" * 60)
        print(f"EXTRACTION RESULT [{result.extraction_method or 'unknown'} method]")
        print("=" * 60)
        print(f"URL: {args.url}")
        print(f"Method: {result.extraction_method or 'N/A'}")

        if result.parse_error:
            print(f"❌ Error: {result.parse_error}")
            print(f"\nRaw response:\n{result.raw_response}")
        else:
            print(f"📌 Title: {result.title or 'N/A'}")
            print(f"📅 Publish Time: {result.publish_time or 'N/A'}")
            print(f"⭐ Confidence: {result.confidence:.2%}")
            if hasattr(result, 'regions_ignored') and result.regions_ignored:
                print(f"🚫 Ignored Regions: {', '.join(result.regions_ignored)}")
            print(f"\n📄 Content Preview:")
            print("-" * 40)
            content = result.content or "N/A"
            if args.full_content:
                print(content)
                print(f"\n[Full content length: {len(content)} chars]")
            else:
                preview_len = 2000  # Show more in preview
                if len(content) > preview_len:
                    print(content[:preview_len] + "...")
                    print(f"\n[Content truncated. Use --full-content for complete content. Length: {len(content)} chars]")
                else:
                    print(content)
                    print(f"\n[Complete content. Length: {len(content)} chars]")

            if result.is_high_confidence:
                print("\n✅ High confidence result")
            else:
                print(f"\n⚠️  Low confidence (threshold: {settings.extraction.min_confidence})")

        print("=" * 60)


def cmd_scrape_batch(args):
    """Scrape multiple URLs from a file."""
    from src.pipeline import BatchWebPipeline

    # Read URLs from file
    urls = []
    with open(args.url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    logger.info("batch_scrape", count=len(urls), output=args.output)

    pipeline = BatchWebPipeline()
    results = pipeline.process_urls_sync(urls)

    # Save results
    import json
    output_data = []
    for i, r in enumerate(results):
        item = {"url": urls[i]}
        if hasattr(r, 'to_dict'):
            item.update(r.to_dict())
        else:
            item["error"] = str(r)
        output_data.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Summary
    success = sum(1 for r in results if hasattr(r, 'parse_error') and r.parse_error is None)
    print(f"\n✅ Batch scraping complete!")
    print(f"   Total: {len(urls)}")
    print(f"   Successful: {success}")
    print(f"   Failed: {len(urls) - success}")
    print(f"   Output: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Web Screenshot Parser using Qwen3-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.set_defaults(func=cmd_serve)

    # process command
    process_parser = subparsers.add_parser("process", help="Process a directory of screenshots")
    process_parser.add_argument("directory", help="Directory containing screenshots")
    process_parser.add_argument("--output", "-o", help="Output JSON file path")
    process_parser.set_defaults(func=cmd_process)

    # single command
    single_parser = subparsers.add_parser("single", help="Process a single screenshot")
    single_parser.add_argument("image", help="Path to screenshot image")
    single_parser.set_defaults(func=cmd_single)

    # info command
    info_parser = subparsers.add_parser("info", help="Show configuration info")
    info_parser.set_defaults(func=cmd_info)

    # scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape a URL: open in browser, screenshot, and extract")
    scrape_parser.add_argument("url", help="Target URL to scrape")
    scrape_parser.add_argument("--screenshot-only", action="store_true", help="Only capture screenshot, skip extraction")
    scrape_parser.add_argument("--full-content", action="store_true", help="Print full content instead of truncated preview")
    scrape_parser.add_argument("--method", choices=["ocr", "vl", "hybrid"], default="hybrid",
                              help="Extraction method: ocr (fast), vl (accurate), hybrid (auto)")
    scrape_parser.set_defaults(func=cmd_scrape)

    # scrape-batch command
    scrape_batch_parser = subparsers.add_parser("scrape-batch", help="Scrape multiple URLs from a file")
    scrape_batch_parser.add_argument("url_file", help="File containing URLs (one per line)")
    scrape_batch_parser.add_argument("--output", "-o", default="batch_results.json", help="Output JSON file")
    scrape_batch_parser.set_defaults(func=cmd_scrape_batch)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
