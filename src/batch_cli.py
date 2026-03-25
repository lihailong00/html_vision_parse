"""Batch CLI for processing JSONL files with URLs or HTML sources."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
from loguru import logger

from .pipeline import WebPagePipeline
from .html_renderer import HTMLRenderer
from config.settings import settings




class BatchCLI:
    """
    Batch processing CLI for JSONL input/output.

    Input JSONL format:
        {"url": "https://example.com/article1"}
        {"html_source": "<html>...</html>", "url": "optional-identifier"}

    Output JSONL format:
        {"url": "https://example.com/article1", "title": "...", "content": "...", "publish_time": "...", "success": true}

    Usage:
        # Process URLs from input.jsonl
        python -m src.batch_cli --input input.jsonl --output results.jsonl

        # Process HTML sources with custom fields
        python -m src.batch_cli --input html_batch.jsonl --output results.jsonl --fields title,content --methods '{"title": "vl", "content": "ocr"}'
    """

    def __init__(self):
        self._pipeline: Optional[WebPagePipeline] = None
        self._html_renderer: Optional[HTMLRenderer] = None

    def _ensure_pipeline(self):
        """Ensure pipeline is initialized."""
        if self._pipeline is None:
            self._pipeline = WebPagePipeline()
        return self._pipeline

    def _ensure_html_renderer(self):
        """Ensure HTML renderer is initialized."""
        if self._html_renderer is None:
            self._html_renderer = HTMLRenderer()
        return self._html_renderer

    async def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        fields: Optional[List[str]] = None,
        methods: Optional[Dict[str, str]] = None,
        max_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a JSONL file and write results to output file.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            fields: List of fields to extract
            methods: Dict mapping field -> method
            max_items: Maximum number of items to process (None = all)

        Returns:
            Statistics dict with processing results
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_file = Path(output_path)
        pipeline = self._ensure_pipeline()
        renderer = self._ensure_html_renderer()

        # Stats tracking
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_time_seconds": 0,
        }

        start_time = time.time()

        logger.info("processing_jsonl", input=input_path, output=output_path, max_items=max_items)

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line_num, line in enumerate(fin):
                if max_items and line_num >= max_items:
                    logger.info("max_items_reached", count=max_items)
                    break

                stats["total"] += 1

                try:
                    # Parse input line
                    record = json.loads(line.strip())

                    # Determine if URL or HTML source
                    if "url" in record and "html_source" not in record:
                        # URL-based extraction
                        url = record["url"]
                        identifier = record.get("identifier", url)

                        result = await pipeline.extract_from_url(
                            url=url,
                            fields=fields,
                            methods=methods,
                        )

                        output_record = {
                            "url": url,
                            "identifier": identifier,
                            "title": result.title,
                            "content": result.content,
                            "publish_time": result.publish_time,
                            "confidence": result.confidence,
                            "extraction_method": result.extraction_method,
                            "success": True,
                        }

                    elif "html_source" in record:
                        # HTML source extraction
                        html_content = record["html_source"]
                        identifier = record.get("identifier", record.get("url", f"line_{line_num}"))
                        url = record.get("url")

                        # Render HTML to image
                        image = await renderer.render_from_html(html_content)

                        # Extract using flexible extractor
                        pipeline._ensure_model_loaded()

                        if fields or methods:
                            result = pipeline._flexible_extractor.extract_fields(
                                image, fields=fields, methods=methods
                            )
                        else:
                            result = pipeline._extractor.extract(image)
                            result.extraction_method = "vl"

                        output_record = {
                            "url": url,
                            "identifier": identifier,
                            "title": result.title,
                            "content": result.content,
                            "publish_time": result.publish_time,
                            "confidence": result.confidence,
                            "extraction_method": result.extraction_method,
                            "success": True,
                        }

                    else:
                        # No url or html_source - skip
                        logger.warning("skipping_record_no_source", line=line_num + 1)
                        output_record = {
                            "line": line_num + 1,
                            "success": False,
                            "error": "No 'url' or 'html_source' field found",
                        }
                        stats["skipped"] += 1

                    fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

                    if output_record.get("success"):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

                    # Progress logging every 100 items
                    if stats["total"] % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = stats["total"] / elapsed if elapsed > 0 else 0
                        logger.info(
                            "batch_progress",
                            processed=stats["total"],
                            success=stats["success"],
                            failed=stats["failed"],
                            rate=f"{rate:.1f} items/sec",
                        )

                except json.JSONDecodeError as e:
                    logger.error("json_parse_failed", line=line_num + 1, error=str(e))
                    output_record = {
                        "line": line_num + 1,
                        "success": False,
                        "error": f"JSON parse error: {str(e)}",
                    }
                    fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    stats["failed"] += 1

                except Exception as e:
                    logger.error("processing_failed", line=line_num + 1, error=str(e))
                    output_record = {
                        "line": line_num + 1,
                        "success": False,
                        "error": str(e),
                    }
                    fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    stats["failed"] += 1

        stats["total_time_seconds"] = time.time() - start_time

        return stats

    def process_jsonl_sync(
        self,
        input_path: str,
        output_path: str,
        fields: Optional[List[str]] = None,
        methods: Optional[Dict[str, str]] = None,
        max_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for process_jsonl."""
        return asyncio.run(self.process_jsonl(
            input_path=input_path,
            output_path=output_path,
            fields=fields,
            methods=methods,
            max_items=max_items,
        ))

    def __del__(self):
        """Cleanup on deletion."""
        if self._pipeline is not None:
            self._pipeline._release_model()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process URLs or HTML sources for content extraction"
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file path"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file path"
    )

    parser.add_argument(
        "--fields", "-f",
        help="Comma-separated fields to extract (default: all)"
    )

    parser.add_argument(
        "--methods", "-m",
        help="JSON dict for per-field methods, e.g. '{\"title\": \"vl\", \"content\": \"ocr\"}'"
    )

    parser.add_argument(
        "--max-items", "-n",
        type=int,
        default=None,
        help="Maximum number of items to process"
    )

    return parser.parse_args()


def main():
    """Main entry point for batch CLI."""
    args = parse_args()

    # Parse fields
    fields = None
    if args.fields:
        fields = [f.strip() for f in args.fields.split(',')]

    # Parse methods
    methods = None
    if args.methods:
        try:
            methods = json.loads(args.methods)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid methods JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Run batch processing
    cli = BatchCLI()

    try:
        stats = cli.process_jsonl(
            input_path=args.input,
            output_path=args.output,
            fields=fields,
            methods=methods,
            max_items=args.max_items,
        )

        print("\n" + "=" * 50)
        print("Batch Processing Complete")
        print("=" * 50)
        print(f"Total:      {stats['total']}")
        print(f"Success:    {stats['success']}")
        print(f"Failed:     {stats['failed']}")
        print(f"Skipped:    {stats['skipped']}")
        print(f"Time:       {stats['total_time_seconds']:.1f}s")
        if stats['total_time_seconds'] > 0:
            rate = stats['total'] / stats['total_time_seconds']
            print(f"Rate:       {rate:.1f} items/sec")
        print(f"Output:     {args.output}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
