"""Batch CLI for processing JSONL files with URLs or HTML sources."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Annotated
import typer

from src.pipeline import WebPagePipeline, get_available_gpus
from src.html_renderer import HTMLRenderer
from config.settings import settings

app = typer.Typer(help="Batch process URLs or HTML sources for content extraction")


@app.command()
def main(
    input: Annotated[str, typer.Argument(help="Input JSONL file path")],
    output: Annotated[str, typer.Argument(help="Output JSONL file path")],
    fields: Annotated[str, typer.Option("-f", "--fields", help="Comma-separated fields to extract")] = None,
    methods: Annotated[str, typer.Option("-m", "--methods", help="JSON dict for per-field methods")] = None,
    max_items: Annotated[int, typer.Option("-n", "--max-items", help="Maximum number of items to process")] = None,
    gpu_id: Annotated[int, typer.Option("-g", "--gpu-id", help="GPU device ID to use")] = None,
    method: Annotated[str, typer.Option(help="Extraction method: vl or ocr")] = None,
):
    """
    Batch process URLs or HTML sources from JSONL file.

    Input JSONL format:
        {"url": "https://example.com/article1"}
        {"html_source": "<html>...</html>", "url": "optional-identifier"}

    Examples:
        python -m src.batch_cli input.jsonl results.jsonl
        python -m src.batch_cli input.jsonl out.jsonl --gpu-id 1 --method ocr
        python -m src.batch_cli input.jsonl out.jsonl --fields title,content
    """

    # Parse fields
    field_list: Optional[List[str]] = None
    if fields:
        field_list = [f.strip() for f in fields.split(',')]

    # Parse methods
    method_dict: Optional[Dict[str, str]] = None
    if methods:
        try:
            method_dict = json.loads(methods)
        except json.JSONDecodeError as e:
            typer.echo(f"Error: Invalid methods JSON: {e}", err=True)
            raise typer.Exit(1)

    # Validate method
    if method and method not in ["vl", "ocr"]:
        typer.echo(f"Error: method must be 'vl' or 'ocr', got '{method}'", err=True)
        raise typer.Exit(1)

    # Show GPU info
    available_gpus = get_available_gpus()
    typer.echo(f"Available GPUs: {available_gpus}")

    # Show config
    extraction_method = method or settings.extraction.extraction_method
    typer.echo(f"Extraction method: {extraction_method}")
    typer.echo(f"GPU ID: {gpu_id if gpu_id is not None else 'auto'}")

    # Run batch processing
    cli = BatchCLI(gpu_id=gpu_id, extraction_method=extraction_method)

    try:
        stats = asyncio.run(cli.process_jsonl(
            input_path=input,
            output_path=output,
            fields=field_list,
            methods=method_dict,
            max_items=max_items,
        ))

        typer.echo("\n" + "=" * 50)
        typer.echo("Batch Processing Complete")
        typer.echo("=" * 50)
        typer.echo(f"Total:          {stats['total']}")
        typer.echo(f"Success:        {stats['success']}")
        typer.echo(f"Failed:         {stats['failed']}")
        typer.echo(f"Skipped:        {stats['skipped']}")
        typer.echo(f"Time:           {stats['total_time_seconds']:.1f}s")
        typer.echo(f"GPU ID:         {stats['gpu_id']}")
        typer.echo(f"Method:         {stats['extraction_method']}")
        if stats['total_time_seconds'] > 0:
            rate = stats['total'] / stats['total_time_seconds']
            typer.echo(f"Rate:           {rate:.1f} items/sec")
        typer.echo(f"Output:         {output}")
        typer.echo("=" * 50)

    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user", err=True)
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


class BatchCLI:
    """
    Batch processing CLI for JSONL input/output.

    Input JSONL format:
        {"url": "https://example.com/article1"}
        {"html_source": "<html>...</html>", "url": "optional-identifier"}

    Output JSONL format:
        {"url": "...", "title": "...", "content": "...", "success": true}
    """

    def __init__(
        self,
        gpu_id: int = None,
        extraction_method: str = None,
    ):
        self._gpu_id = gpu_id
        self._extraction_method = extraction_method or settings.extraction.extraction_method
        self._pipeline: Optional[WebPagePipeline] = None
        self._html_renderer: Optional[HTMLRenderer] = None

    def _ensure_pipeline(self):
        """Ensure pipeline is initialized."""
        if self._pipeline is None:
            self._pipeline = WebPagePipeline(
                gpu_id=self._gpu_id,
                extraction_method=self._extraction_method,
            )
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
        """Process a JSONL file and write results to output file."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_file = Path(output_path)
        pipeline = self._ensure_pipeline()
        renderer = self._ensure_html_renderer()

        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_time_seconds": 0,
            "gpu_id": self._gpu_id,
            "extraction_method": self._extraction_method,
        }

        start_time = time.time()

        typer.echo(f"Processing: {input_path} -> {output_path}")

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line_num, line in enumerate(fin):
                if max_items and line_num >= max_items:
                    typer.echo(f"Max items reached: {max_items}")
                    break

                stats["total"] += 1

                try:
                    record = json.loads(line.strip())

                    if "url" in record and "html_source" not in record:
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
                        html_content = record["html_source"]
                        identifier = record.get("identifier", record.get("url", f"line_{line_num}"))
                        url = record.get("url")

                        image = await renderer.render_from_html(html_content)

                        if self._extraction_method == "ocr" and methods is None:
                            result = pipeline._extract_with_ocr(image, url)
                        else:
                            pipeline._ensure_model_loaded()
                            if fields or methods:
                                result = pipeline._flexible_extractor.extract_fields(
                                    image, fields=fields, methods=methods
                                )
                            else:
                                result = pipeline._extractor.extract(image)
                                result.extraction_method = self._extraction_method

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
                        typer.echo(f"Warning: Skipping line {line_num + 1} - no url or html_source")
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

                    if stats["total"] % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = stats["total"] / elapsed if elapsed > 0 else 0
                        typer.echo(f"Progress: {stats['total']} processed, {stats['success']} success, {stats['failed']} failed, {rate:.1f} items/sec")

                except json.JSONDecodeError as e:
                    typer.echo(f"Error: JSON parse failed at line {line_num + 1}: {e}")
                    output_record = {
                        "line": line_num + 1,
                        "success": False,
                        "error": f"JSON parse error: {str(e)}",
                    }
                    fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    stats["failed"] += 1

                except Exception as e:
                    typer.echo(f"Error: Processing failed at line {line_num + 1}: {e}")
                    output_record = {
                        "line": line_num + 1,
                        "success": False,
                        "error": str(e),
                    }
                    fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    stats["failed"] += 1

        stats["total_time_seconds"] = time.time() - start_time

        return stats

    def __del__(self):
        """Cleanup on deletion."""
        if self._pipeline is not None:
            self._pipeline._release_model()


if __name__ == "__main__":
    app()
