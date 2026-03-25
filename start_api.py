#!/usr/bin/env python3
"""
Multi-GPU API server launcher with Typer.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Annotated
import typer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_available_gpus

app = typer.Typer(help="Start API server with multi-GPU support", add_completion=False)


def get_gpu_memory_info(gpu_id: int) -> str:
    """Get GPU memory info."""
    try:
        import torch
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            return f"{mem_total:.1f}GB"
    except Exception:
        pass
    return "unknown"


@app.command()
def main(
    gpus: Annotated[
        Optional[List[int]],
        typer.Option("-g", "--gpus", help="GPU IDs to use. Default: all available GPUs")
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option("-w", "--workers", help="Number of workers. Default: number of GPUs")
    ] = None,
    method: Annotated[
        Optional[str],
        typer.Option(help="Extraction method: vl or ocr")
    ] = None,
    port: Annotated[
        int,
        typer.Option("-p", "--port", help="Starting port (default: 8000)")
    ] = 8000,
    host: Annotated[
        str,
        typer.Option(help="Host to bind (default: 0.0.0.0)")
    ] = "0.0.0.0",
):
    """
    Start API server with multi-GPU support.

    Examples:
        python start_api.py
        python start_api.py --gpus 0 1 2 3
        python start_api.py --gpus 0 1 --workers 2 --method ocr
    """
    # Determine GPUs to use
    if gpus is not None:
        available_gpus = gpus
    else:
        available_gpus = get_available_gpus()

    if not available_gpus:
        typer.echo("No GPUs available! Running on CPU (will be slow)")
        available_gpus = [-1]
    else:
        typer.echo(f"GPUs to use: {available_gpus}")

    # Determine number of workers
    num_workers = workers or len(available_gpus)
    typer.echo(f"Number of workers: {num_workers}")

    # Determine extraction method
    extraction_method = method or "vl"
    typer.echo(f"Extraction method: {extraction_method}")

    # Print GPU info
    for i in range(min(num_workers, len(available_gpus))):
        gpu_id = available_gpus[i % len(available_gpus)]
        if gpu_id >= 0:
            mem = get_gpu_memory_info(gpu_id)
            typer.echo(f"  Worker {i}: GPU {gpu_id} ({mem})")
        else:
            typer.echo(f"  Worker {i}: CPU")

    typer.echo(f"\nStarting {num_workers} worker(s) on ports {port} - {port + num_workers - 1}...")

    processes = []

    try:
        for worker_id in range(num_workers):
            gpu_id = available_gpus[worker_id % len(available_gpus)]
            worker_port = port + worker_id

            # Set environment for this worker
            worker_env = os.environ.copy()
            worker_env["WORKER_ID"] = str(worker_id)

            if gpu_id >= 0:
                worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            else:
                worker_env["CUDA_VISIBLE_DEVICES"] = ""

            # Build command
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", host,
                "--port", str(worker_port),
                "--workers", "1",
            ]

            typer.echo(f"Starting worker {worker_id}: GPU {gpu_id}, port {worker_port}")

            proc = subprocess.Popen(
                cmd,
                env=worker_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker_id, gpu_id, worker_port, proc))

            time.sleep(1)

        typer.echo("\n" + "=" * 60)
        typer.echo("API Server Started")
        typer.echo("=" * 60)
        typer.echo(f"Workers: {num_workers}")
        typer.echo(f"GPUs: {available_gpus[:num_workers]}")
        typer.echo(f"Method: {extraction_method}")
        typer.echo(f"Base URL: http://{host}:{port}")
        typer.echo(f"API Docs: http://{host}:{port}/docs")
        typer.echo("\nPress Ctrl+C to stop all workers")
        typer.echo("=" * 60)

        # Wait for processes
        for worker_id, gpu_id, worker_port, proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        typer.echo("\nStopping all workers...")
        for worker_id, gpu_id, worker_port, proc in processes:
            typer.echo(f"Stopping worker {worker_id} (GPU {gpu_id}, port {worker_port})...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        typer.echo("All workers stopped.")

    except Exception as e:
        typer.echo(f"Error: {e}")
        for worker_id, gpu_id, worker_port, proc in processes:
            proc.terminate()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
