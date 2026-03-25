#!/usr/bin/env python3
"""
Multi-GPU API server launcher.

Usage:
    # Single GPU
    python start_api.py

    # Use specific GPUs (2 GPUs)
    python start_api.py --gpus 0 1

    # Use specific GPUs with workers (4 GPUs, 4 workers)
    python start_api.py --gpus 0 1 2 3 --workers 4

    # Use OCR method (faster, no VL model)
    python start_api.py --method ocr

    # Use specific GPU only
    python start_api.py --gpus 2 --method ocr
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_available_gpus


def parse_args():
    parser = argparse.ArgumentParser(description="Start API server with multi-GPU support")

    parser.add_argument(
        "--gpus", "-g",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to use. Default: all available GPUs"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of workers. Default: number of GPUs"
    )

    parser.add_argument(
        "--method",
        choices=["vl", "ocr"],
        default=None,
        help="Extraction method: vl or ocr"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Starting port (default: 8000)"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )

    return parser.parse_args()


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


def main():
    args = parse_args()

    # Determine GPUs to use
    if args.gpus is not None:
        gpus = args.gpus
    else:
        gpus = get_available_gpus()

    if not gpus:
        print("No GPUs available! Running on CPU (will be slow)")
        gpus = [-1]
    else:
        print(f"GPUs to use: {gpus}")

    # Determine number of workers
    num_workers = args.workers or len(gpus)
    print(f"Number of workers: {num_workers}")

    # Determine extraction method
    method = args.method or "vl"
    print(f"Extraction method: {method}")

    # Build environment
    env = os.environ.copy()

    # Print GPU info
    for i, gpu_id in enumerate(gpus[:num_workers]):
        if gpu_id >= 0:
            mem = get_gpu_memory_info(gpu_id)
            print(f"  Worker {i}: GPU {gpu_id} ({mem})")

    print(f"\nStarting {num_workers} worker(s) on ports {args.port} - {args.port + num_workers - 1}...")

    processes = []

    try:
        for worker_id in range(num_workers):
            gpu_id = gpus[worker_id % len(gpus)]
            port = args.port + worker_id

            # Set environment for this worker
            worker_env = env.copy()
            worker_env["WORKER_ID"] = str(worker_id)

            if gpu_id >= 0:
                worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            else:
                worker_env["CUDA_VISIBLE_DEVICES"] = ""

            # Build command
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", args.host,
                "--port", str(port),
                "--workers", "1",
            ]

            print(f"Starting worker {worker_id}: GPU {gpu_id}, port {port}")
            print(f"  Command: {' '.join(cmd)}")

            proc = subprocess.Popen(
                cmd,
                env=worker_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker_id, gpu_id, port, proc))

            # Small delay between starts
            time.sleep(1)

        print("\n" + "=" * 60)
        print("API Server Started")
        print("=" * 60)
        print(f"Workers: {num_workers}")
        print(f"GPUs: {gpus[:num_workers]}")
        print(f"Method: {method}")
        print(f"Base URL: http://{args.host}:{args.port}")
        print(f"API Docs: http://{args.host}:{args.port}/docs")
        print("\nPress Ctrl+C to stop all workers")
        print("=" * 60)

        # Wait for processes
        for worker_id, gpu_id, port, proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nStopping all workers...")
        for worker_id, gpu_id, port, proc in processes:
            print(f"Stopping worker {worker_id} (GPU {gpu_id}, port {port})...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print("All workers stopped.")

    except Exception as e:
        print(f"Error: {e}")
        for worker_id, gpu_id, port, proc in processes:
            proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
