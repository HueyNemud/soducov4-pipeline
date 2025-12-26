from __future__ import annotations
from pathlib import Path
import sys
import time

from loguru import logger

from pipeline.assembly.stage import Assembly
from pipeline.core.executor import Pipeline


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "assembly",
        help="Assemble chunking + extraction (materialize line indices; output items as JSONL)",
    )
    parser.add_argument("pdf", type=str, help="Path to input PDF file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on any mismatch / out-of-range line index.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if cached results exist.",
    )

    parser.set_defaults(func=run)


def run(args):
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Error: File does not exist: {pdf_path}")

    start = time.perf_counter()

    stage = Assembly()

    pipeline = Pipeline.for_pdf(
        pdf_path,
        stages=[stage],
        debug=args.debug,
        verbose=args.verbose,
        enable_cache=True,
    )
    force_compute = [stage.name] if args.force else []
    for stage, _ in pipeline.run(force_compute=force_compute):
        logger.success(f"🎉 Stage '{stage}' completed.")

    end = time.perf_counter()
    total_time = end - start
    logger.success(f"🔚 Pipeline completed in {total_time:.2f} seconds")
