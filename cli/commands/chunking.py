import argparse
from pathlib import Path
import sys
import time
from pipeline.chunking.stage import Chunking, ChunkingParameters
from pipeline.core.executor import Pipeline
from pipeline.logging import logger


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "chunking",
        help="Build chunks (chunks.jsonl) from OCR document.jsonl",
    )
    parser.add_argument("pdf", type=str, help="Path to input PDF file.")

    def _spurious_threshold(value: str) -> float:
        try:
            fvalue = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid spuriousness threshold: {value}")
        if not (0.0 <= fvalue <= 1.0):
            raise argparse.ArgumentTypeError(
                f"Spuriousness threshold must be between 0.0 and 1.0: {value}"
            )
        return fvalue

    parser.add_argument(
        "--max-spuriousness",
        type=_spurious_threshold,
        default=1.0,
        help="Discard OCR lines with spurious score > threshold (0..1). Default: 1.0 (keep all).",
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

    stage = Chunking()

    pipeline = Pipeline.for_pdf(
        pdf_path,
        stages=[stage],
        debug=args.debug,
        verbose=args.verbose,
        enable_cache=True,
    )
    force_compute = [stage.name] if args.force else []
    for stage, _ in pipeline.run(
        force_compute=force_compute,
        stage_params={
            stage.name: ChunkingParameters(spuriousness_threshold=args.max_spuriousness)
        },
    ):
        logger.success(f"🎉 Stage '{stage}' completed.")

    end = time.perf_counter()
    total_time = end - start
    logger.success(f"🔚 Pipeline completed in {total_time:.2f} seconds")
