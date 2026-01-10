import sys
from pathlib import Path
from pipeline import OCR
from pipeline import Pipeline
from pipeline.logging import logger
import time


def add_subparser(subparsers, parents):
    """Configures the 'ocr' command and its specific arguments."""

    parser = subparsers.add_parser(
        "ocr",
        parents=parents,  # Inherits 'pdf' and '--force'
        help="Run Surya OCR pipeline on a PDF file.",
    )

    parser.set_defaults(func=run)


def run(args):
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Error: File does not exist: {pdf_path}")

    start = time.perf_counter()

    stage = OCR()

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
