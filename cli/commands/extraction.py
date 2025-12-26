from __future__ import annotations
import os
from pathlib import Path
import sys
import time
from loguru import logger
from pipeline.core.executor import Pipeline
from pipeline.extraction.stage import Extraction


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "extraction",
        help="Run LLM extraction over chunks.jsonl",
    )
    parser.add_argument("pdf", type=str, help="Path to input PDF file.")
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Path to system prompt file for extraction LLM.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt content for extraction LLM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if cached results exist.",
    )
    parser.add_argument(
        "--mistral-api-key",
        type=str,
        default=None,
        help="API key for Mistral extraction engine.",
    )
    parser.set_defaults(func=run)


def run(args):
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Error: File does not exist: {pdf_path}")

    start = time.perf_counter()

    stage = Extraction()

    pipeline = Pipeline.for_pdf(
        pdf_path,
        stages=[stage],
        debug=args.debug,
        verbose=args.verbose,
        enable_cache=True,
    )
    force_compute = [stage.name] if args.force else []

    mistral_api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY")

    for stage, _ in pipeline.run(
        force_compute=force_compute,
        stage_params={
            stage.name: {
                "engine": "mistral",
                "model": "mistral-large-2512",
                "system_prompt": get_system_prompt(args),
                "mistral_api_key": mistral_api_key,
            },
        },
    ):
        logger.success(f"🎉 Stage '{stage}' completed.")

    end = time.perf_counter()
    total_time = end - start
    logger.success(f"🔚 Pipeline completed in {total_time:.2f} seconds")


def get_system_prompt(args) -> str:
    if getattr(args, "system_prompt_file", None):
        path = Path(args.system_prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"System prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    elif getattr(args, "system_prompt", None):
        return args.system_prompt
    raise ValueError("Either --system-prompt-file or --system-prompt must be provided.")
