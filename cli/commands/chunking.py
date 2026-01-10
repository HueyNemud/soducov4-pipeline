"""
CLI Subcommand for Document Chunking.

This module provides the command-line interface to trigger the text segmentation
stage, transforming OCR lines into logical, semantic chunks suitable for LLM processing.
"""

from __future__ import annotations
import sys
from cli.shared import execute_pipeline_stage
from pipeline.chunking.stage import Chunking


def add_subparser(subparsers, parents):
    """Configures the 'chunking' command and its specific arguments."""

    parser = subparsers.add_parser(
        "chunking",
        parents=parents,  # Inherits 'pdf' and '--force'
        help="Build logical text chunks from an OCR-processed document.",
    )

    # Segmentation Controls
    parser.add_argument(
        "--max-spuriousness",
        type=float,
        default=1.0,
        help="Spuriousness threshold (0.0 to 1.0). Discard lines above this score. Default: 1.0.",
    )

    parser.set_defaults(func=run)


def run(args):
    # Specific validation stays in the subcommand
    if not (0.0 <= args.max_spuriousness <= 1.0):
        sys.exit("Error: --max-spuriousness must be between 0.0 and 1.0")

    stage = Chunking()
    params = {stage.name: {"engine": {"spuriousness_threshold": args.max_spuriousness}}}
    execute_pipeline_stage(args, stage, params)
