"""
CLI Subcommand for Document Assembly.

This module provides the command-line interface to fuse OCR geometry,
text chunks, and LLM extractions into a final enriched document (RichStructured).
"""

from __future__ import annotations
from cli.shared import execute_pipeline_stage
from pipeline.assembly.stage import Assembly


def add_subparser(subparsers, parents):
    """Configures the 'assembly' command and its specific arguments."""

    parser = subparsers.add_parser(
        "assembly",
        parents=parents,  # Inherits 'pdf' and '--force'
        help="Fuse chunks and extractions into a spatially-aware rich document.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error immediately if line indices are out of range.",
    )

    parser.set_defaults(func=run)


def run(args):
    stage = Assembly()
    params = {stage.name: {"strict": args.strict}}
    execute_pipeline_stage(args, stage, params)
