"""
CLI Subcommand for Document Extraction.

This module provides the command-line interface to trigger the LLM-based
information extraction stage, handling parameter injection from files,
environment variables, and CLI arguments.
"""

from __future__ import annotations

import os
from pathlib import Path
from cli.shared import execute_pipeline_stage
from pipeline.extraction.stage import Extraction


def add_subparser(subparsers, parents):
    """Configures the 'extraction' command and its specific arguments."""

    parser = subparsers.add_parser(
        "extraction",
        parents=parents,  # Inherits 'pdf' and '--force'
        help="Run LLM extraction over pre-processed document chunks.",
    )

    # Prompt Configuration
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--system-prompt-file",
        type=str,
        help="Path to a text file containing the LLM system prompt.",
    )
    prompt_group.add_argument(
        "--system-prompt",
        type=str,
        help="Direct string input for the LLM system prompt.",
    )

    parser.add_argument(
        "--mistral-api-key",
        type=str,
        help="Mistral API key (overrides MISTRAL_API_KEY environment variable).",
        default=None,
    )

    parser.set_defaults(func=run)


def run(args):
    api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY")
    stage = Extraction()
    params = {
        stage.name: {
            "provider": "mistral",
            "mistral": {
                "model": "mistral-large-latest",
                "system_prompt": _resolve_system_prompt(args),
                "api_key": api_key,
            },
        }
    }
    execute_pipeline_stage(args, stage, params)


def _resolve_system_prompt(args) -> str:
    """Extracts the system prompt content from either a file or a raw string."""
    if args.system_prompt_file:
        prompt_file = Path(args.system_prompt_file)
        if not prompt_file.exists():
            raise FileNotFoundError(f"System prompt file does not exist: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8").strip()

    return args.system_prompt.strip()
