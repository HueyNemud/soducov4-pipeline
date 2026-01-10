"""Shared utilities for CLI execution logic."""

import sys
import time
from pathlib import Path
from loguru import logger
from pipeline.core.executor import Pipeline


def execute_pipeline_stage(args, stage, stage_params: dict):
    """Orchestrates the common execution flow for any pipeline stage."""
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        sys.exit(1)

    start_time = time.perf_counter()

    pipeline = Pipeline.for_pdf(
        pdf_path,
        stages=[stage],
        debug=getattr(args, "debug", False),
        verbose=getattr(args, "verbose", False),
        enable_cache=True,
    )

    force_list = [stage.name] if args.force else []

    try:
        for stage_name, _ in pipeline.run(
            force_compute=force_list,
            stage_params=stage_params,
        ):
            logger.success(f"Stage '{stage_name}' completed.")
    except Exception as e:
        logger.exception(f"Pipeline failure in {stage.name}: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Finished {stage.name} in {elapsed:.2f}s.")
