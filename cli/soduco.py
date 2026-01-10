#!/usr/bin/env python3
import argparse
from cli.commands import assembly, chunking, extraction, ocr

COMMANDS = {
    "ocr": ocr,
    "chunking": chunking,
    "extraction": extraction,
    "assembly": assembly,
}


def main():
    # 1. Create a Parent Parser for shared arguments
    # These are inherited by all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("pdf", type=str, help="Path to the source PDF file.")
    parent_parser.add_argument(
        "--force", action="store_true", help="Bypass cache and force re-processing."
    )

    # 2. Main Parser
    parser = argparse.ArgumentParser(
        prog="soduco", description="Soduco: Document Intelligence Pipeline"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    subparsers = parser.add_subparsers(
        title="subcommands", dest="command", required=True
    )

    # 3. Register Subcommands using the parent_parser
    for cmd_module in COMMANDS.values():
        # Each module's add_subparser now takes the parents list
        cmd_module.add_subparser(subparsers, parents=[parent_parser])

    args = parser.parse_args()

    # Execution
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        # Global catch-all for unhandled CLI errors
        from loguru import logger

        logger.error(f"Execution failed: {e}")


if __name__ == "__main__":
    main()
