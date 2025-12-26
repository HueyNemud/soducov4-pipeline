#!/usr/bin/env python3
import argparse

# Import all subcommand modules (package-aware import so `-m cli.soduco` works)
from cli.commands import assembly, chunking, extraction, ocr

COMMANDS = {
    "ocr": ocr,
    "chunking": chunking,
    "extraction": extraction,
    "assembly": assembly,
}


def main():
    parser = argparse.ArgumentParser(
        prog="soduco", description="Soduco: Multi-purpose CLI tool"
    )

    # Meta flags
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for all commands"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for all commands"
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Available subcommands",
        dest="command",
        required=True,
    )

    # Dynamically add subcommands
    for cmd_name, cmd_module in COMMANDS.items():
        cmd_module.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
