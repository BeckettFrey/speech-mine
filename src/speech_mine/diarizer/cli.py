"""
Main CLI entry point for speech-mine.

This module provides a unified CLI with subcommands for all tools:
  extract  - transcribe audio with speaker diarization
  format   - format CSV transcript into a readable script
  chunk    - split audio into chunks via YAML config
  search   - fuzzy search a transcript CSV
"""

import argparse
import sys
from typing import List

from .cli_extract import extract_command, create_extract_parser
from .cli_format import format_command, create_format_parser
from speech_mine.pickaxe.cli_chunk import chunk_command, create_chunk_parser
from speech_mine.cli import search_command, create_search_parser


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="speech-mine",
        description="Speech-to-Text with Speaker Diarization and Transcript Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  speech-mine extract interview.mp3 output.csv --hf-token YOUR_TOKEN --compute-type float32
  speech-mine format output.csv script.txt
  speech-mine format output.csv script.txt --speakers names.json
  speech-mine chunk recording.wav config.yaml chunks/
  speech-mine search "hello world" output.csv
        """
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="{extract,format,chunk,search}"
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Transcribe audio with speaker diarization",
        parents=[create_extract_parser()],
        add_help=False
    )
    extract_parser.set_defaults(func=extract_command)

    format_parser = subparsers.add_parser(
        "format",
        help="Format CSV transcript into a movie-style script",
        parents=[create_format_parser()],
        add_help=False
    )
    format_parser.set_defaults(func=format_command)

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split a .wav file into chunks via YAML config",
        parents=[create_chunk_parser()],
        add_help=False
    )
    chunk_parser.set_defaults(func=chunk_command)

    search_parser = subparsers.add_parser(
        "search",
        help="Fuzzy search a transcript CSV",
        parents=[create_search_parser()],
        add_help=False
    )
    search_parser.set_defaults(func=search_command)

    return parser


def main(args: List[str] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_main_parser()

    if args is None:
        args = sys.argv[1:]

    if not args:
        parser.print_help()
        return 1

    parsed_args = parser.parse_args(args)

    if not hasattr(parsed_args, 'func'):
        parser.print_help()
        return 1

    return parsed_args.func(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
