"""

Command-line interface for CLI tools.



This module provides CLI tools for searching transcripts using fuzzy matching
and accessing transcript data with various output formats.

"""



import argparse
import json
import os
import sys
from typing import Tuple, Dict, Any, List

from .access import TranscriptionAccessTool
from .fuzz import speech_fuzzy_match


def create_search_parser() -> argparse.ArgumentParser:
    """Create argument parser for search command."""
    parser = argparse.ArgumentParser(
        description="Search transcripts using fuzzy matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  speech-mine search "hello world" data.csv --similarity-range 0.7 1.0
  speech-mine search "meeting" data.csv data_metadata.json --top-k 5 --output-type utterance
  speech-mine search "presentation" data.csv --pretty
  speech-mine search "presentation" data.csv --save-path results.json
        """
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query string (word, phrase, or sentence)"
    )
    parser.add_argument(
        "path_to_csv",
        type=str,
        help="Path to CSV file containing transcript data"
    )
    parser.add_argument(
        "path_to_json",
        type=str,
        nargs='?',
        help="Path to JSON metadata file (optional)"
    )
    parser.add_argument(
        "--similarity-range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=('MIN', 'MAX'),
        help="Similarity range as two floats (default: 0.0 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)"
    )
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["utterance", "timestamp"],
        default="utterance",
        help="Output format: 'utterance' for utterance indices, 'timestamp' for time windows (default: utterance)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save JSON output file (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Display results in formatted, human-readable output"
    )

    return parser


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.sss"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def display_pretty_results(output_data: Dict[str, Any]) -> None:
    """Display search results in formatted, colored output."""
    CYAN = '\033[1;36m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[1;31m'
    BOLD = '\033[1;37m'
    DIM = '\033[0;90m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'

    results = output_data.get('results', [])
    output_type = output_data.get('search_parameters', {}).get('output_type', 'utterance')

    print(f"\n{CYAN}{'═' * 67}{NC}")
    print(f"{CYAN}                        SEARCH RESULTS{NC}")
    print(f"{CYAN}{'═' * 67}{NC}\n")

    if not results:
        print(f"{YELLOW}No matches found.{NC}")
        return

    for i, result in enumerate(results, 1):
        print(f"{CYAN}{'─' * 67}{NC}")
        print(f"{CYAN}Result #{i}{NC}")
        print(f"{CYAN}{'─' * 67}{NC}")

        score = result.get('similarity_score', 0)
        score_color = GREEN if score >= 0.8 else YELLOW if score >= 0.6 else RED
        print(f"  {BOLD}Similarity:{NC} {score_color}{score:.3f}{NC}")

        matched = result.get('matched_text', '')
        print(f"  {BOLD}Match:{NC} \"{matched}\"")

        if output_type == 'timestamp':
            tw = result.get('time_window', {})
            if tw:
                print(f"  {BOLD}Time:{NC} {MAGENTA}{_fmt_time(tw['start_time'])}{NC} → {MAGENTA}{_fmt_time(tw['end_time'])}{NC} ({tw['duration']:.3f}s)")
        else:
            ts = result.get('time_span', {})
            if ts:
                print(f"  {BOLD}Time:{NC} {MAGENTA}{_fmt_time(ts['start'])}{NC} → {MAGENTA}{_fmt_time(ts['end'])}{NC} ({ts['duration']:.3f}s)")

        ctx = result.get('context', {})
        if ctx.get('speaker'):
            print(f"  {BOLD}Speaker:{NC} {ctx['speaker']}")
        if ctx.get('full_segment_text') and ctx['full_segment_text'] != matched:
            text = ctx['full_segment_text']
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"  {BOLD}Context:{NC} {DIM}\"{text}\"{NC}")

        if output_type == 'utterance' and result.get('utterance_number') is not None:
            print(f"  {BOLD}Utterance #{NC}: {result['utterance_number']}")

        print()

    print(f"{CYAN}{'═' * 67}{NC}")
    n = len(results)
    print(f"{CYAN}Found {GREEN}{n}{CYAN} result{'s' if n != 1 else ''}{NC}")
    print(f"{CYAN}{'═' * 67}{NC}")


def format_utterance_results(tool: TranscriptionAccessTool, matches: List[Tuple[int, int, float]]) -> List[Dict[str, Any]]:
    """
    Format fuzzy match results with utterance information.

    Args:
        tool: TranscriptionAccessTool instance
        matches: List of (start_idx, end_idx, similarity) tuples

    Returns:
        List of formatted result dictionaries
    """
    results = []

    for start_idx, end_idx, similarity in matches:
        if start_idx < len(tool.words) and end_idx < len(tool.words):
            matched_words = tool.words[start_idx:end_idx + 1]

            if matched_words:
                utterance_num = matched_words[0].utterance_number
                utterance_words = tool.words_by_utterance.get(utterance_num, [])

                utterance_start_idx = None
                utterance_end_idx = None

                for i, word in enumerate(utterance_words):
                    if (utterance_start_idx is None and
                        word.word == matched_words[0].word and
                        abs(word.start - matched_words[0].start) < 0.1):
                        utterance_start_idx = i

                    if (utterance_start_idx is not None and
                        word.word == matched_words[-1].word and
                        abs(word.start - matched_words[-1].start) < 0.1):
                        utterance_end_idx = i
                        break

                if utterance_start_idx is not None and utterance_end_idx is not None:
                    word_range = tool.get_word_range(utterance_num, utterance_start_idx, utterance_end_idx)

                    if word_range:
                        result = {
                            "match_indices": {
                                "global_start_index": start_idx,
                                "global_end_index": end_idx,
                                "utterance_start_index": utterance_start_idx,
                                "utterance_end_index": utterance_end_idx
                            },
                            "similarity_score": round(similarity, 4),
                            "utterance_number": word_range["utterance_number"],
                            "matched_words": [word["word"] for word in word_range["words"]],
                            "matched_text": " ".join([word["word"] for word in word_range["words"]]),
                            "time_span": {
                                "start": word_range["time_span"]["start"],
                                "end": word_range["time_span"]["end"],
                                "duration": round(word_range["time_span"]["duration"], 3)
                            },
                            "context": {
                                "full_segment_text": word_range["segment_data"]["text"],
                                "speaker": word_range["segment_data"]["speaker"],
                                "segment_confidence": word_range["segment_data"]["confidence"]
                            }
                        }
                        results.append(result)

    return results


def format_timestamp_results(tool: TranscriptionAccessTool, matches: List[Tuple[int, int, float]]) -> List[Dict[str, Any]]:
    """
    Format fuzzy match results with timestamp window information.

    Args:
        tool: TranscriptionAccessTool instance
        matches: List of (start_idx, end_idx, similarity) tuples

    Returns:
        List of formatted result dictionaries
    """
    results = []

    for start_idx, end_idx, similarity in matches:
        if start_idx < len(tool.words) and end_idx < len(tool.words):
            matched_words = tool.words[start_idx:end_idx + 1]

            if matched_words:
                result = {
                    "similarity_score": round(similarity, 4),
                    "time_window": {
                        "start_time": matched_words[0].start,
                        "end_time": matched_words[-1].end,
                        "duration": round(matched_words[-1].end - matched_words[0].start, 3)
                    },
                    "matched_words": [word.word for word in matched_words],
                    "matched_text": " ".join([word.word for word in matched_words]),
                    "word_details": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.confidence
                        }
                        for word in matched_words
                    ],
                    "context": {
                        "full_segment_text": matched_words[0].text,
                        "speaker": matched_words[0].speaker,
                        "utterance_number": matched_words[0].utterance_number
                    }
                }
                results.append(result)

    return results


def search_command(args: argparse.Namespace) -> int:
    """
    Execute the search command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        if not os.path.exists(args.path_to_csv):
            print(f"❌ Error: CSV file not found: {args.path_to_csv}")
            return 1

        if args.path_to_json and not os.path.exists(args.path_to_json):
            print(f"❌ Error: JSON file not found: {args.path_to_json}")
            return 1

        min_sim, max_sim = args.similarity_range
        if not (0.0 <= min_sim <= max_sim <= 1.0):
            print("❌ Error: Similarity range must be between 0.0 and 1.0, with min <= max")
            return 1

        print(f"🔍 Loading transcript data from: {args.path_to_csv}")

        tool = TranscriptionAccessTool()
        tool.load_from_files(args.path_to_csv, args.path_to_json)

        print(f"📊 Loaded {len(tool.words)} words from {len(tool.segments)} segments")
        print(f"🔎 Searching for: '{args.query}'")

        matches = speech_fuzzy_match(
            word_list=tool.words,
            query=args.query,
            similarity_range=(min_sim, max_sim),
            top_k=args.top_k
        )

        if not matches:
            print("❌ No matches found within the specified similarity range.")
            return 0

        print(f"✅ Found {len(matches)} matches")

        if args.output_type == "utterance":
            formatted_results = format_utterance_results(tool, matches)
        else:
            formatted_results = format_timestamp_results(tool, matches)

        output_data = {
            "query": args.query,
            "search_parameters": {
                "similarity_range": {"min": min_sim, "max": max_sim},
                "top_k": args.top_k,
                "output_type": args.output_type
            },
            "transcript_info": tool.get_stats(),
            "results": formatted_results,
            "total_matches": len(matches)
        }

        if args.pretty:
            display_pretty_results(output_data)
        elif args.save_path:
            os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
            with open(args.save_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {args.save_path}")
        else:
            print("\n" + "="*60)
            print("SEARCH RESULTS")
            print("="*60)
            print(json.dumps(output_data, indent=2, ensure_ascii=False))

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="speech-mine-search",
        description="Speech Mining Tools - Search and analyze speech transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="{search}"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search transcripts using fuzzy matching",
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
