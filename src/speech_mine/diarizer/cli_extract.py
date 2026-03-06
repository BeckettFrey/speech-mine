"""
Command-line interface for speech extraction.

This module provides the CLI for the speech diarization and transcription tool.
"""

import argparse
import logging
import os
import sys

from .processor import SpeechDiarizationProcessor


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_extract_parser() -> argparse.ArgumentParser:
    """Create argument parser for extract command."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text with Speaker Diarization and Word-Level Timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  extract meeting.wav output.csv --hf-token your_token_here
  extract interview.wav result.csv --model large-v3 --device cuda --num-speakers 2
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input audio file (.wav, .mp3, .ogg, .flac, .m4a, .webm)"
    )
    parser.add_argument(
        "output_file", 
        help="Output .csv file for results"
    )
    parser.add_argument(
        "--hf-token", 
        required=True,
        help="HuggingFace access token (required for pyannote models)"
    )
    parser.add_argument(
        "--model", 
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v2", "turbo"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--device", 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--compute-type", 
        default="float16",
        choices=["float16", "int8", "float32"],
        help="Compute type for Whisper (default: float16)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Number of speakers (if known). Improves accuracy when specified."
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum number of speakers (default: 1)"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers. If not specified, no maximum limit."
    )
    
    return parser


def extract_command(args: argparse.Namespace) -> int:
    """
    Execute the extract command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    setup_logging(args.verbose)
    
    # Validate arguments
    from .processor import SpeechDiarizationProcessor
    supported = SpeechDiarizationProcessor.SUPPORTED_FORMATS
    if not args.input_file.lower().endswith(supported):
        print(f"❌ Error: Unsupported format. Supported: {supported}")
        return 1
    
    if not args.output_file.lower().endswith('.csv'):
        print("❌ Error: Output file must be a .csv file")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        print("🎤 Script Extractor Initialized")
        
        # Initialize processor
        processor = SpeechDiarizationProcessor(
            whisper_model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        
        # Process audio file
        processor.process_audio_file(args.input_file, args.output_file)
        
        print("\n✅ Processing completed successfully!")
        print(f"📄 Results saved to: {args.output_file}")
        print(f"📋 Metadata saved to: {args.output_file.replace('.csv', '_metadata.json')}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def main() -> int:
    """Main entry point for the extract CLI."""
    parser = create_extract_parser()
    args = parser.parse_args()
    return extract_command(args)


if __name__ == "__main__":
    sys.exit(main())
