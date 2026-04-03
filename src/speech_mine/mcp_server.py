"""
MCP server for speech-mine.

Exposes speech-mine capabilities as tools for Claude Code:
  - search_transcript  : fuzzy search a CSV transcript
  - get_transcript_stats : stats about a transcript
  - read_transcript    : export transcript data
  - format_transcript  : convert CSV to readable script
  - extract_audio      : transcribe audio (spawns subprocess)
  - chunk_audio        : split a WAV into timed segments
"""

import json
import subprocess
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .access import TranscriptionAccessTool
from .fuzz import speech_fuzzy_match
from .diarizer.formatter import ScriptFormatter

mcp = FastMCP("speech-mine")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tool(csv_path: str, metadata_path: Optional[str] = None) -> TranscriptionAccessTool:
    tool = TranscriptionAccessTool()
    tool.load_from_files(csv_path, metadata_path)
    return tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_transcript(
    csv_path: str,
    query: str,
    min_similarity: float = 0.0,
    max_similarity: float = 1.0,
    top_k: int = 10,
    output_type: str = "utterance",
    metadata_path: Optional[str] = None,
) -> str:
    """
    Fuzzy-search a speech-mine CSV transcript for a word, phrase, or sentence.

    Args:
        csv_path: Path to the transcript CSV produced by `speech-mine extract`.
        query: Word, phrase, or sentence to search for.
        min_similarity: Minimum similarity score (0.0–1.0). Default 0.0.
        max_similarity: Maximum similarity score (0.0–1.0). Default 1.0.
        top_k: Maximum number of results to return. Default 10.
        output_type: "utterance" (default) or "timestamp".
        metadata_path: Optional path to the companion _metadata.json file.

    Returns:
        JSON string with search results including matched text, timestamps,
        speaker, and similarity scores.
    """
    tool = _load_tool(csv_path, metadata_path)

    matches = speech_fuzzy_match(
        word_list=tool.words,
        query=query,
        similarity_range=(min_similarity, max_similarity),
        top_k=top_k,
    )

    if not matches:
        return json.dumps({"query": query, "total_matches": 0, "results": []}, indent=2)

    if output_type == "timestamp":
        results = []
        for start_idx, end_idx, similarity in matches:
            if start_idx < len(tool.words) and end_idx < len(tool.words):
                words = tool.words[start_idx:end_idx + 1]
                if words:
                    results.append({
                        "similarity_score": round(similarity, 4),
                        "matched_text": " ".join(w.word for w in words),
                        "time_window": {
                            "start_time": words[0].start,
                            "end_time": words[-1].end,
                            "duration": round(words[-1].end - words[0].start, 3),
                        },
                        "context": {
                            "speaker": words[0].speaker,
                            "full_segment_text": words[0].text,
                        },
                    })
    else:
        results = []
        for start_idx, end_idx, similarity in matches:
            if start_idx < len(tool.words) and end_idx < len(tool.words):
                matched_words = tool.words[start_idx:end_idx + 1]
                if matched_words:
                    utt_num = matched_words[0].utterance_number
                    utt_words = tool.words_by_utterance.get(utt_num, [])
                    utt_start = next(
                        (i for i, w in enumerate(utt_words)
                         if w.word == matched_words[0].word
                         and abs(w.start - matched_words[0].start) < 0.1),
                        None,
                    )
                    utt_end = next(
                        (i for i, w in enumerate(utt_words)
                         if w.word == matched_words[-1].word
                         and abs(w.start - matched_words[-1].start) < 0.1),
                        None,
                    ) if utt_start is not None else None

                    word_range = (
                        tool.get_word_range(utt_num, utt_start, utt_end)
                        if utt_start is not None and utt_end is not None
                        else None
                    )

                    result = {
                        "similarity_score": round(similarity, 4),
                        "matched_text": " ".join(w.word for w in matched_words),
                        "utterance_number": utt_num,
                    }
                    if word_range:
                        result["time_span"] = word_range.get("time_span")
                        seg = word_range.get("segment_data") or {}
                        result["context"] = {
                            "speaker": seg.get("speaker"),
                            "full_segment_text": seg.get("text"),
                        }
                    results.append(result)

    output = {
        "query": query,
        "search_parameters": {
            "similarity_range": {"min": min_similarity, "max": max_similarity},
            "top_k": top_k,
            "output_type": output_type,
        },
        "total_matches": len(results),
        "results": results,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


@mcp.tool()
def get_transcript_stats(
    csv_path: str,
    metadata_path: Optional[str] = None,
) -> str:
    """
    Return statistics about a speech-mine CSV transcript.

    Args:
        csv_path: Path to the transcript CSV.
        metadata_path: Optional path to the companion _metadata.json file.

    Returns:
        JSON string with word count, segment count, speaker list,
        average confidence, and duration.
    """
    tool = _load_tool(csv_path, metadata_path)
    return json.dumps(tool.get_stats(), indent=2, ensure_ascii=False)


@mcp.tool()
def read_transcript(
    csv_path: str,
    format_type: str = "utterances",
    metadata_path: Optional[str] = None,
) -> str:
    """
    Export transcript data in various formats.

    Args:
        csv_path: Path to the transcript CSV.
        format_type: One of "utterances" (default), "segments", "words", or "json".
        metadata_path: Optional path to the companion _metadata.json file.

    Returns:
        JSON string with the transcript data in the requested format.
    """
    tool = _load_tool(csv_path, metadata_path)
    data = tool.export(format_type)
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
def format_transcript(
    input_csv: str,
    output_txt: str,
    speakers_json: Optional[str] = None,
) -> str:
    """
    Convert a speech-mine CSV transcript into a human-readable script file.

    Args:
        input_csv: Path to the transcript CSV produced by `speech-mine extract`.
        output_txt: Destination path for the formatted script (.txt).
        speakers_json: Optional JSON file mapping SPEAKER_00 labels to real names.

    Returns:
        The content of the formatted script, or an error message.
    """
    import os

    if not os.path.exists(input_csv):
        return f"Error: CSV file not found: {input_csv}"

    custom_speakers = None
    if speakers_json:
        custom_speakers = ScriptFormatter.load_custom_speakers(speakers_json)

    output_dir = os.path.dirname(output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    formatter = ScriptFormatter(custom_speakers)
    formatter.format_script(input_csv, output_txt)

    with open(output_txt, "r", encoding="utf-8") as f:
        return f.read()


@mcp.tool()
def extract_audio(
    input_file: str,
    output_csv: str,
    hf_token: str,
    model: str = "large-v3",
    device: str = "auto",
    compute_type: str = "float16",
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: Optional[int] = None,
) -> str:
    """
    Transcribe an audio file with speaker diarization and save to CSV.

    This runs `speech-mine extract` in a subprocess — it can take several
    minutes depending on file length and hardware.

    Args:
        input_file: Path to the input audio (.wav, .mp3, .ogg, .flac, .m4a, .webm).
        output_csv: Destination path for the output CSV transcript.
        hf_token: HuggingFace access token (required for pyannote models).
        model: Whisper model size. One of tiny, base, small, medium,
               large-v2, large-v3 (default), turbo.
        device: Device — "auto" (default), "cpu", or "cuda".
        compute_type: "float16" (default), "int8", or "float32".
        num_speakers: Exact speaker count if known (improves accuracy).
        min_speakers: Minimum speakers. Default 1.
        max_speakers: Maximum speakers. Omit for no limit.

    Returns:
        Status message with the path to the output CSV on success, or an
        error message on failure.
    """
    cmd = [
        sys.executable, "-m", "speech_mine.diarizer.cli",
        "extract", input_file, output_csv,
        "--hf-token", hf_token,
        "--model", model,
        "--device", device,
        "--compute-type", compute_type,
        "--min-speakers", str(min_speakers),
    ]
    if num_speakers is not None:
        cmd += ["--num-speakers", str(num_speakers)]
    if max_speakers is not None:
        cmd += ["--max-speakers", str(max_speakers)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return f"Extraction complete. CSV saved to: {output_csv}\n\n{result.stdout}"
    return f"Extraction failed (exit {result.returncode}):\n{result.stderr}"


@mcp.tool()
def chunk_audio(
    audio_file: str,
    config_file: str,
    output_dir: str,
    fade_in: int = 0,
    fade_out: int = 0,
    padding: int = 0,
) -> str:
    """
    Split a WAV audio file into timed segments defined by a YAML config.

    YAML format:
        chunks:
          - start: 0.0
            end: 30.0
            name: "intro"
          - start: 30.0
            end: 120.0
            name: "main_discussion"

    Args:
        audio_file: Path to the input .wav file.
        config_file: Path to the YAML config defining chunk boundaries.
        output_dir: Directory to write the output chunk files.
        fade_in: Fade-in duration in milliseconds (default 0).
        fade_out: Fade-out duration in milliseconds (default 0).
        padding: Silence padding in milliseconds added around each chunk (default 0).

    Returns:
        JSON list of paths to the generated chunk files, or an error message.
    """
    from .pickaxe.chunk import chunk_audio_file

    import os
    if not os.path.exists(audio_file):
        return f"Error: audio file not found: {audio_file}"
    if not os.path.exists(config_file):
        return f"Error: config file not found: {config_file}"

    output_files = chunk_audio_file(
        audio_path=audio_file,
        config_path=config_file,
        output_dir=output_dir,
        fade_in=fade_in,
        fade_out=fade_out,
        silence_padding=padding,
    )
    return json.dumps({"output_files": output_files, "count": len(output_files)}, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
