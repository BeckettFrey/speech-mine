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
import os
import subprocess
import sys
import traceback
from functools import wraps
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .access import TranscriptionAccessTool
from .fuzz import speech_fuzzy_match
from .diarizer.formatter import ScriptFormatter

mcp = FastMCP("speech-mine")


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

class ToolInputError(ValueError):
    """Raised for user-facing input validation failures in MCP tools."""


def _error(message: str, **context: Any) -> str:
    payload = {"error": message}
    if context:
        payload["details"] = {k: v for k, v in context.items() if v is not None}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _require_file(path: str, label: str) -> None:
    if not path:
        raise ToolInputError(f"{label} is required")
    if not os.path.exists(path):
        raise ToolInputError(f"{label} not found: {path}")
    if not os.path.isfile(path):
        raise ToolInputError(f"{label} is not a file: {path}")


def _require_choice(value: str, choices: tuple, label: str) -> None:
    if value not in choices:
        raise ToolInputError(
            f"invalid {label}: {value!r}. Must be one of: {', '.join(choices)}"
        )


def _safe_tool(func):
    """Decorator: convert exceptions in an MCP tool into a clear JSON error string."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolInputError as e:
            return _error(str(e), tool=func.__name__, error_type="input_error")
        except FileNotFoundError as e:
            return _error(
                f"file not found: {e.filename or e}",
                tool=func.__name__,
                error_type="file_not_found",
            )
        except PermissionError as e:
            return _error(
                f"permission denied: {e.filename or e}",
                tool=func.__name__,
                error_type="permission_error",
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _error(
                f"failed to parse file: {e}",
                tool=func.__name__,
                error_type="parse_error",
            )
        except KeyError as e:
            return _error(
                f"missing expected field: {e}. The input file may be malformed or "
                f"not a speech-mine transcript.",
                tool=func.__name__,
                error_type="schema_error",
            )
        except Exception as e:
            return _error(
                f"{type(e).__name__}: {e}",
                tool=func.__name__,
                error_type="internal_error",
                traceback=traceback.format_exc(limit=5),
            )
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tool(csv_path: str, metadata_path: Optional[str] = None) -> TranscriptionAccessTool:
    _require_file(csv_path, "csv_path")
    if metadata_path:
        _require_file(metadata_path, "metadata_path")
    tool = TranscriptionAccessTool()
    tool.load_from_files(csv_path, metadata_path)
    return tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe_tool
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
        speaker, and similarity scores. On failure, returns a JSON object
        with an `error` field describing what went wrong.
    """
    if not query or not query.strip():
        raise ToolInputError("query cannot be empty")
    _require_choice(output_type, ("utterance", "timestamp"), "output_type")
    if not 0.0 <= min_similarity <= 1.0:
        raise ToolInputError(
            f"min_similarity must be between 0.0 and 1.0, got {min_similarity}"
        )
    if not 0.0 <= max_similarity <= 1.0:
        raise ToolInputError(
            f"max_similarity must be between 0.0 and 1.0, got {max_similarity}"
        )
    if min_similarity > max_similarity:
        raise ToolInputError(
            f"min_similarity ({min_similarity}) cannot exceed max_similarity ({max_similarity})"
        )
    if top_k < 1:
        raise ToolInputError(f"top_k must be >= 1, got {top_k}")

    tool = _load_tool(csv_path, metadata_path)
    if not tool.words:
        raise ToolInputError(
            f"no words found in transcript: {csv_path}. The file may be empty "
            f"or not a speech-mine transcript CSV."
        )

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
@_safe_tool
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
@_safe_tool
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
    _require_choice(
        format_type,
        ("utterances", "segments", "words", "json"),
        "format_type",
    )
    tool = _load_tool(csv_path, metadata_path)
    data = tool.export(format_type)
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
@_safe_tool
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
        The content of the formatted script, or a JSON error payload.
    """
    _require_file(input_csv, "input_csv")
    if not output_txt:
        raise ToolInputError("output_txt is required")
    if speakers_json:
        _require_file(speakers_json, "speakers_json")

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
@_safe_tool
def extract_audio(
    input_file: str,
    output_csv: str,
    hf_token: Optional[str] = None,
    model: str = "large-v3",
    device: str = "auto",
    compute_type: str = "float16",
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: Optional[int] = None,
    batch_size: int = 16,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe an audio file with forced alignment and speaker diarization, saving to CSV.

    Uses WhisperX: transcription → forced word-level alignment → speaker diarization.
    This runs `speech-mine extract` in a subprocess — it can take several minutes.

    Args:
        input_file: Path to the input audio (.wav, .mp3, .ogg, .flac, .m4a, .webm).
        output_csv: Destination path for the output CSV transcript.
        hf_token: HuggingFace access token (required for speaker diarization).
            Prefer leaving this unset and configuring `HF_TOKEN` in the MCP
            server's env block — the server will pick it up automatically and
            the secret never enters the LLM conversation.
        model: Whisper model size. One of tiny, base, small, medium,
               large-v2, large-v3 (default), turbo.
        device: Device — "auto" (default), "cpu", or "cuda".
        compute_type: "float16" (default), "int8", or "float32".
        num_speakers: Exact speaker count if known (improves accuracy).
        min_speakers: Minimum speakers. Default 1.
        max_speakers: Maximum speakers. Omit for no limit.
        batch_size: WhisperX transcription batch size (default 16, reduce if OOM).
        language: Language code (e.g. "en", "fr"). Auto-detected if omitted.

    Returns:
        Status message with the path to the output CSV on success, or an
        error message on failure.
    """
    _require_file(input_file, "input_file")
    if not output_csv:
        raise ToolInputError("output_csv is required")
    _require_choice(device, ("auto", "cpu", "cuda"), "device")
    _require_choice(compute_type, ("float16", "int8", "float32"), "compute_type")
    _require_choice(
        model,
        ("tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"),
        "model",
    )
    if min_speakers < 1:
        raise ToolInputError(f"min_speakers must be >= 1, got {min_speakers}")
    if num_speakers is not None and num_speakers < 1:
        raise ToolInputError(f"num_speakers must be >= 1, got {num_speakers}")
    if max_speakers is not None and max_speakers < min_speakers:
        raise ToolInputError(
            f"max_speakers ({max_speakers}) cannot be less than min_speakers ({min_speakers})"
        )
    if batch_size < 1:
        raise ToolInputError(f"batch_size must be >= 1, got {batch_size}")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ToolInputError(
            "no HuggingFace token available. Set HF_TOKEN in the MCP server's "
            "environment (preferred) or pass it as the `hf_token` argument."
        )

    cmd = [
        sys.executable, "-m", "speech_mine.diarizer.cli",
        "extract", input_file, output_csv,
        "--hf-token", token,
        "--model", model,
        "--device", device,
        "--compute-type", compute_type,
        "--min-speakers", str(min_speakers),
        "--batch-size", str(batch_size),
    ]
    if num_speakers is not None:
        cmd += ["--num-speakers", str(num_speakers)]
    if max_speakers is not None:
        cmd += ["--max-speakers", str(max_speakers)]
    if language is not None:
        cmd += ["--language", language]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        return _error(
            f"failed to launch extraction subprocess: {e}. Is the speech-mine "
            f"package installed in the MCP server's Python environment?",
            tool="extract_audio",
            error_type="subprocess_launch_error",
        )

    if result.returncode == 0:
        return f"Extraction complete. CSV saved to: {output_csv}\n\n{result.stdout}"
    return _error(
        f"extraction subprocess exited with code {result.returncode}",
        tool="extract_audio",
        error_type="subprocess_failed",
        exit_code=result.returncode,
        stderr=(result.stderr or "").strip() or None,
        stdout_tail=(result.stdout or "").strip()[-500:] or None,
        command=" ".join(cmd[:3]) + " ...",
    )


@mcp.tool()
@_safe_tool
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

    _require_file(audio_file, "audio_file")
    _require_file(config_file, "config_file")
    if not output_dir:
        raise ToolInputError("output_dir is required")
    if fade_in < 0:
        raise ToolInputError(f"fade_in must be >= 0, got {fade_in}")
    if fade_out < 0:
        raise ToolInputError(f"fade_out must be >= 0, got {fade_out}")
    if padding < 0:
        raise ToolInputError(f"padding must be >= 0, got {padding}")

    os.makedirs(output_dir, exist_ok=True)

    output_files = chunk_audio_file(
        audio_path=audio_file,
        config=config_file,
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
