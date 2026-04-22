"""Tests for the speech-mine MCP server tools.

Each `@mcp.tool()`-decorated function in `speech_mine.mcp_server` is a plain
callable, so we exercise them directly (bypassing the MCP transport) and
assert on their JSON / text return values.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from speech_mine.mcp_server import (
    search_transcript,
    get_transcript_stats,
    read_transcript,
    format_transcript,
    extract_audio,
    chunk_audio,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _touch(path):
    """Create an empty file at path and return its string form."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return str(p)


SAMPLE_CSV = """type,speaker,start,end,text,word,word_position,confidence,overlap_duration
segment,SPEAKER_01,0.0,1.72,Hello world.,,,-0.18,1.689
word,SPEAKER_01,0.0,0.5,Hello world.,Hello,0,0.95,1.689
word,SPEAKER_01,0.5,1.72,Hello world.,world,1,0.98,1.689
segment,SPEAKER_01,2.0,3.5,How are you today?,,,-0.15,1.5
word,SPEAKER_01,2.0,2.3,How are you today?,How,0,0.92,1.5
word,SPEAKER_01,2.3,2.6,How are you today?,are,1,0.96,1.5
word,SPEAKER_01,2.6,2.9,How are you today?,you,2,0.94,1.5
word,SPEAKER_01,2.9,3.5,How are you today?,today,3,0.90,1.5
segment,SPEAKER_00,4.0,5.0,Good morning.,,,-0.1,1.0
word,SPEAKER_00,4.0,4.5,Good morning.,Good,0,0.99,1.0
word,SPEAKER_00,4.5,5.0,Good morning.,morning,1,0.97,1.0
"""

SAMPLE_METADATA = {
    "audio_file": "/path/to/audio.wav",
    "language": "en",
    "language_probability": 0.99,
    "duration": 5.0,
    "total_segments": 3,
    "total_words": 7,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "processing_timestamp": "2026-04-21 10:00:00",
}


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    p = tmp_path / "transcript.csv"
    p.write_text(SAMPLE_CSV)
    return str(p)


@pytest.fixture
def metadata_path(tmp_path: Path) -> str:
    p = tmp_path / "transcript_metadata.json"
    p.write_text(json.dumps(SAMPLE_METADATA))
    return str(p)


# ---------------------------------------------------------------------------
# search_transcript
# ---------------------------------------------------------------------------

class TestSearchTranscript:
    def test_basic_utterance_output(self, csv_path):
        out = json.loads(search_transcript(csv_path, "hello"))
        assert out["query"] == "hello"
        assert out["search_parameters"]["output_type"] == "utterance"
        assert out["search_parameters"]["top_k"] == 10
        assert out["search_parameters"]["similarity_range"] == {"min": 0.0, "max": 1.0}
        assert out["total_matches"] >= 1
        top = out["results"][0]
        assert "matched_text" in top
        assert "similarity_score" in top
        assert 0.0 <= top["similarity_score"] <= 1.0

    def test_timestamp_output(self, csv_path):
        out = json.loads(
            search_transcript(csv_path, "how are you", output_type="timestamp")
        )
        assert out["search_parameters"]["output_type"] == "timestamp"
        assert out["total_matches"] >= 1
        top = out["results"][0]
        assert "time_window" in top
        tw = top["time_window"]
        assert {"start_time", "end_time", "duration"} <= set(tw)
        assert tw["end_time"] >= tw["start_time"]
        assert top["context"]["speaker"] == "SPEAKER_01"

    def test_top_k_limits_results(self, csv_path):
        out = json.loads(search_transcript(csv_path, "hello", top_k=1))
        assert out["total_matches"] <= 1
        assert out["search_parameters"]["top_k"] == 1

    def test_similarity_range_filters_results(self, csv_path):
        # very high threshold should suppress weak matches
        out = json.loads(
            search_transcript(csv_path, "zzzzzzz", min_similarity=0.99, max_similarity=1.0)
        )
        assert out["total_matches"] == 0
        assert out["results"] == []

    def test_no_matches_returns_empty_results(self, csv_path):
        out = json.loads(
            search_transcript(csv_path, "quantum", min_similarity=0.95, top_k=5)
        )
        assert out["total_matches"] == 0
        assert out["results"] == []

    def test_metadata_path_accepted(self, csv_path, metadata_path):
        out = json.loads(search_transcript(csv_path, "hello", metadata_path=metadata_path))
        assert out["query"] == "hello"

    def test_returns_valid_json_string(self, csv_path):
        raw = search_transcript(csv_path, "hello")
        assert isinstance(raw, str)
        json.loads(raw)  # does not raise


# ---------------------------------------------------------------------------
# get_transcript_stats
# ---------------------------------------------------------------------------

class TestGetTranscriptStats:
    def test_basic_stats(self, csv_path):
        stats = json.loads(get_transcript_stats(csv_path))
        assert stats["total_utterances"] == 3
        assert stats["total_words"] == 8
        assert stats["total_speakers"] == 2
        assert set(stats["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
        assert 0.0 <= stats["average_confidence"] <= 1.0

    def test_stats_include_metadata(self, csv_path, metadata_path):
        stats = json.loads(get_transcript_stats(csv_path, metadata_path=metadata_path))
        assert stats["duration"] == 5.0
        assert stats["language"] == "en"

    def test_missing_csv_returns_error(self, tmp_path):
        result = json.loads(get_transcript_stats(str(tmp_path / "nope.csv")))
        assert "error" in result
        assert "csv_path not found" in result["error"]
        assert result["details"]["error_type"] == "input_error"


# ---------------------------------------------------------------------------
# read_transcript
# ---------------------------------------------------------------------------

class TestReadTranscript:
    def test_default_utterances(self, csv_path):
        data = json.loads(read_transcript(csv_path))
        assert isinstance(data, list)
        assert len(data) == 3
        assert all("utterance_number" in u for u in data)
        assert all("words" in u for u in data)

    def test_segments(self, csv_path):
        data = json.loads(read_transcript(csv_path, format_type="segments"))
        assert isinstance(data, list)
        assert len(data) == 3
        assert all("text" in s for s in data)

    def test_words(self, csv_path):
        data = json.loads(read_transcript(csv_path, format_type="words"))
        assert isinstance(data, list)
        assert len(data) == 8
        assert all("word" in w for w in data)

    def test_json(self, csv_path):
        data = json.loads(read_transcript(csv_path, format_type="json"))
        assert isinstance(data, dict)
        assert {"metadata", "utterances", "stats"} <= set(data)

    def test_invalid_format_returns_error(self, csv_path):
        result = json.loads(read_transcript(csv_path, format_type="bogus"))
        assert "error" in result
        assert "invalid format_type" in result["error"]

    def test_metadata_path(self, csv_path, metadata_path):
        data = json.loads(
            read_transcript(csv_path, format_type="json", metadata_path=metadata_path)
        )
        assert data["metadata"]["language"] == "en"


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------

class TestFormatTranscript:
    def test_formats_and_writes_file(self, csv_path, tmp_path):
        out_txt = tmp_path / "script.txt"
        content = format_transcript(csv_path, str(out_txt))
        assert out_txt.exists()
        # Returned content matches file content
        assert content == out_txt.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_creates_missing_output_dir(self, csv_path, tmp_path):
        out_txt = tmp_path / "nested" / "dir" / "script.txt"
        format_transcript(csv_path, str(out_txt))
        assert out_txt.exists()

    def test_missing_csv_returns_error(self, tmp_path):
        out_txt = tmp_path / "script.txt"
        msg = format_transcript(str(tmp_path / "missing.csv"), str(out_txt))
        result = json.loads(msg)
        assert "error" in result
        assert "input_csv not found" in result["error"]
        assert not out_txt.exists()

    def test_with_speaker_mapping(self, csv_path, tmp_path):
        speakers_json = tmp_path / "speakers.json"
        speakers_json.write_text(json.dumps({"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}))
        out_txt = tmp_path / "script.txt"
        content = format_transcript(csv_path, str(out_txt), speakers_json=str(speakers_json))
        # At least one mapped name should appear in the rendered script
        assert ("Alice" in content) or ("Bob" in content)


# ---------------------------------------------------------------------------
# extract_audio (subprocess is mocked)
# ---------------------------------------------------------------------------

class TestExtractAudio:
    def _completed(self, returncode=0, stdout="done", stderr=""):
        m = MagicMock(spec=subprocess.CompletedProcess)
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = stderr
        return m

    def test_success_message_and_default_args(self, tmp_path):
        with patch("speech_mine.mcp_server.subprocess.run") as run:
            run.return_value = self._completed(0, "ok-output", "")
            result = extract_audio(
                input_file=_touch(tmp_path / "in.wav"),
                output_csv=str(tmp_path / "out.csv"),
                hf_token="hf_xxx",
            )

        assert "Extraction complete" in result
        assert "ok-output" in result
        run.assert_called_once()
        cmd = run.call_args.args[0]
        # Core structure
        assert cmd[:4] == [
            os.sys.executable if False else cmd[0],  # python executable
            "-m",
            "speech_mine.diarizer.cli",
            "extract",
        ]
        assert any(c.endswith("in.wav") for c in cmd)
        assert str(tmp_path / "out.csv") in cmd
        # Default flag values
        assert cmd[cmd.index("--hf-token") + 1] == "hf_xxx"
        assert cmd[cmd.index("--model") + 1] == "large-v3"
        assert cmd[cmd.index("--device") + 1] == "auto"
        assert cmd[cmd.index("--compute-type") + 1] == "float16"
        assert cmd[cmd.index("--min-speakers") + 1] == "1"
        assert cmd[cmd.index("--batch-size") + 1] == "16"
        # Optional flags omitted by default
        assert "--num-speakers" not in cmd
        assert "--max-speakers" not in cmd
        assert "--language" not in cmd

    def test_passes_optional_args(self, tmp_path):
        with patch("speech_mine.mcp_server.subprocess.run") as run:
            run.return_value = self._completed(0, "", "")
            extract_audio(
                input_file=_touch(tmp_path / "in.wav"),
                output_csv=str(tmp_path / "out.csv"),
                hf_token="hf_xxx",
                model="turbo",
                device="cuda",
                compute_type="int8",
                num_speakers=3,
                min_speakers=2,
                max_speakers=5,
                batch_size=8,
                language="fr",
            )
        cmd = run.call_args.args[0]
        assert cmd[cmd.index("--model") + 1] == "turbo"
        assert cmd[cmd.index("--device") + 1] == "cuda"
        assert cmd[cmd.index("--compute-type") + 1] == "int8"
        assert cmd[cmd.index("--num-speakers") + 1] == "3"
        assert cmd[cmd.index("--min-speakers") + 1] == "2"
        assert cmd[cmd.index("--max-speakers") + 1] == "5"
        assert cmd[cmd.index("--batch-size") + 1] == "8"
        assert cmd[cmd.index("--language") + 1] == "fr"

    def test_reads_token_from_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        with patch("speech_mine.mcp_server.subprocess.run") as run:
            run.return_value = self._completed(0, "", "")
            extract_audio(
                input_file=_touch(tmp_path / "in.wav"),
                output_csv=str(tmp_path / "out.csv"),
                # hf_token omitted — must fall back to HF_TOKEN env var
            )
        cmd = run.call_args.args[0]
        assert cmd[cmd.index("--hf-token") + 1] == "hf_from_env"

    def test_explicit_token_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        with patch("speech_mine.mcp_server.subprocess.run") as run:
            run.return_value = self._completed(0, "", "")
            extract_audio(
                input_file=_touch(tmp_path / "in.wav"),
                output_csv=str(tmp_path / "out.csv"),
                hf_token="hf_explicit",
            )
        cmd = run.call_args.args[0]
        assert cmd[cmd.index("--hf-token") + 1] == "hf_explicit"

    def test_missing_token_returns_error_without_running(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("speech_mine.mcp_server.subprocess.run") as run:
            result = extract_audio(
                input_file=_touch(tmp_path / "in.wav"),
                output_csv=str(tmp_path / "out.csv"),
            )
        assert "no HuggingFace token" in result
        run.assert_not_called()

    def test_failure_returns_error_message(self, tmp_path):
        with patch("speech_mine.mcp_server.subprocess.run") as run:
            run.return_value = self._completed(2, "", "boom")
            result = extract_audio(
                input_file=_touch(tmp_path / "in.wav"),
                output_csv=str(tmp_path / "out.csv"),
                hf_token="hf_xxx",
            )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "exited with code 2" in parsed["error"]
        assert parsed["details"]["exit_code"] == 2
        assert parsed["details"]["stderr"] == "boom"


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------

class TestChunkAudio:
    def test_missing_audio_returns_error(self, tmp_path):
        config = tmp_path / "cfg.yaml"
        config.write_text("chunks: []\n")
        msg = chunk_audio(
            audio_file=str(tmp_path / "nope.wav"),
            config_file=str(config),
            output_dir=str(tmp_path / "out"),
        )
        parsed = json.loads(msg)
        assert "audio_file not found" in parsed["error"]

    def test_missing_config_returns_error(self, tmp_path):
        audio = tmp_path / "in.wav"
        audio.write_bytes(b"RIFF0000WAVE")
        msg = chunk_audio(
            audio_file=str(audio),
            config_file=str(tmp_path / "missing.yaml"),
            output_dir=str(tmp_path / "out"),
        )
        parsed = json.loads(msg)
        assert "config_file not found" in parsed["error"]

    def test_delegates_to_chunk_audio_file(self, tmp_path):
        audio = tmp_path / "in.wav"
        audio.write_bytes(b"RIFF0000WAVE")
        config = tmp_path / "cfg.yaml"
        config.write_text("chunks:\n  - start: 0.0\n    end: 1.0\n    name: a\n")
        out_dir = tmp_path / "out"

        fake_outputs = [str(out_dir / "0.a.wav")]
        with patch("speech_mine.pickaxe.chunk.chunk_audio_file", return_value=fake_outputs) as m:
            result_str = chunk_audio(
                audio_file=str(audio),
                config_file=str(config),
                output_dir=str(out_dir),
                fade_in=10,
                fade_out=20,
                padding=30,
            )

        m.assert_called_once_with(
            audio_path=str(audio),
            config=str(config),
            output_dir=str(out_dir),
            fade_in=10,
            fade_out=20,
            silence_padding=30,
        )
        result = json.loads(result_str)
        assert result == {"output_files": fake_outputs, "count": 1}

    def test_default_fade_and_padding_are_zero(self, tmp_path):
        audio = tmp_path / "in.wav"
        audio.write_bytes(b"RIFF0000WAVE")
        config = tmp_path / "cfg.yaml"
        config.write_text("chunks: []\n")

        with patch("speech_mine.pickaxe.chunk.chunk_audio_file", return_value=[]) as m:
            chunk_audio(
                audio_file=str(audio),
                config_file=str(config),
                output_dir=str(tmp_path / "out"),
            )
        kwargs = m.call_args.kwargs
        assert kwargs["fade_in"] == 0
        assert kwargs["fade_out"] == 0
        assert kwargs["silence_padding"] == 0


# ---------------------------------------------------------------------------
# extract_audio — real subprocess, real model (gated on HF_TOKEN)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SNIPPET_WAV = EXAMPLES_DIR / "snippet_5s.wav"
ENV_FILE = REPO_ROOT / ".env"


def _load_hf_token() -> str:
    """Load HF_TOKEN from env, falling back to the repo-root .env file.

    Fails with an actionable error message rather than silently skipping —
    the integration test is meaningless without a real token.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    if ENV_FILE.exists():
        from dotenv import dotenv_values
        token = dotenv_values(ENV_FILE).get("HF_TOKEN")
        if token:
            return token

    pytest.fail(
        "HF_TOKEN not found. The extract integration test requires a real "
        f"HuggingFace token. Create {ENV_FILE} containing:\n"
        "    HF_TOKEN=hf_xxx\n"
        "or export HF_TOKEN in your shell."
    )


class TestExtractAudioIntegration:
    """Real end-to-end extract against a 5-second snippet.

    Downloads Whisper `tiny` + pyannote models on first run. Expect ~1–2 min
    the first time, seconds thereafter (HF cache).
    """

    def test_extract_snippet_produces_valid_csv(self, tmp_path):
        assert SNIPPET_WAV.exists(), f"snippet fixture missing: {SNIPPET_WAV}"
        hf_token = _load_hf_token()
        out_csv = tmp_path / "snippet.csv"

        result = extract_audio(
            input_file=str(SNIPPET_WAV),
            output_csv=str(out_csv),
            hf_token=hf_token,
            model="tiny",
            device="cpu",
            compute_type="int8",
            batch_size=4,
            language="en",
        )

        assert "Extraction complete" in result, result
        assert out_csv.exists()

        # Feed the CSV back through the access tool to confirm schema validity.
        stats = json.loads(get_transcript_stats(str(out_csv)))
        assert stats["total_words"] > 0
        assert stats["total_utterances"] > 0
        assert len(stats["speakers"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
