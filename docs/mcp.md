# MCP Server

speech-mine ships an [MCP](https://modelcontextprotocol.io) (Model Context Protocol) server that exposes all its capabilities as tools for Claude Code and other MCP-compatible clients.

## Installation

**One command, no clone needed:**

```bash
claude mcp add speech-mine -- uvx --from speech-mine speech-mine-mcp
```

This uses [`uvx`](https://docs.astral.sh/uv/guides/tools/) to pull the published version from PyPI and run the server. Restart Claude Code after running it.

**If you cloned the repo:** the included `.mcp.json` configures the server automatically when you open the project in Claude Code.

## Available tools

| Tool | Description |
|------|-------------|
| `search_transcript` | Fuzzy-search a transcript CSV for a word or phrase |
| `get_transcript_stats` | Word count, speaker list, duration, and confidence |
| `read_transcript` | Export transcript data as utterances, segments, words, or JSON |
| `format_transcript` | Convert a CSV to a human-readable script file |
| `extract_audio` | Transcribe audio with speaker diarization (spawns subprocess) |
| `chunk_audio` | Split a WAV file into timed segments from a YAML config |

## Usage examples

Once installed, Claude Code can use the tools directly. Some examples of what you can ask:

```
Search output.csv for every time someone mentioned "budget"
```

```
Get the stats for my transcript at ~/recordings/interview.csv
```

```
Chunk sample.wav using chunks.yaml and save to ./clips/
```

```
Format output.csv into a readable script and save to script.txt,
mapping SPEAKER_00 to "Alice" and SPEAKER_01 to "Bob" using speakers.json
```

## Tool reference

### `search_transcript`

```
csv_path       Path to the transcript CSV
query          Word, phrase, or sentence to search for
min_similarity Minimum similarity score (0.0–1.0, default 0.0)
max_similarity Maximum similarity score (0.0–1.0, default 1.0)
top_k          Maximum results to return (default 10)
output_type    "utterance" (default) or "timestamp"
metadata_path  Optional path to the companion _metadata.json
```

### `get_transcript_stats`

```
csv_path       Path to the transcript CSV
metadata_path  Optional path to the companion _metadata.json
```

### `read_transcript`

```
csv_path       Path to the transcript CSV
format_type    "utterances" (default), "segments", "words", or "json"
metadata_path  Optional path to the companion _metadata.json
```

### `format_transcript`

```
input_csv      Path to the transcript CSV
output_txt     Destination path for the formatted script
speakers_json  Optional JSON mapping SPEAKER_00 labels to real names
```

### `extract_audio`

Runs `speech-mine extract` in a subprocess. Can take several minutes for long files.

```
input_file     Path to audio (.wav, .mp3, .ogg, .flac, .m4a, .webm)
output_csv     Destination path for the output CSV
hf_token       HuggingFace access token (required for pyannote)
model          Whisper model size (default: large-v3)
device         "auto" (default), "cpu", or "cuda"
compute_type   "float16" (default), "int8", or "float32"
num_speakers   Exact speaker count if known
min_speakers   Minimum speakers (default 1)
max_speakers   Maximum speakers
batch_size     WhisperX transcription batch size (default 16, reduce if OOM)
language       Language code e.g. "en", "fr" (auto-detected if omitted)
```

### `chunk_audio`

```
audio_file     Path to the input .wav file
config_file    Path to the YAML config defining chunk boundaries
output_dir     Directory to write output chunk files
fade_in        Fade-in in milliseconds (default 0)
fade_out       Fade-out in milliseconds (default 0)
padding        Silence padding in milliseconds (default 0)
```

YAML config format:

```yaml
chunks:
  - start: 0.0
    end: 30.0
    name: "intro"
  - start: 30.0
    end: 120.0
    name: "main_discussion"
```
