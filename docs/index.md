# speech-mine

Speech diarization and transcript analysis toolkit. Extract speaker-labeled transcripts from audio using [WhisperX](https://github.com/m-bain/whisperX), then format, search, and chunk them.

## Installation

```bash
pip install speech-mine
```

## Claude Code / MCP

speech-mine includes an MCP server. To add it to Claude Code:

```bash
claude mcp add speech-mine -- uvx --from speech-mine speech-mine-mcp
```

See [MCP Server](mcp.md) for all available tools and usage.

## Modules

| Module | Description |
|--------|-------------|
| [`extract`](extract.md) | Transcribe audio with speaker diarization |
| [`format`](format.md) | Format CSV transcripts into readable scripts |
| [`chunk`](chunk.md) | Split audio into segments via YAML config |
| [`search`](search.md) | Fuzzy search transcripts by word or phrase |

## Quick Start

```bash
# 1. Extract a transcript
speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_TOKEN \
  --num-speakers 2 \
  --compute-type float32

# 2. Format it into a readable script
speech-mine format output.csv script.txt

# 3. Search it
speech-mine search "topic of interest" output.csv --pretty
```

See [Installation](installation.md) to get started.
