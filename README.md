
# speech-mine

![speech-mine](docs/hero.png)

Speech diarization and transcript analysis toolkit. Extract speaker-labeled transcripts from audio, format them into readable scripts, search them with fuzzy matching, and pre-process audio with chunking.

## Modules

| Module | Description | Docs |
|--------|-------------|------|
| `extract` | Transcribe audio with speaker diarization | [→](docs/extract.md) |
| `format` | Format CSV transcripts into readable scripts | [→](docs/format.md) |
| `chunk` | Split audio into segments via YAML config | [→](docs/chunk.md) |
| `search` | Fuzzy search transcripts by word or phrase | [→](docs/search.md) |

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository-url>
cd speech-mine
uv sync
```

See [docs/installation.md](docs/installation.md) for library dependency setup and HuggingFace token configuration.

## Quick Start

```bash
# 1. Extract a transcript
uv run speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_TOKEN \
  --num-speakers 2 \
  --compute-type float32

# 2. Format into a readable script
uv run speech-mine format output.csv script.txt

# 3. Search it
uv run speech-mine search "topic of interest" output.csv --pretty
```

## Documentation

```bash
# Serve docs locally
uv run mkdocs serve
```

Or browse the `docs/` folder directly.

## License

TBD
