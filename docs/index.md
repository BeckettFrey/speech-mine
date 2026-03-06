# speech-mine

![speech-mine](hero.png)

Speech diarization and transcript analysis toolkit. Extract speaker-labeled transcripts from audio, format them into readable scripts, search them with fuzzy matching, and pre-process audio with chunking.

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
uv run speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_TOKEN \
  --num-speakers 2 \
  --compute-type float32

# 2. Format it into a readable script
uv run speech-mine format output.csv script.txt

# 3. Search it
uv run speech-mine search "topic of interest" output.csv --pretty
```

See [Installation](installation.md) to get started.
