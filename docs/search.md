# search — Fuzzy Transcript Search

![search diagram](diagrams/search.svg)

Searches a transcript CSV for words or phrases using fuzzy matching. Returns ranked matches with timestamps, speaker context, and similarity scores.

## CLI

```bash
speech-mine search <query> <transcript.csv> [metadata.json] [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--similarity-range MIN MAX` | `0.0 1.0` | Filter results by similarity score |
| `--top-k N` | `10` | Maximum results to return |
| `--output-type` | `utterance` | `utterance` or `timestamp` |
| `--save-path FILE` | — | Save results as JSON (default: stdout) |
| `--pretty` | — | Display results in formatted, colored output |

### Examples

```bash
# Search and print JSON to stdout
speech-mine search "childhood abuse" output.csv

# Formatted colored output
speech-mine search "childhood abuse" output.csv --pretty

# High-confidence matches, save to file
speech-mine search "radio career" output.csv output_metadata.json \
  --similarity-range 0.8 1.0 \
  --top-k 5 \
  --output-type timestamp \
  --save-path results.json
```

### Output types

**`utterance`** (default) — Returns the utterance number, word positions within the utterance, and full segment context:

```json
{
  "similarity_score": 0.923,
  "utterance_number": 14,
  "matched_text": "radio career",
  "time_span": { "start": 42.1, "end": 43.0, "duration": 0.9 },
  "context": {
    "speaker": "SPEAKER_00",
    "full_segment_text": "Sure, I started out in radio..."
  }
}
```

**`timestamp`** — Returns a time window with per-word start/end times:

```json
{
  "similarity_score": 0.923,
  "matched_text": "radio career",
  "time_window": { "start_time": 42.1, "end_time": 43.0, "duration": 0.9 },
  "word_details": [
    { "word": "radio", "start": 42.1, "end": 42.5, "confidence": 0.98 },
    { "word": "career", "start": 42.6, "end": 43.0, "confidence": 0.95 }
  ]
}
```

## Library

```python
from speech_mine.access import TranscriptionAccessTool
from speech_mine.fuzz import speech_fuzzy_match

# Load transcript
tool = TranscriptionAccessTool()
tool.load_from_files("output.csv", "output_metadata.json")

# Search
matches = speech_fuzzy_match(
    word_list=tool.words,
    query="radio career",
    similarity_range=(0.8, 1.0),
    top_k=5,
)

# matches: list of (start_index, end_index, similarity_score)
for start_idx, end_idx, score in matches:
    words = tool.words[start_idx:end_idx + 1]
    print(f"{score:.2f}: {' '.join(w.word for w in words)}")
```

## How matching works

The fuzzy matcher uses [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) to compare your query against sliding windows of words in the transcript. Window sizes of `query_length - 1`, `query_length`, and `query_length + 1` words are all tested. Overlapping matches are deduplicated, keeping the highest-scoring one.

Similarity scores range from `0.0` (no match) to `1.0` (exact match).
