
# speech-mine

Speech diarization and transcript analysis toolkit. Extract speaker-labeled transcripts from audio, format them into readable scripts, search them with fuzzy matching, and pre-process audio with chunking.

## Requirements

- Python 3.11+
- HuggingFace access token (for pyannote models)
- GPU recommended for faster processing
- ffmpeg installed (for audio loading)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd speech-mine

# Install dependencies and create virtual environment
uv sync
```

---

## Modules

### `extract` — Transcription + Speaker Diarization

Transcribes audio and labels each segment with the speaker who said it. Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription and [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

**Usage:**
```bash
uv run speech-mine extract <audio> <output.csv> --hf-token TOKEN [options]
```

**Supported audio formats:** `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm`

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-token TOKEN` | *(required)* | HuggingFace access token |
| `--model SIZE` | `large-v3` | Whisper model size |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--compute-type` | `float16` | `float16` (GPU), `float32` (CPU), `int8` |
| `--num-speakers N` | — | Exact speaker count (best accuracy when known) |
| `--min-speakers N` | `1` | Minimum expected speakers |
| `--max-speakers N` | — | Maximum expected speakers |
| `--verbose` | — | Enable verbose logging |

**Examples:**
```bash
# Basic (CPU)
uv run speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_TOKEN \
  --compute-type float32

# 2-person interview with known speaker count
uv run speech-mine extract interview.wav output.csv \
  --hf-token YOUR_TOKEN \
  --num-speakers 2 \
  --compute-type float32

# GPU with best accuracy model
uv run speech-mine extract meeting.wav output.csv \
  --hf-token YOUR_TOKEN \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --num-speakers 4
```

**Output:** Two files are written:
- `output.csv` — segment and word-level transcript data
- `output_metadata.json` — language, duration, speaker list, processing info

**⚠️ Important:** Always use `--compute-type float32` when running on CPU. The default (`float16`) requires a GPU.

---

### `format` — Script Formatting

Converts the CSV output from `extract` into a human-readable, movie-style script.

**Usage:**
```bash
uv run speech-mine format <input.csv> <output.txt> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--speakers FILE` | JSON file mapping `SPEAKER_00` → custom name |
| `--create-template` | Generate a speaker names template JSON from the CSV |

**Examples:**
```bash
# Basic formatting
uv run speech-mine format output.csv script.txt

# Create a speaker names template to fill in
uv run speech-mine format output.csv script.txt --create-template

# Format with custom speaker names
uv run speech-mine format output.csv script.txt --speakers output_speaker_names.json
```

**Custom speaker names workflow:**
```bash
# 1. Generate template (creates output_speaker_names.json)
uv run speech-mine format output.csv script.txt --create-template

# 2. Edit the template
# {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}

# 3. Format with names
uv run speech-mine format output.csv final_script.txt --speakers output_speaker_names.json
```

**Output format:**
```
================================================================================
TRANSCRIPT
================================================================================

RECORDING DETAILS:
----------------------------------------
File: interview.mp3
Duration: 08:47
Language: EN (confidence: 99.0%)
Speakers: 2
Processed: 2026-03-05 22:00:00

CAST:
----------------------------------------
SPEAKER A
SPEAKER B

TRANSCRIPT:
----------------------------------------

[00:00 - 00:05] SPEAKER A:
    So tell me about your background.

[00:06 - 00:12] SPEAKER B:
    Sure, I started out in radio back in the eighties.

    [...5:30 pause...]

[05:42 - 05:50] SPEAKER A:
    And how did that shape your career?
```

---

### `chunk-audio` — Audio Chunking

Splits a `.wav` file into smaller segments based on a YAML configuration defining time boundaries. Useful for pre-processing long recordings before extraction.

\*\*Usage:\*\*
```bash
uv run speech-mine chunk <audio.wav> <config.yaml> <output_dir/> [options]
```

**Note:** Only `.wav` input files are supported. Output chunks are also `.wav`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--fade-in MS` | `0` | Fade in duration in milliseconds |
| `--fade-out MS` | `0` | Fade out duration in milliseconds |
| `--padding MS` | `0` | Silence padding added to both ends (ms) |
| `--verbose` | — | Print file sizes for each chunk |

**YAML config format:**
```yaml
chunks:
  - start: 0.0
    end: 30.0
    name: "intro"        # optional — used in output filename
  - start: 30.0
    end: 120.0
    name: "discussion"
  - start: 120.0
    end: 300.0
    # no name — output will be "2.wav"
```

Output filenames follow the pattern `{index}.{name}.wav` or `{index}.wav` if no name is set. Chunks are sorted by start time before indexing.

**Examples:**
```bash
# Basic chunking
uv run speech-mine chunk recording.wav config.yaml chunks/

# With fade effects and padding
uv run speech-mine chunk recording.wav config.yaml chunks/ \
  --fade-in 500 \
  --fade-out 500 \
  --padding 100 \
  --verbose
```

---

### `search` — Fuzzy Transcript Search

Searches a transcript CSV for words or phrases using fuzzy matching. Returns ranked matches with timestamps, speaker context, and similarity scores.

\*\*Usage:\*\*
```bash
uv run speech-mine search <query> <transcript.csv> [metadata.json] [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--similarity-range MIN MAX` | `0.0 1.0` | Filter results by similarity score |
| `--top-k N` | `10` | Maximum results to return |
| `--output-type` | `utterance` | `utterance` or `timestamp` |
| `--save-path FILE` | — | Save JSON output to file (default: stdout) |

**Output types:**
- `utterance` — returns utterance index, word positions within the utterance, and full segment context
- `timestamp` — returns a time window with per-word start/end times

**Examples:**
```bash
# Search for a phrase, print to stdout
uv run speech-mine search "childhood abuse" output.csv

# High-confidence matches, save to file
uv run speech-mine search "radio career" output.csv output_metadata.json \
  --similarity-range 0.8 1.0 \
  --top-k 5 \
  --output-type timestamp \
  --save-path results.json
```

---

## Output Format Reference

### CSV (`output.csv`)

Contains interleaved segment-level and word-level rows:

| Column | Description |
|--------|-------------|
| `type` | `"segment"` or `"word"` |
| `speaker` | Speaker ID (e.g. `SPEAKER_00`) |
| `start` | Start time in seconds |
| `end` | End time in seconds |
| `text` | Full segment text |
| `word` | Individual word (word rows only) |
| `word_position` | Word index within segment (word rows only) |
| `confidence` | Confidence score (0–1) |
| `overlap_duration` | Speaker overlap duration in seconds |

### Metadata (`output_metadata.json`)

```json
{
  "audio_file": "interview.mp3",
  "language": "en",
  "language_probability": 0.99,
  "duration": 527.19,
  "total_segments": 87,
  "total_words": 1234,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "processing_timestamp": "2026-03-05 22:00:00"
}
```

---

## Setup

### HuggingFace Token

Required for pyannote diarization models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → New token (read permissions)
3. Accept the user agreement at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

Pass the token via `--hf-token YOUR_TOKEN` on every `extract` call.

---

## Model Options

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐⭐ | Quick tests |
| `base` | ⚡⚡⚡⚡ | ⭐⭐⭐ | Fast with decent quality |
| `small` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good balance |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐ | Higher accuracy |
| `large-v3` | ⚡ | ⭐⭐⭐⭐⭐ | Best accuracy (default) |
| `turbo` | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Fast + accurate |

**Compute type:**
- `float32` — works on CPU and GPU (required for CPU)
- `float16` — GPU only, faster (default)
- `int8` — fastest, slightly lower accuracy

**Speaker count tip:** Specifying `--num-speakers` when you know the exact count improves diarization accuracy by 15–30%.

---

## Troubleshooting

```bash
# float16 error on CPU → use float32
uv run speech-mine extract audio.wav out.csv \
  --hf-token TOKEN \
  --compute-type float32

# Safest CPU command
uv run speech-mine extract audio.wav out.csv \
  --hf-token TOKEN \
  --device cpu \
  --compute-type float32 \
  --model base
```

---

## License

TBD
