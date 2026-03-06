# speech-mine

**Speech diarization and transcript analysis toolkit.**

Turn raw audio files into speaker-labeled transcripts, then search, format, and slice them — all from the command line or your own Python code.

---

## Installation

```bash
pip install speech-mine
```

> **Requirements:** Python 3.11+, [ffmpeg](https://ffmpeg.org/) installed and on `PATH`. GPU recommended for faster transcription (not required).

---

## What it does

`speech-mine` is built around four focused tools that take you from raw audio to analyzed transcripts:

| Tool | What it does |
|------|--------------|
| `extract` | Transcribes an audio file and labels each segment with the speaker who said it. Produces a structured CSV transcript. |
| `format` | Converts a CSV transcript into a clean, human-readable script. |
| `search` | Fuzzy-searches a transcript for a word or phrase, returning the matching lines and their timestamps. |
| `chunk` | Splits a long audio file into smaller segments based on a YAML config. |

---

## Quick start

### 1. Extract a transcript

```bash
speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_HUGGINGFACE_TOKEN \
  --num-speakers 2
```

This produces a CSV with columns: `speaker`, `start`, `end`, `text`.

> The `extract` command requires a free [HuggingFace](https://huggingface.co/) access token and acceptance of the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model agreement.

### 2. Format into a readable script

```bash
speech-mine format output.csv script.txt
```

### 3. Search the transcript

```bash
speech-mine search "topic of interest" output.csv --pretty
```

### 4. Pre-process audio by chunking

```bash
speech-mine chunk config.yaml
```

---

## Programmatic usage

Every tool is also available as a Python API:

```python
from speech_mine.diarizer import SpeechDiarizationProcessor, ScriptFormatter
from speech_mine.fuzz import speech_fuzzy_match
from speech_mine.pickaxe import AudioChunker

# Extract a transcript
processor = SpeechDiarizationProcessor(hf_token="YOUR_TOKEN")
processor.process("interview.mp3", "output.csv", num_speakers=2)

# Format into a readable script
formatter = ScriptFormatter()
formatter.format_csv("output.csv", "script.txt")

# Search the transcript
results = speech_fuzzy_match(word_list, "topic of interest")

# Chunk audio
chunker = AudioChunker("config.yaml")
chunker.run()
```

See the [full documentation](https://beckettfrey.github.io/speech-mine) for detailed API reference.

---

## Documentation

Full documentation, including installation details, model options, output format reference, and troubleshooting, is available on the project's GitHub Pages site:

**[https://beckettfrey.github.io/speech-mine](https://beckettfrey.github.io/speech-mine)**

---

## License

TBD
