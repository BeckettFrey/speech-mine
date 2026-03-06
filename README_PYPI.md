# speech-mine

[![GitHub](https://img.shields.io/badge/GitHub-BeckettFrey%2Fspeech--mine-181717?logo=github)](https://github.com/BeckettFrey/speech-mine)

Speech diarization and transcript analysis toolkit. Extract speaker-labeled transcripts from audio, format them into readable scripts, search them with fuzzy matching, and pre-process audio with chunking.

## Modules

| Module | Description |
|--------|-------------|
| `extract` | Transcribe audio with speaker diarization |
| `format` | Format CSV transcripts into readable scripts |
| `chunk` | Split audio into segments via YAML config |
| `search` | Fuzzy search transcripts by word or phrase |

## Installation

```bash
pip install speech-mine
```

Requires Python 3.11+. See the [full installation guide](https://beckettfrey.github.io/speech-mine/installation) for library dependency setup and HuggingFace token configuration.

## Usage

speech-mine can be used as a **CLI tool** or as a **Python library**.

**CLI** — a `speech-mine` command is installed automatically. Run `speech-mine --help` for all commands and options, or see the [documentation](https://beckettfrey.github.io/speech-mine) for full details.

**Python API** — all modules are importable directly:

```python
from speech_mine import SpeechDiarizationProcessor, ScriptFormatter
from speech_mine import TranscriptionAccessTool, speech_fuzzy_match
from speech_mine.pickaxe.chunk import chunk_audio_file
```

## Programmatic Usage

### Extract — Diarize & transcribe audio

```python
from speech_mine import SpeechDiarizationProcessor

processor = SpeechDiarizationProcessor(
    whisper_model_size="large-v3",
    hf_token="YOUR_HF_TOKEN",
    num_speakers=2,
    compute_type="float32",
)

processor.process_audio_file("interview.mp3", "output.csv")
# Produces output.csv and output_metadata.json
```

### Format — Render a readable script

```python
from speech_mine import ScriptFormatter

formatter = ScriptFormatter(
    custom_speakers={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
)
formatter.format_script("output.csv", "script.txt")
```

### Search — Fuzzy phrase search

```python
from speech_mine import TranscriptionAccessTool, speech_fuzzy_match

tool = TranscriptionAccessTool()
tool.load_from_files("output.csv", "output_metadata.json")

# Fuzzy search across all words
results = speech_fuzzy_match(tool.words, "climate change", similarity_range=(0.8, 1.0), top_k=5)
for start_idx, end_idx, score in results:
    print(f"Match {score:.2f}: words {start_idx}–{end_idx}")

# Exact substring search
matches = tool.search_words("revenue")
for m in matches:
    print(m["word_data"]["word"], "at", m["word_data"]["start"])

# Words within a time window
words = tool.get_words_by_time_range(30.0, 60.0)
```

### Access — Inspect utterances and words

```python
from speech_mine import TranscriptionAccessTool

tool = TranscriptionAccessTool()
tool.load_from_files("output.csv", "output_metadata.json")

# Fetch a single word (utterance 0, word index 2)
word = tool.get_word(0, 2)

# Fetch a word range
word_range = tool.get_word_range(0, 1, 4)

# Fetch an entire utterance
utterance = tool.get_utterance(0)

# Summary statistics
stats = tool.get_stats()
print(stats)

# Export to JSON
data = tool.export("json")
```

### Chunk — Split audio by time boundaries

Only `.wav` files are supported as input.

```python
from speech_mine.pickaxe.chunk import chunk_audio_file

# chunks.yaml defines start/end times for each segment
output_files = chunk_audio_file(
    audio_path="long_recording.wav",
    config_path="chunks.yaml",
    output_dir="chunks/",
    fade_in=50,
    fade_out=50,
)
print(output_files)  # ['chunks/0.intro.wav', 'chunks/1.wav', ...]
```

Example `chunks.yaml`:

```yaml
chunks:
  - start: 0
    end: 120
    name: intro
  - start: 120
    end: 300
```

## Documentation

Full documentation is available at [beckettfrey.github.io/speech-mine](https://beckettfrey.github.io/speech-mine).

## License

MIT
