# extract — Transcription + Speaker Diarization

![extract diagram](diagrams/extract.svg)

Transcribes audio with word-level timestamps and labels each word with the speaker who said it. Powered by [WhisperX](https://github.com/m-bain/whisperX): Whisper transcription → wav2vec2 forced alignment → pyannote speaker diarization.

**Supported audio formats:** `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm`

## CLI

```bash
speech-mine extract <audio> <output.csv> --hf-token TOKEN [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-token TOKEN` | *(required)* | HuggingFace access token |
| `--model SIZE` | `large-v3` | Whisper model size |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--compute-type` | `float16` | `float16` (GPU), `float32` (CPU), `int8` |
| `--num-speakers N` | — | Exact speaker count (best accuracy when known) |
| `--min-speakers N` | `1` | Minimum expected speakers |
| `--max-speakers N` | — | Maximum expected speakers |
| `--batch-size N` | `16` | Transcription batch size (reduce if out of memory) |
| `--language CODE` | — | Language code e.g. `en`, `fr` (auto-detected if omitted) |
| `--verbose` | — | Enable verbose logging |

### Examples

```bash
# Basic (CPU)
speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_TOKEN \
  --compute-type float32

# 2-person interview with known speaker count
speech-mine extract interview.wav output.csv \
  --hf-token YOUR_TOKEN \
  --num-speakers 2 \
  --compute-type float32

# GPU with best accuracy model
speech-mine extract meeting.wav output.csv \
  --hf-token YOUR_TOKEN \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --num-speakers 4

# Speaker range when count is unknown
speech-mine extract conference.wav output.csv \
  --hf-token YOUR_TOKEN \
  --min-speakers 2 \
  --max-speakers 8 \
  --compute-type float32

# Known language — skips auto-detection, slightly faster
speech-mine extract interview.wav output.csv \
  --hf-token YOUR_TOKEN \
  --language en \
  --compute-type float32
```

!!! warning
    Always use `--compute-type float32` when running on CPU. The default (`float16`) requires a GPU and will raise an error on CPU.

## Library

```python
from speech_mine.diarizer.processor import SpeechDiarizationProcessor

processor = SpeechDiarizationProcessor(
    hf_token="YOUR_TOKEN",
    num_speakers=2,
    whisper_model_size="large-v3",
    language="en",   # optional
    batch_size=16,   # reduce if OOM
)

# Full pipeline in one call
processor.process_audio_file("interview.mp3", "output.csv")
```

### Individual pipeline steps

```python
# Step 1: Transcribe
audio, result = processor.transcribe_audio("interview.mp3")

# Step 2: Forced alignment (word-level timestamps via wav2vec2)
result = processor.align(audio, result)

# Step 3: Speaker diarization + word assignment
result = processor.diarize(audio, result)

# Step 4: Save to CSV
processor.save_to_csv(result, "output.csv", {"audio_file": "interview.mp3", ...})
```

## Output

Two files are written:

- `output.csv` — segment and word-level transcript data ([see example](https://github.com/BeckettFrey/speech-mine/blob/main/examples/example_extract_output.csv))
- `output_metadata.json` — language, duration, speaker list, processing info ([see example](https://github.com/BeckettFrey/speech-mine/blob/main/examples/example_extract_metadata.json))

See [Output Format](output-format.md) for full column/field reference.

## Speaker Count Tips

Specifying `--num-speakers` when you know the exact count improves diarization accuracy.

| Parameter | When to use |
|-----------|-------------|
| `--num-speakers N` | You know exactly how many people speak |
| `--min-speakers N` | You know there are at least N speakers |
| `--max-speakers N` | You want to cap false speaker detection |

See [Model Options](models.md) for Whisper model and compute type guidance.
