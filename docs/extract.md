# extract — Transcription + Speaker Diarization

Transcribes audio and labels each segment with the speaker who said it. Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription and [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

**Supported audio formats:** `.wav`, `.mp3`, `.ogg`, `.flac`

## CLI

```bash
uv run speech-mine extract <audio> <output.csv> --hf-token TOKEN [options]
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
| `--verbose` | — | Enable verbose logging |

### Examples

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

# Speaker range when count is unknown
uv run speech-mine extract conference.wav output.csv \
  --hf-token YOUR_TOKEN \
  --min-speakers 2 \
  --max-speakers 8 \
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
)

# Full pipeline in one call
processor.process_audio_file("interview.mp3", "output.csv")
```

### Individual pipeline steps

```python
# Step 1: Transcribe
segments, info = processor.transcribe_audio("interview.mp3")

# Step 2: Diarize
diarization = processor.perform_speaker_diarization("interview.mp3")

# Step 3: Align
aligned = processor.align_transcription_with_speakers(segments, diarization)

# Step 4: Save
processor.save_to_csv(aligned, "output.csv", info)
```

## Output

Two files are written:

- `output.csv` — segment and word-level transcript data ([see example](https://github.com/your-org/speech-mine/blob/main/examples/example_extract_output.csv))
- `output_metadata.json` — language, duration, speaker list, processing info ([see example](https://github.com/your-org/speech-mine/blob/main/examples/example_extract_metadata.json))

See [Output Format](output-format.md) for full column/field reference.

## Speaker Count Tips

Specifying `--num-speakers` when you know the exact count improves diarization accuracy by 15–30%.

| Parameter | When to use |
|-----------|-------------|
| `--num-speakers N` | You know exactly how many people speak |
| `--min-speakers N` | You know there are at least N speakers |
| `--max-speakers N` | You want to cap false speaker detection |

See [Model Options](models.md) for whisper model and compute type guidance.
