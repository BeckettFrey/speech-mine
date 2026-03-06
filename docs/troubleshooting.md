# Troubleshooting

## `float16` error on CPU

```
Requested float16 compute type, but the target device or backend do not support efficient float16 computation.
```

**Fix:** Add `--compute-type float32`:

```bash
uv run speech-mine extract audio.wav out.csv \
  --hf-token TOKEN \
  --compute-type float32
```

Safest command for any system:

```bash
uv run speech-mine extract audio.wav out.csv \
  --hf-token TOKEN \
  --device cpu \
  --compute-type float32 \
  --model base
```

## pyannote model access error

```
Failed to load pyannote pipeline
```

**Fix:** Make sure you have:

1. A valid HuggingFace token with read permissions
2. Accepted the user agreement at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## No matches found in search

If `speech-mine search` returns no results, try lowering the similarity range:

```bash
uv run speech-mine search "your query" output.csv --similarity-range 0.5 1.0
```

The default range is `0.0 1.0` (all matches), but if no results appear it may be that no windows score above 0.0 — try the query with simpler wording.
