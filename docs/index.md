# speech-mine

Speech diarization and transcript analysis toolkit. Extract speaker-labeled transcripts from audio, format them into readable scripts, search them with fuzzy matching, and pre-process audio with chunking.

## Why not WhisperX?

WhisperX is a solid general-purpose transcription tool. If you need maximum word-level timestamp precision, its wav2vec2 forced alignment step delivers that. But forced alignment comes with real costs: a third model to download on top of Whisper and pyannote, language-specific alignment models that don't exist for every language, and an extra failure surface on noisy audio or heavy accents.

`speech-mine` skips forced alignment entirely, which also means less time per run — both use the same faster-whisper and pyannote models, so transcription and diarization are equivalent, but eliminating the wav2vec2 alignment pass over the full audio is a meaningful saving on longer recordings. It uses faster-whisper's native word timestamps — accurate enough for content analysis, scripting, and phrase search — and assigns speakers via segment-level overlap rather than per-word alignment. This means a simpler dependency tree, fewer failure modes, and consistent behavior across languages without hunting down alignment model support.

It also captures things WhisperX doesn't expose, like `overlap_duration` per segment for identifying cross-talk, and it treats `extract` as step one of a complete workflow rather than a standalone tool. The goal isn't phoneme-level precision — it's a reliable pipeline from raw audio to a searchable, human-readable transcript.

When used as a library, `speech-mine` also lets you step into the pipeline at any point. Each stage — transcription, diarization, alignment, and saving — is an individually callable method on the same processor object, returning the intermediate data so you can inspect, transform, or replace it before passing it to the next step. WhisperX exposes its stages as separate utility functions with inconsistent interfaces; there's no clean handoff point if you want to inject custom logic between diarization and speaker assignment. With `speech-mine`, that's just calling `align_transcription_with_speakers` with whatever segments or diarization output you provide.

More broadly, WhisperX is a low-level transcription tool designed for one-shot use — you give it audio, you get a timestamped transcript. `speech-mine` is higher-level, designed for iterative pipelines where the transcript is an artifact you work with repeatedly: re-formatting it with corrected speaker names, re-searching it as your queries evolve, re-chunking the audio when your segmentation changes. The modules are composable by design, and the CSV/JSON output format is stable and predictable so downstream scripts don't break between runs.

If you need sub-word timestamps for forced subtitle sync or phonetic research, use WhisperX. If you need a transcript you can read, search, and work with across an iterative workflow, use `speech-mine`.

## Installation

```bash
pip install speech-mine
```

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
