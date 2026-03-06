# Output Format Reference

The `extract` command produces two output files.

## CSV (`output.csv`)

Contains interleaved segment-level and word-level rows. See [example](https://github.com/your-org/speech-mine/blob/main/examples/example_extract_output.csv).

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

Segment rows and their corresponding word rows are interleaved — each segment row is immediately followed by its word rows.

## Metadata (`output_metadata.json`)

See [example](https://github.com/your-org/speech-mine/blob/main/examples/example_extract_metadata.json).

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

| Field | Description |
|-------|-------------|
| `audio_file` | Path to the input audio file |
| `language` | Detected language code (e.g. `"en"`) |
| `language_probability` | Confidence of language detection (0–1) |
| `duration` | Total audio duration in seconds |
| `total_segments` | Number of speaker segments |
| `total_words` | Total word count across all segments |
| `speakers` | List of detected speaker IDs |
| `processing_timestamp` | When the file was processed |
