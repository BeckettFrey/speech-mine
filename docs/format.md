# format — Script Formatting

Converts the CSV output from `extract` into a human-readable, movie-style script.

## CLI

```bash
uv run speech-mine format <input.csv> <output.txt> [options]
```

### Options

| Flag | Description |
|------|-------------|
| `--speakers FILE` | JSON file mapping `SPEAKER_00` → custom name |
| `--create-template` | Generate a speaker names template JSON from the CSV |

### Examples

```bash
# Basic formatting
uv run speech-mine format output.csv script.txt

# Generate a speaker names template
uv run speech-mine format output.csv script.txt --create-template

# Format with custom speaker names
uv run speech-mine format output.csv script.txt --speakers output_speaker_names.json
```

### Custom speaker names workflow

```bash
# 1. Generate template — creates output_speaker_names.json
uv run speech-mine format output.csv script.txt --create-template

# 2. Edit the template
# {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}

# 3. Format with names applied
uv run speech-mine format output.csv final_script.txt --speakers output_speaker_names.json
```

## Library

```python
from speech_mine.diarizer.formatter import ScriptFormatter

# Basic formatting
formatter = ScriptFormatter()
formatter.format_script("output.csv", "script.txt")

# With custom speaker names
formatter = ScriptFormatter(custom_speakers={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
formatter.format_script("output.csv", "script.txt")

# Generate a speaker names template from a CSV
template_path = ScriptFormatter.create_custom_speakers_template("output.csv")

# Load speaker names from a JSON file
speakers = ScriptFormatter.load_custom_speakers("names.json")
formatter = ScriptFormatter(custom_speakers=speakers)
```

## Output format

See [examples/example_format_output.txt](https://github.com/your-org/speech-mine/blob/main/examples/example_format_output.txt) for a full sample.

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
