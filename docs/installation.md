# Installation

## As a CLI tool

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repository-url>
cd speech-mine
uv sync
```

## As a library dependency

**uv / pyproject.toml:**
```toml
dependencies = [
    "speech-mine @ git+https://github.com/your-org/speech-mine.git",
]
```
```bash
uv add git+https://github.com/your-org/speech-mine.git
```

**pip:**
```bash
pip install git+https://github.com/your-org/speech-mine.git
```

## HuggingFace Token

The `extract` module requires a HuggingFace token to download pyannote models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → New token (read permissions)
3. Accept the user agreement at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

Pass the token via `--hf-token YOUR_TOKEN` on every `extract` call.

## Requirements

- Python 3.11+
- ffmpeg installed and on `PATH`
- GPU recommended for faster processing (not required)
