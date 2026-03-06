# Installation

## As a CLI tool

Requires ffmpeg on your `PATH`.

```bash
pip install speech-mine
```

Or with [pipx](https://pipx.pypa.io/) to install into an isolated environment:

```bash
pipx install speech-mine
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install speech-mine
```

## As a library dependency

**pip:**
```bash
pip install speech-mine
```

**uv:**
```bash
uv add speech-mine
```

**pyproject.toml:**
```toml
dependencies = [
    "speech-mine",
]
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
