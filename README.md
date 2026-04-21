
<p align="center">
  <img src="docs/hero.png" width="350" alt="speech-mine" />
</p>

<p align="center">
  <a href="https://pypi.org/project/speech-mine/"><img src="https://img.shields.io/pypi/v/speech-mine" alt="PyPI"></a>
  <a href="https://beckettfrey.github.io/speech-mine/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
</p>

<p align="center">
A Python toolkit for speaker-diarized transcription and transcript analysis. Built on <a href="https://github.com/m-bain/whisperX">WhisperX</a>; extract word-level, forced-aligned, speaker-labeled CSVs from audio, then search, format, and chunk them.
</p>

## Modules

| Module | Description | Docs |
|--------|-------------|------|
| `extract` | Transcribe audio with speaker diarization | [→](docs/extract.md) |
| `format` | Format CSV transcripts into readable scripts | [→](docs/format.md) |
| `chunk` | Split audio into segments via YAML config | [→](docs/chunk.md) |
| `search` | Fuzzy search transcripts by word or phrase | [→](docs/search.md) |


## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/beckettfrey/speech-mine
cd speech-mine
uv sync
```

See [docs/installation.md](docs/installation.md) for library dependency setup and HuggingFace token configuration.

> [!Warning]
> **Never pass `HF_TOKEN` as an argument to the `extract_audio` MCP tool** — that routes the secret through the LLM's conversation context where it can be logged, cached, or echoed back. Use your MCP client's native environment-injection so the token reaches the server subprocess directly. The server reads `HF_TOKEN` from its own environment automatically; the LLM should never see it.
>
> **Claude Code — PyPI install (recommended):**
> ```bash
> claude mcp add speech-mine -e HF_TOKEN=hf_xxx -- uvx --from speech-mine speech-mine-mcp
> ```
>
> **Claude Code — editing `.mcp.json` or `~/.claude.json` directly:**
> ```json
> {
>   "mcpServers": {
>     "speech-mine": {
>       "command": "uvx",
>       "args": ["--from", "speech-mine", "speech-mine-mcp"],
>       "env": { "HF_TOKEN": "hf_xxx" }
>     }
>   }
> }
> ```
>
> **Other MCP clients** (Claude Desktop, etc.): use the equivalent environment/secret field on the server entry.
>
> The `.env` file in this repo is strictly for **local development** (test suite, CLI runs) — never commit it, paste it into a chat, or forward its contents as tool arguments.

## Quick Start

> [!Note]
> speech-mine is flexible and adapts to your use case. The commands below show a generalized example workflow. For more granular control, use the Python API directly.

```bash
# 1. (Optional) Chunk a long recording into segments
uv run speech-mine chunk recording.wav chunks.yaml chunks/

# 2. Extract a transcript
uv run speech-mine extract interview.mp3 output.csv \
  --hf-token YOUR_TOKEN \
  --num-speakers 2 \
  --compute-type float32

# 3. Format into a readable script
uv run speech-mine format output.csv script.txt

# 4. Search it
uv run speech-mine search "topic of interest" output.csv --pretty

# 5. (Optional) Chunk the recording again around segments of interest
uv run speech-mine chunk recording.wav segments.yaml clips/
```

## MCP Server

speech-mine includes an [MCP](https://modelcontextprotocol.io) server that exposes all tools to Claude Code and other MCP clients.

**Install globally (no clone needed):**

```bash
claude mcp add speech-mine -e HF_TOKEN=hf_xxx -- uvx --from speech-mine speech-mine-mcp
```

This pulls the latest published version from PyPI via `uvx` and injects `HF_TOKEN` into the server's environment so `extract_audio` can run without routing the token through the LLM. Restart Claude Code after running it — `search_transcript`, `extract_audio`, `chunk_audio`, and the other tools will be available.

> [!Tip]
> Omit `-e HF_TOKEN=...` if you only intend to use the read-only tools (`search_transcript`, `get_transcript_stats`, `read_transcript`, `format_transcript`, `chunk_audio`). The token is only required for `extract_audio`.

**If you cloned the repo:** the included `.mcp.json` configures the server automatically when you open the project in Claude Code. Add an `env` block with your token to enable `extract_audio`:

```json
{
  "mcpServers": {
    "speech-mine": {
      "command": "uvx",
      "args": ["--from", "speech-mine", "speech-mine-mcp"],
      "env": { "HF_TOKEN": "hf_xxx" }
    }
  }
}
```

See the security callout under [Installation](#installation) for why tokens should never be passed as tool arguments.

## Documentation

```bash
# Serve docs locally
uv run mkdocs serve
```

Or browse the `docs/` folder directly.

## License

MIT

