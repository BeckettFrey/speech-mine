# Model Options

## Whisper Models

Available via the `--model` flag on `extract`.

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐⭐ | Quick tests |
| `base` | ⚡⚡⚡⚡ | ⭐⭐⭐ | Fast with decent quality |
| `small` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good balance |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐ | Higher accuracy |
| `large-v3` | ⚡ | ⭐⭐⭐⭐⭐ | Best accuracy (default) |
| `turbo` | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Fast + accurate |

## Device

| Value | Description |
|-------|-------------|
| `auto` | Detect best available device (default) |
| `cpu` | Force CPU |
| `cuda` | Force NVIDIA GPU |

## Compute Type

| Value | Requires | Speed | Notes |
|-------|----------|-------|-------|
| `float32` | CPU or GPU | Slower | Use on CPU |
| `float16` | GPU only | Faster | Default — will error on CPU |
| `int8` | CPU or GPU | Fastest | Slight accuracy tradeoff |

!!! warning
    `float16` is the default but requires a GPU. Use `--compute-type float32` on CPU systems.
