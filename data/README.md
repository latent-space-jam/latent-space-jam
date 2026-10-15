# Layout Data

This directory contains the spatio-temporal layout masks used in the paper. Each subdirectory represents a category of layouts, with a `layouts.json` file defining the prompts, masks, and phrase-to-word mappings.

## Structure

```
data/
  dynamic/          Single-object dynamic trajectories (e.g., left-to-right, slalom, curves)
  multi/            Multi-object compositions (e.g., two fish circling, skier + trees)
  static/           Static layouts derived from DAVIS video segmentation masks
```

Each category contains:
- **`layouts.json`** -- Array of layout definitions (see format below).
- **`masks/`** -- 81-frame binary mask videos.
- **`masks_121/`** -- 121-frame binary mask videos (same trajectories, longer duration).

## Layout Format

Each entry in `layouts.json` has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Short identifier for the layout. |
| `prompt` | string | Text prompt for generation. |
| `negative_prompt` | string | Negative prompt to suppress unwanted content. |
| `masks` | list[string] | Relative paths to mask videos, one per subject. |
| `mask-word-pairs` | list[list[int]] | Word indices (0-indexed) in the prompt for each mask. |
| `mask-labels` | list[string] | Human-readable label for each subject (used for evaluation). |
| `height` / `width` | int | Target generation resolution. |

### Example

```json
{
  "title": "cow-tl-to-br",
  "prompt": "A cow walking on a grassy field",
  "negative_prompt": "blurry, low resolution, ...",
  "masks": ["masks/tl-to-br.mp4"],
  "mask-word-pairs": [[1]],
  "mask-labels": ["cow"],
  "height": 480,
  "width": 720
}
```

Here, `mask-word-pairs: [[1]]` links the single mask to word index 1 ("cow") in the prompt.

## Mask Format

Mask videos are binary: **white (255) = subject region**, **black (0) = background**. They are standard `.mp4` files and can be created with any video editor or programmatically.
