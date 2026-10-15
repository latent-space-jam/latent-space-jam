# Parameters Guide

This page describes the key parameters for controlling layout-guided generation. We recommend starting with the defaults in `main.py` and iterating by inspecting the generated **attention maps** (saved to `output_path/attention_maps/`) alongside the output video.

> **General rule of thumb:** if the subject drifts from the mask, increase `num_refinements` or `lr_start`. If the video quality degrades, reduce them.

---

## Layout Definition

| Parameter | Description |
|-----------|-------------|
| `phrase_indices` | Maps each mask to word indices in the prompt (0-indexed). Determines which tokens the attention loss is computed over. For example, `[[1, 2]]` links a mask to the second and third words. |
| `bbox_paths` | Paths to binary mask videos (`.mp4`). Each mask defines where its corresponding subject should appear across frames. White = subject region, black = background. |

## Optimization

| Parameter | Description |
|-----------|-------------|
| `num_refinements` | How many gradient steps to run on the latent at each selected diffusion step. Higher values push the layout harder but may reduce visual quality. Typical range: **1 -- 5**. |
| `lr_start` / `lr_end` | Learning rate schedule (cosine annealing) for latent refinement. Controls how aggressively the latent is updated. Start low and increase if the layout is not followed. Typical value: **0.01**. |
| `refine_steps` | Which diffusion timesteps to apply refinement at. Early steps shape the global composition, later steps affect fine details. An empty list `[]` skips refinement; include all steps for maximum layout adherence. |

## Attention Extraction

| Parameter | Description |
|-----------|-------------|
| `layers_to_hook` | Which transformer layers to extract cross-attention from. Deeper layers capture higher-level semantics; shallower layers capture finer spatial detail. Defaults: **`[8, 9, 10]`** (LTX), **`range(0, 10)`** (Wan). |

## Pipeline

| Parameter | Description |
|-----------|-------------|
| `num_stages` | Number of progressive upscaling stages (**LTX only**). Stage 1 generates at low resolution with layout guidance; subsequent stages 2x-upscale with distilled denoising. Typical: **2 -- 3**. |
| `num_inference_steps` | Total number of diffusion denoising steps. More steps generally improve quality. Typical: **25 -- 50**. |
| `cfg_guidance_scale` / `guidance_scale` | Classifier-free guidance strength. Higher values increase prompt adherence at the cost of diversity. Typical: **5.0 -- 7.0**. |
| `seed` | Random seed for reproducibility. |

## Stage 2 (Wan only)

| Parameter | Description |
|-----------|-------------|
| `stage2_start_noise_step` | How many timesteps into the schedule to start denoising from for the refinement stage. Lower = more faithful to stage 1 motion, higher = more freedom to refine. Typical: **3 -- 7**. |
