"""Latent Space Jamming -- LTX Video Generation.

Layout-guided video generation using the LTX multi-stage pipeline.
Stage 1 generates a low-resolution video with attention-based layout guidance,
and subsequent stages progressively upscale using distilled denoising.

Example (generates a cow walking diagonally across the frame):

    python -m src.ltx.main

Requirements:
    - LTX-2 model weights under ``models/`` (see README for download links).
    - Mask videos under ``data/`` (included in this repository).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.media_io import encode_video
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusers.utils import load_video

from src.evaluation import (
    calculate_ap50,
    calculate_ap50_over_min,
    calculate_centroid_distance,
    calculate_clip_similarity,
    calculate_intersection_over_min,
    calculate_iou,
    create_debug_video,
    detect_owlvit,
    load_clip_model,
    load_masks_from_video,
    save_results_to_json,
)
from src.ltx.pipeline import DiffusersCallbackAdapter, LayoutGuidedLTXMultiStagePipeline, load_multi_stage_pipeline
from src.optimization import (
    AttentionStore,
    GuidingPhrase,
    LatentRefiner,
    LayoutKLLoss,
    OptimizerType,
    get_token_indices_for_phrase_indices,
)
from src.video_utils import (
    SaveAttnMapsCallbackLTX,
    downsample_mask_to_latent_resolution,
    load_video_as_tensor,
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")


def load_ltx_pipeline(
    checkpoint_path, distilled_lora_path, spatial_upsampler_path, gemma_root_path, device="cuda", quantization: QuantizationPolicy | None = None
):
    """Load the LTX multi-stage pipeline."""
    logger.info("Loading official LTX multi-stage pipeline...")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Initializing pipeline...")
    pipe = load_multi_stage_pipeline(
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_root_path,
        device=device,
        quantization=quantization,
        distilled_lora_path=distilled_lora_path,
        spatial_upsampler_path=spatial_upsampler_path,
    )
    return pipe


def setup_optimization(
    bbox_paths,
    phrase_indices,
    prompt,
    tokenizer_path,
    height,
    width,
    num_frames,
    num_refinements,
    lr_start,
    lr_end,
    num_stages=2,
):
    """Setup optimization components including loss function and refiner."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    indices = get_token_indices_for_phrase_indices(phrase_indices, prompt, tokenizer)
    del tokenizer
    torch.cuda.empty_cache()

    words = prompt.split()
    guiding_phrases = [GuidingPhrase(" ".join(words[i] for i in group), idx) for group, idx in zip(phrase_indices, indices)]

    logger.info(f"Guiding phrases: {guiding_phrases}")

    bboxes = torch.cat([load_video_as_tensor(p, height=height, width=width, mode="L")[:, :, :num_frames] for p in bbox_paths])
    vae_spatial_scale_factor = 2 ** (num_stages - 1) * 32
    bboxes = downsample_mask_to_latent_resolution(bboxes, 8, vae_spatial_scale_factor)
    bboxes[bboxes > 0.5] = 1.0
    bboxes[bboxes <= 0.5] = 0.0
    logger.info(f"Bboxes shape: {bboxes.shape}")

    loss_fn = LayoutKLLoss(bboxes=bboxes, indices=indices, eps=1e-8)
    refiner = LatentRefiner(
        loss_fn=loss_fn,
        num_refinements=num_refinements,
        lr_start=lr_start,
        lr_end=lr_end,
        betas=(0.9, 0.999),
        optimizer_type=OptimizerType.ADAMW8BIT,
    )
    return guiding_phrases, refiner


def generate_video(
    pipe: LayoutGuidedLTXMultiStagePipeline,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    seed: int,
    guiding_phrases: list[GuidingPhrase],
    refiner: LatentRefiner,
    output_path: str,
    num_stages: int = 3,
    layers_to_hook=None,
    refine_steps=None,
    phrase_indices=None,
    tiling_config=None,
    enhance_prompt=False,
):
    """Generate video with layout guidance using multi-stage pipeline."""
    logger.info(f"Generating video with layout guidance ({num_stages} stages)...")
    store = AttentionStore()

    downscale = 2 ** (num_stages - 1)
    ltx_callback = SaveAttnMapsCallbackLTX(
        store=store,
        guiding_phrases=guiding_phrases,
        video_shape=(num_frames, height // downscale, width // downscale),
        output_path=str(Path(output_path) / "attention_maps"),
        fps=3.0,
    )

    callback = DiffusersCallbackAdapter(ltx_callback)

    if layers_to_hook is None:
        layers_to_hook = [10, 11, 12]
    if refine_steps is None:
        refine_steps = []

    stage_results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        seed=seed,
        num_stages=num_stages,
        images=[],
        tiling_config=tiling_config,
        enhance_prompt=enhance_prompt,
        attention_store=store,
        layers_to_hook=layers_to_hook,
        refiner=refiner,
        refine_steps=refine_steps,
        callback=callback,
        phrase_indices=phrase_indices,
    )

    return stage_results


@torch.no_grad()
def save_videos(stage_results, output_path, frame_rate, video_chunks_number):
    """Save generated videos to disk for all stages."""
    paths = []
    for i, (video, audio) in enumerate(stage_results, 1):
        stage_path = Path(output_path) / f"stage_{i}.mp4"
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=stage_path,
            video_chunks_number=video_chunks_number,
        )
        logger.success(f"Stage {i} Video saved to: {stage_path}")
        paths.append(stage_path)

    return paths


def evaluate_video(
    video_path: str,
    gt_mask_paths: list[str],
    subjects: list[str],
    prompt: str,
    stage_name: str,
    eval_output_dir: Path,
    frame_rate: float,
    clip_model,
    clip_processor,
) -> dict:
    logger.info(f"Evaluating {stage_name}...")
    frames = load_video(video_path)
    w, h = frames[0].size
    frame_size = float(np.linalg.norm([h, w]))

    metric_keys = ["iou", "ap50", "centroid_dist", "intersection_over_min", "ap50_over_min"]
    per_subject_results = []

    for subject, mask_path in zip(subjects, gt_mask_paths):
        logger.info(f"  Evaluating subject '{subject}' with mask {mask_path}...")
        gt_masks = load_masks_from_video(mask_path, h, w, len(frames))
        pred_masks = detect_owlvit(frames, subject)

        debug_video_path = eval_output_dir / f"{stage_name}_{subject}_debug.mp4"
        create_debug_video(frames, pred_masks, gt_masks, str(debug_video_path), fps=int(frame_rate))

        subject_result = {
            "subject": subject,
            "gt_mask_path": mask_path,
            "iou": float(calculate_iou(pred_masks, gt_masks)),
            "ap50": float(calculate_ap50(pred_masks, gt_masks)),
            "centroid_dist": float(calculate_centroid_distance(pred_masks, gt_masks, normalized=False) / frame_size),
            "intersection_over_min": float(calculate_intersection_over_min(pred_masks, gt_masks)),
            "ap50_over_min": float(calculate_ap50_over_min(pred_masks, gt_masks)),
        }
        logger.info(f"    IoU={subject_result['iou']:.4f} AP50={subject_result['ap50']:.4f} IoM={subject_result['intersection_over_min']:.4f}")
        per_subject_results.append(subject_result)

    clip_sim = float(calculate_clip_similarity(frames, prompt, clip_model, clip_processor))

    avg_metrics = {}
    for key in metric_keys:
        avg_metrics[key] = float(np.mean([r[key] for r in per_subject_results]))

    results = {
        "video_path": str(video_path),
        "gt_mask_paths": gt_mask_paths,
        "subjects": subjects,
        "prompt": prompt,
        "stage": stage_name,
        "frames": len(frames),
        "resolution": f"{w}x{h}",
        **avg_metrics,
        "clip_sim": clip_sim,
        "per_subject": per_subject_results,
    }

    save_results_to_json(results, eval_output_dir / f"{stage_name}_results.json")
    logger.success(f"Saved results to {eval_output_dir / f'{stage_name}_results.json'}")
    return results


def run_evaluation(
    stage_paths: list[Path],
    gt_mask_paths: list[str],
    subjects: list[str],
    prompt: str,
    output_path: str,
    frame_rate: float,
):
    logger.info("=" * 80)
    logger.info("Starting Evaluation")
    logger.info("=" * 80)

    eval_output_dir = Path(output_path) / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CLIP model...")
    clip_model, clip_processor = load_clip_model()

    metric_keys = ["iou", "ap50", "centroid_dist", "intersection_over_min", "ap50_over_min", "clip_sim"]
    all_stage_results = {}

    for i, path in enumerate(stage_paths, 1):
        stage_name = f"stage{i}"
        logger.info(f"Evaluating Stage {i}...")
        all_stage_results[stage_name] = evaluate_video(
            str(path),
            gt_mask_paths,
            subjects,
            prompt,
            stage_name,
            eval_output_dir,
            frame_rate,
            clip_model,
            clip_processor,
        )

    first_results = all_stage_results["stage1"]
    last_results = all_stage_results[f"stage{len(stage_paths)}"]

    improvement = {}
    for key in metric_keys:
        improvement[key] = last_results[key] - first_results[key]

    summary = {
        **all_stage_results,
        "improvement_first_to_last": improvement,
    }
    save_results_to_json(summary, eval_output_dir / "summary.json")

    logger.success("=" * 80)
    logger.success("EVALUATION SUMMARY")
    logger.success("=" * 80)
    logger.info(f"{'Stage':<12} {'IoU':<10} {'AP50':<10} {'CD':<10} {'IoM':<10} {'AP50-IoM':<12} {'CLIP':<10}")
    logger.info("=" * 80)
    for i in range(1, len(stage_paths) + 1):
        r = all_stage_results[f"stage{i}"]
        logger.info(
            f"{'Stage ' + str(i):<12} {r['iou']:<10.4f} {r['ap50']:<10.4f} "
            f"{r['centroid_dist']:<10.4f} {r['intersection_over_min']:<10.4f} "
            f"{r['ap50_over_min']:<12.4f} {r['clip_sim']:<10.4f}"
        )
    logger.info("-" * 80)
    logger.info(
        f"{'Improvement':<12} {improvement['iou']:<+10.4f} "
        f"{improvement['ap50']:<+10.4f} "
        f"{improvement['centroid_dist']:<+10.4f} "
        f"{improvement['intersection_over_min']:<+10.4f} "
        f"{improvement['ap50_over_min']:<+12.4f} "
        f"{improvement['clip_sim']:<+10.4f}"
    )
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {eval_output_dir}")
    logger.success("Evaluation Complete!")

    return summary


def main():
    # --- Model paths (download LTX-2 weights into models/) ---
    checkpoint_path = os.path.join(MODELS_PATH, "ltx-2-19b-dev.safetensors")
    gemma_root_path = os.path.join(MODELS_PATH, "gemma-3-12b-it-qat-q4_0-unquantized")
    distilled_lora_path = os.path.join(MODELS_PATH, "ltx-2-19b-distilled-lora-384.safetensors")
    spatial_upsampler_path = os.path.join(MODELS_PATH, "ltx-2-spatial-upscaler-x2-1.0.safetensors")

    # --- Generation config ---
    prompt = "A cow walking on a grassy field, following a diagonal path across the frame from the top left corner to the bottom right corner."
    negative_prompt = DEFAULT_NEGATIVE_PROMPT
    phrase_indices = [[1]]
    seed = 37
    output_path = os.path.join(OUTPUTS_PATH, "ltx", str(seed), "cow-tl-to-br")

    num_stages = 2
    height = 640
    width = 640
    num_frames = 81
    frame_rate = 24.0
    num_inference_steps = 40
    cfg_guidance_scale = 7.0

    # --- Layout guidance ---
    bbox_paths = [
        os.path.join(DATA_PATH, "dynamic", "masks", "tl-to-br.mp4"),
    ]

    lr_start = 0.01
    lr_end = 0.01
    num_refinements = 1
    layers_to_hook = [8, 9, 10]
    refine_steps = list(range(num_inference_steps))

    logger.info(f"Config: {width}x{height}, {num_stages} stages, {num_frames} frames, seed={seed}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    pipe = load_ltx_pipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora_path=distilled_lora_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root_path=gemma_root_path,
        device="cuda",
        quantization=None,
    )

    guiding_phrases, refiner = setup_optimization(
        bbox_paths=bbox_paths,
        phrase_indices=phrase_indices,
        prompt=prompt,
        tokenizer_path=gemma_root_path,
        height=height,
        width=width,
        num_frames=num_frames,
        num_refinements=num_refinements,
        lr_start=lr_start,
        lr_end=lr_end,
        num_stages=num_stages,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    stage_results = generate_video(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        seed=seed,
        guiding_phrases=guiding_phrases,
        refiner=refiner,
        output_path=output_path,
        num_stages=num_stages,
        layers_to_hook=layers_to_hook,
        refine_steps=refine_steps,
        phrase_indices=phrase_indices,
        tiling_config=tiling_config,
        enhance_prompt=False,
    )

    save_videos(
        stage_results=stage_results,
        output_path=output_path,
        frame_rate=frame_rate,
        video_chunks_number=video_chunks_number,
    )

    logger.success("Generation complete!")


if __name__ == "__main__":
    main()
