"""Latent Space Jamming -- Wan Video Generation.

Layout-guided video generation using the Wan 2.2 pipeline.
Stage 1 generates a low-resolution video with attention-based layout guidance,
and Stage 2 refines motion at higher resolution via SDEdit.

Example (generates a golden retriever walking in a slalom pattern):

    python -m src.wan.main

Requirements:
    - Wan 2.2 model weights (auto-downloaded from HuggingFace).
    - Mask videos under ``data/`` (included in this repository).
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video, load_video
from diffusers.utils.torch_utils import randn_tensor
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.optimization import (
    AttentionStore,
    GuidingPhrase,
    LatentRefiner,
    LayoutKLLoss,
    get_token_indices_for_phrase_indices,
)
from src.video_utils import (
    SaveAttnMapsCallbackWAN,
    downsample_mask_to_latent_resolution,
    load_video_as_tensor,
    video_to_tensor,
)
from src.wan.attn_processor import (
    register_attention_hook,
    unregister_attention_hook,
)
from src.wan.pipeline import LayoutGuidedWanPipeline, encode_video_to_latents, load_pipeline

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")


def setup_optimization(
    bbox_paths,
    phrase_indices,
    prompt,
    tokenizer,
    height,
    width,
    num_frames,
    num_refinements,
    lr_start,
    lr_end,
    vae_scale_factor_temporal=4,
    vae_scale_factor_spatial=8,
):
    """Setup optimization components including loss function and refiner."""
    indices = get_token_indices_for_phrase_indices(phrase_indices, prompt, tokenizer)

    words = prompt.split()
    guiding_phrases = [GuidingPhrase(phrase=" ".join(words[i] for i in group), indices=idx) for group, idx in zip(phrase_indices, indices)]
    logger.info(f"Guiding phrases: {guiding_phrases}")

    bboxes = torch.cat([load_video_as_tensor(p, height=height, width=width, mode="L")[:, :, :num_frames] for p in bbox_paths])
    bboxes = downsample_mask_to_latent_resolution(bboxes, vae_scale_factor_temporal, vae_scale_factor_spatial * 2)
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
    )
    return guiding_phrases, refiner


def generate_video(
    pipe: LayoutGuidedWanPipeline,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_scale_2: float,
    seed: int,
    guiding_phrases: list[GuidingPhrase],
    refiner: LatentRefiner,
    output_path: str,
    layers_to_hook=None,
    refine_steps=None,
):
    """Generate video with layout guidance (stage 1)."""
    logger.info("Generating video with layout guidance...")

    if layers_to_hook is None:
        layers_to_hook = [10, 11, 12]
    if refine_steps is None:
        refine_steps = []

    store = AttentionStore()
    original_attn_processors = register_attention_hook(pipe, store, layers_to_hook=layers_to_hook)

    lat_shape = (
        num_frames,
        height // (pipe.vae_scale_factor_spatial * 2),
        width // (pipe.vae_scale_factor_spatial * 2),
    )
    patch_size = pipe.transformer.config["patch_size"]
    callback = SaveAttnMapsCallbackWAN(
        store=store,
        guiding_phrases=guiding_phrases,
        latent_shape=lat_shape,
        patch_size=patch_size,
        output_path=str(Path(output_path) / "attention_maps"),
        fps=3.0,
    )

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        num_inference_steps=num_inference_steps,
        generator=generator,
        callback_on_step_end=callback,
        store=store,
        output_type="pil",
        refine_steps=refine_steps,
        refiner=refiner,
    ).frames[0]

    unregister_attention_hook(pipe, original_attn_processors)

    return output


def stage2_motion_refinement(
    pipe: LayoutGuidedWanPipeline,
    prompt: str,
    negative_prompt: str,
    stage1_frames: list,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_scale_2: float,
    seed: int,
    start_noise_step: int,
):
    """Stage 2: Motion refinement using noised latents from stage 1."""
    logger.info("Stage 2: Motion refinement...")

    motion_signal = [f.resize((width, height)) for f in stage1_frames]
    motion_signal = video_to_tensor(motion_signal)
    motion_signal_latents = encode_video_to_latents(pipe, motion_signal)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    noise = randn_tensor(
        motion_signal_latents.shape,
        generator,
        device=pipe.device,
        dtype=motion_signal_latents.dtype,
    )

    pipe.scheduler.set_timesteps(num_inference_steps)
    t = pipe.scheduler.timesteps[start_noise_step]
    t = torch.as_tensor(t, device=pipe.device, dtype=torch.long).view(1)
    noised_motion_signal_latents = pipe.scheduler.add_noise(motion_signal_latents, noise, t.long())

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        num_inference_steps=num_inference_steps,
        generator=generator,
        latents=noised_motion_signal_latents,
        output_type="pil",
        start_noise_at=start_noise_step,
    ).frames[0]

    return output


@torch.no_grad()
def save_videos(stage1_output, stage2_output, output_path, fps):
    """Save generated videos to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stage1_path = output_path / "base.mp4"
    export_to_video(stage1_output, str(stage1_path), fps=fps)
    logger.success(f"Stage 1 video saved to: {stage1_path}")

    stage2_path = output_path / "final.mp4"
    export_to_video(stage2_output, str(stage2_path), fps=fps)
    logger.success(f"Stage 2 video saved to: {stage2_path}")

    return stage1_path, stage2_path


def evaluate_video(
    video_path: str,
    gt_mask_paths: list[str],
    subjects: list[str],
    prompt: str,
    stage_name: str,
    eval_output_dir: Path,
    fps: float,
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
        create_debug_video(frames, pred_masks, gt_masks, str(debug_video_path), fps=int(fps))

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
    stage1_path: Path,
    stage2_path: Path,
    gt_mask_paths: list[str],
    subjects: list[str],
    prompt: str,
    output_path: str,
    fps: float,
):
    logger.info("=" * 80)
    logger.info("Starting Evaluation")
    logger.info("=" * 80)

    eval_output_dir = Path(output_path) / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CLIP model...")
    clip_model, clip_processor = load_clip_model()

    logger.info("Evaluating Stage 1...")
    stage1_results = evaluate_video(
        str(stage1_path),
        gt_mask_paths,
        subjects,
        prompt,
        "stage1",
        eval_output_dir,
        fps,
        clip_model,
        clip_processor,
    )

    logger.info("Evaluating Stage 2...")
    stage2_results = evaluate_video(
        str(stage2_path),
        gt_mask_paths,
        subjects,
        prompt,
        "stage2",
        eval_output_dir,
        fps,
        clip_model,
        clip_processor,
    )

    summary = {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "improvement": {
            "iou": stage2_results["iou"] - stage1_results["iou"],
            "ap50": stage2_results["ap50"] - stage1_results["ap50"],
            "centroid_dist": stage2_results["centroid_dist"] - stage1_results["centroid_dist"],
            "intersection_over_min": stage2_results["intersection_over_min"] - stage1_results["intersection_over_min"],
            "ap50_over_min": stage2_results["ap50_over_min"] - stage1_results["ap50_over_min"],
            "clip_sim": stage2_results["clip_sim"] - stage1_results["clip_sim"],
        },
    }
    save_results_to_json(summary, eval_output_dir / "summary.json")

    logger.success("=" * 80)
    logger.success("EVALUATION SUMMARY")
    logger.success("=" * 80)
    logger.info(f"{'Stage':<12} {'IoU':<10} {'AP50':<10} {'CD':<10} {'IoM':<10} {'AP50-IoM':<12} {'CLIP':<10}")
    logger.info("=" * 80)
    logger.info(
        f"{'Stage 1':<12} {stage1_results['iou']:<10.4f} {stage1_results['ap50']:<10.4f} "
        f"{stage1_results['centroid_dist']:<10.4f} {stage1_results['intersection_over_min']:<10.4f} "
        f"{stage1_results['ap50_over_min']:<12.4f} {stage1_results['clip_sim']:<10.4f}"
    )
    logger.info(
        f"{'Stage 2':<12} {stage2_results['iou']:<10.4f} {stage2_results['ap50']:<10.4f} "
        f"{stage2_results['centroid_dist']:<10.4f} {stage2_results['intersection_over_min']:<10.4f} "
        f"{stage2_results['ap50_over_min']:<12.4f} {stage2_results['clip_sim']:<10.4f}"
    )
    logger.info("-" * 80)
    logger.info(
        f"{'Improvement':<12} {summary['improvement']['iou']:<+10.4f} "
        f"{summary['improvement']['ap50']:<+10.4f} "
        f"{summary['improvement']['centroid_dist']:<+10.4f} "
        f"{summary['improvement']['intersection_over_min']:<+10.4f} "
        f"{summary['improvement']['ap50_over_min']:<+12.4f} "
        f"{summary['improvement']['clip_sim']:<+10.4f}"
    )
    logger.info("=" * 80)

    return summary


def main():
    # --- Generation config ---
    prompt = "A golden retriever walks in a slalom pattern through a park, starting at the top right and ending at the bottom right."
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    phrase_indices = [[1, 2]]
    seed = 37
    output_path = os.path.join(OUTPUTS_PATH, "wan", str(seed), "golden-retriever-slalom-from-tr")

    num_frames = 81
    fps = 16
    guidance_scale = 5.0
    guidance_scale_2 = None

    # Stage 1: layout-guided generation (low resolution)
    stage1_height = 320
    stage1_width = 320
    stage1_num_inference_steps = 25

    # Stage 2: motion refinement (full resolution)
    stage2_height = 512
    stage2_width = 512
    stage2_num_inference_steps = 50
    stage2_start_noise_step = 3

    # --- Layout guidance ---
    bbox_paths = [
        os.path.join(DATA_PATH, "dynamic", "masks", "slalom-from-tr.mp4"),
    ]

    lr_start = 0.01
    lr_end = 0.01
    num_refinements = 3
    layers_to_hook = list(range(0, 10))
    refine_steps = list(range(stage1_num_inference_steps))

    logger.info(f"Config: {stage2_width}x{stage2_height}, {num_frames} frames, seed={seed}")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline()

    guiding_phrases, refiner = setup_optimization(
        bbox_paths=bbox_paths,
        phrase_indices=phrase_indices,
        prompt=prompt,
        tokenizer=pipe.tokenizer,
        height=stage1_height,
        width=stage1_width,
        num_frames=num_frames,
        num_refinements=num_refinements,
        lr_start=lr_start,
        lr_end=lr_end,
        vae_scale_factor_temporal=pipe.vae_scale_factor_temporal,
        vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
    )

    stage1_output = generate_video(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=stage1_height,
        width=stage1_width,
        num_frames=num_frames,
        num_inference_steps=stage1_num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        seed=seed,
        guiding_phrases=guiding_phrases,
        refiner=refiner,
        output_path=output_path,
        layers_to_hook=layers_to_hook,
        refine_steps=refine_steps,
    )

    stage2_output = stage2_motion_refinement(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        stage1_frames=stage1_output,
        height=stage2_height,
        width=stage2_width,
        num_frames=num_frames,
        num_inference_steps=stage2_num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        seed=seed,
        start_noise_step=stage2_start_noise_step,
    )

    save_videos(
        stage1_output=stage1_output,
        stage2_output=stage2_output,
        output_path=output_path,
        fps=fps,
    )

    logger.success("Generation complete!")


if __name__ == "__main__":
    main()
