import os
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_core.utils import to_velocity
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    generate_enhanced_prompt,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
    modality_from_latent_state,
    post_process_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import load_video_conditioning
from ltx_pipelines.utils.types import DenoisingFunc
from tqdm import tqdm

from src.ltx.attn_processor import register_attention_hook, unregister_attention_hook
from src.optimization import AttentionStore, LatentRefiner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

STAGE_2_DISTILLED_SIGMA_VALUES = [0.959375, 0.725, 0.421875, 0.0]


class PipelineCallback(Protocol):
    def __call__(self, stepper, step_index: int, timestep: float, callback_kwargs: dict) -> None: ...


class DiffusersCallbackAdapter(PipelineCallback):
    def __init__(self, diffusers_callback):
        self.callback = diffusers_callback

    def __call__(self, stepper, step_index: int, timestep: float, callback_kwargs: dict):
        return self.callback.callback_fn(None, step_index, timestep, callback_kwargs)


class EulerDiffusionStepWithCallback(DiffusionStepProtocol):
    def __init__(self, callback: PipelineCallback = None):
        self.callback = callback

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)
        denoised = (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)

        if self.callback is not None:
            callback_kwargs = {"x0": sample, "denoised": denoised}
            self.callback(self, step_index, sigma.item(), callback_kwargs)

        return denoised


class RefineFunc(Protocol):
    def __call__(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: torch.Tensor,
        step_index: int,
        progress_bar: tqdm | None = None,
    ) -> tuple[LatentState, LatentState]: ...


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    refine_fn: RefineFunc | None = None,
    refine_steps: list[int] | None = None,
) -> tuple[LatentState, LatentState]:
    pbar = tqdm(sigmas[:-1], desc="Denoising")

    for step_idx, _ in enumerate(pbar):
        if refine_fn is not None and refine_steps is not None and step_idx in refine_steps:
            video_state, audio_state = refine_fn(video_state, audio_state, sigmas, step_idx, pbar)

        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

    return video_state, audio_state


def validate_refinement_inputs(
    refine_steps: list[int],
    refiner: LatentRefiner | None,
    attention_store: AttentionStore | None,
    layers_to_hook: list[int],
) -> None:
    if refine_steps:
        assert refiner is not None, "refiner must be provided when refine_steps is specified"
        assert attention_store is not None, "attention_store must be provided when refine_steps is specified"

    if refiner is not None:
        assert attention_store is not None, "attention_store must be provided when refiner is specified"

    if attention_store is not None:
        assert layers_to_hook, "layers_to_hook must be provided when attention_store is specified"


class LayoutGuidedLTXMultiStagePipeline(TI2VidTwoStagesPipeline):
    """Multi-stage pipeline supporting an arbitrary number of stages.

    Stage 1:    Low-resolution generation with layout guidance and optional refinement.
                Resolution = (width / 2^(N-1)) x (height / 2^(N-1))
    Stages 2..N: Progressive 2x spatial upscaling with distilled denoising.
                Stage k resolution = (width / 2^(N-k)) x (height / 2^(N-k))

    For num_stages=2 this is equivalent to the original two-stage pipeline.
    For num_stages=3 with a 1280x720 target:
        Stage 1: 320x180  ->  Stage 2: 640x360  ->  Stage 3: 1280x720
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        num_stages: int = 2,
        images: list[tuple[str, int, float]] = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        attention_store: AttentionStore | None = None,
        layers_to_hook: list[int] = None,
        refiner: LatentRefiner | None = None,
        refine_steps: list[int] | None = None,
        callback=None,
        phrase_indices: list[list[int]] = None,
        source_video_path: str | None = None,
        denoise_from: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run the multi-stage pipeline.

        Args:
            num_stages: Number of pipeline stages (>= 2). Stage 1 generates at
                the lowest resolution; stages 2..N each 2x-upscale and denoise.
            source_video_path: Path to a source video for SDEdit-style editing.
                The video is encoded to latents and partially noised.
            denoise_from: Step index to start denoising from (SDEdit).
                Trims the sigma schedule so denoising begins at sigmas[denoise_from].
                Only used when source_video_path is provided.

        Returns:
            List of (decoded_video, decoded_audio) tuples, one per stage.
        """
        assert num_stages >= 2, f"num_stages must be >= 2, got {num_stages}"

        images = images or []
        layers_to_hook = layers_to_hook or []
        refine_steps = refine_steps or []

        validate_refinement_inputs(refine_steps, refiner, attention_store, layers_to_hook)
        assert_resolution(height=height, width=width, is_two_stage=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stage_1_stepper = EulerDiffusionStepWithCallback(callback=callback) if callback else EulerDiffusionStep()

        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        # --- Text encoding ---
        text_encoder = self.stage_1_model_ledger.text_encoder()
        if enhance_prompt:
            if refiner is not None:
                raise ValueError("Enhanced prompt is forbidden when refiner is provided")

            prompt = generate_enhanced_prompt(
                text_encoder,
                prompt,
                images[0][0] if len(images) > 0 else None,
                seed=seed,
            )

        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()

        # ==================================================================
        # Stage 1: Low-resolution generation with layout guidance
        # ==================================================================
        downscale_factor = 2 ** (num_stages - 1)
        stage_1_width = width // downscale_factor
        stage_1_height = height // downscale_factor

        print(f"[Stage 1/{num_stages}] Generating at {stage_1_width}x{stage_1_height}")

        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()

        if refiner is not None:
            transformer.velocity_model.set_gradient_checkpointing(True)

        original_attentions = {}
        if attention_store is not None and layers_to_hook:
            original_attentions = register_attention_hook(transformer, attention_store, layers_to_hook)

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        initial_video_latent = None
        stage_1_sigmas = sigmas
        if source_video_path is not None:
            pixel_video = load_video_conditioning(
                video_path=source_video_path,
                height=stage_1_height,
                width=stage_1_width,
                frame_cap=num_frames,
                dtype=dtype,
                device=self.device,
            )
            initial_video_latent = video_encoder(pixel_video)
            if denoise_from is not None:
                stage_1_sigmas = sigmas[denoise_from:]

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper,
        ) -> tuple[LatentState, LatentState]:
            refine_fn = None
            if refiner is not None and attention_store is not None:

                def refine_latents(
                    video_state: LatentState,
                    audio_state: LatentState,
                    sigmas: torch.Tensor,
                    step_index: int,
                    progress_bar: "tqdm.tqdm | None" = None,
                ) -> tuple[LatentState, LatentState]:
                    sigma = sigmas[step_index]

                    def forward_fn(video_latent: torch.Tensor) -> None:
                        refined_video_state = replace(video_state, latent=video_latent)
                        video_modality = modality_from_latent_state(refined_video_state, v_context_p, sigma)
                        audio_modality = modality_from_latent_state(audio_state, a_context_p, sigma)
                        _ = transformer(video=video_modality, audio=audio_modality, perturbations=None)

                    refined_latent = refiner.optimize(
                        latents=video_state.latent,
                        forward_fn=forward_fn,
                        store=attention_store,
                        progress_bar=progress_bar,
                    )

                    refined_video_state = replace(video_state, latent=refined_latent)
                    return refined_video_state, audio_state

                refine_fn = refine_latents

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,
                ),
                refine_fn=refine_fn,
                refine_steps=refine_steps,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=stage_1_width,
            height=stage_1_height,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        denoise_kwargs = {}
        if initial_video_latent is not None:
            denoise_kwargs["initial_video_latent"] = initial_video_latent
            denoise_kwargs["noise_scale"] = stage_1_sigmas[0].item()

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stage_1_stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            **denoise_kwargs,
        )

        torch.cuda.synchronize()
        if original_attentions:
            unregister_attention_hook(transformer, original_attentions)
        del transformer
        cleanup_memory()

        # Decode stage 1
        decoded_video_s1 = vae_decode_video(
            video_state.latent,
            self.stage_1_model_ledger.video_decoder(),
            tiling_config,
            generator=generator,
        )
        decoded_audio_s1 = vae_decode_audio(
            audio_state.latent,
            self.stage_1_model_ledger.audio_decoder(),
            self.stage_1_model_ledger.vocoder(),
        )
        stage_results = [(decoded_video_s1, decoded_audio_s1)]

        # ==================================================================
        # Stages 2..N: Progressive 2x upscaling with distilled denoising
        # ==================================================================
        upsampler = self.stage_2_model_ledger.spatial_upsampler()
        stage_2_transformer = self.stage_2_model_ledger.transformer()
        stage_2_stepper = EulerDiffusionStep()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        # Cache decoders so they aren't reloaded every iteration
        stage_2_video_decoder = self.stage_2_model_ledger.video_decoder()
        stage_2_audio_decoder = self.stage_2_model_ledger.audio_decoder()
        stage_2_vocoder = self.stage_2_model_ledger.vocoder()

        def upscale_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=stage_2_transformer,
                ),
            )

        for stage_idx in range(2, num_stages + 1):
            remaining_scale = 2 ** (num_stages - stage_idx)
            stage_width = width // remaining_scale
            stage_height = height // remaining_scale

            print(f"[Stage {stage_idx}/{num_stages}] Upscaling to {stage_width}x{stage_height}")

            # 2x spatial upsample of the video latent
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=upsampler,
            )

            torch.cuda.synchronize()
            cleanup_memory()

            stage_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=stage_width,
                height=stage_height,
                fps=frame_rate,
            )
            stage_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_height,
                width=stage_width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_output_shape,
                conditionings=stage_conditionings,
                noiser=noiser,
                sigmas=distilled_sigmas,
                stepper=stage_2_stepper,
                denoising_loop_fn=upscale_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                noise_scale=distilled_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent,
            )

            torch.cuda.synchronize()
            cleanup_memory()

            # Decode this stage
            decoded_video = vae_decode_video(
                video_state.latent,
                stage_2_video_decoder,
                tiling_config,
                generator,
            )
            decoded_audio = vae_decode_audio(
                audio_state.latent,
                stage_2_audio_decoder,
                stage_2_vocoder,
            )
            stage_results.append((decoded_video, decoded_audio))

        # Final cleanup
        torch.cuda.synchronize()
        del stage_2_transformer, video_encoder, upsampler
        del stage_2_video_decoder, stage_2_audio_decoder, stage_2_vocoder
        cleanup_memory()

        return stage_results


def load_multi_stage_pipeline(
    checkpoint_path: str = os.path.join(MODELS_PATH, "ltx-2-19b-dev.safetensors"),
    gemma_root_path: str = os.path.join(MODELS_PATH, "gemma-3-12b-it-qat-q4_0-unquantized"),
    spatial_upsampler_path: str = os.path.join(MODELS_PATH, "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
    distilled_lora_path: str = os.path.join(MODELS_PATH, "ltx-2-19b-distilled-lora-384.safetensors"),
    device: str = "cuda",
    quantization: QuantizationPolicy | None = None,
    loras: list[LoraPathStrengthAndSDOps] = None,
) -> LayoutGuidedLTXMultiStagePipeline:
    distilled_lora = LoraPathStrengthAndSDOps(
        distilled_lora_path,
        1.0,
        LTXV_LORA_COMFY_RENAMING_MAP,
    )
    pipeline = LayoutGuidedLTXMultiStagePipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[distilled_lora],
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root_path,
        loras=loras or [],
        device=device,
        quantization=quantization,
    )

    return pipeline
