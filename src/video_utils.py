import logging
import os
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.callbacks import PipelineCallback
from diffusers.utils import export_to_video, load_video
from loguru import logger
from PIL.Image import Image
from torchvision.transforms import ToPILImage

from src.optimization import AttentionStore, GuidingPhrase

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)


def load_video_as_tensor(path: str, height: int, width: int, mode: Literal["RGB", "L"] = "RGB") -> torch.Tensor:
    frames = load_video(path)
    frames = [frame.convert(mode).resize((width, height)) for frame in frames]
    frames = np.array(frames)
    if mode == "L":
        frames = np.expand_dims(frames, -1)
    frames = torch.tensor(frames).permute(0, 3, 1, 2)
    frames = frames.float() / 255.0
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return frames


def downsample_mask_to_latent_resolution(mask: torch.Tensor, vae_temporal_scale_factor: int, vae_spatial_scale_factor: int) -> torch.Tensor:
    t1 = mask[:, :, 0:1]
    t2 = mask[:, :, 1::vae_temporal_scale_factor]
    t_sampled = torch.cat([t1, t2], dim=2)

    t = t_sampled.shape[2]
    h_m = mask.shape[-2] // vae_spatial_scale_factor
    w_m = mask.shape[-1] // vae_spatial_scale_factor

    sampled = F.interpolate(t_sampled, size=(t, h_m, w_m), mode="nearest")
    return sampled


def upsample_attention_to_patch_resolution(attn: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    ph, pw = patch_size[1], patch_size[2]
    attn = attn.repeat_interleave(ph, dim=-2)
    attn = attn.repeat_interleave(pw, dim=-1)
    return attn


def attn_to_pil(attn: torch.Tensor) -> list[Image]:
    frames = []
    for frame in attn:
        frame -= frame.min()
        frame /= frame.max() + 1e-8
        img = (frame * 255).to(torch.uint8)
        frames.append(ToPILImage()(img.permute(2, 0, 1)))
    return frames


def video_to_tensor(video: list[Image]) -> torch.Tensor:
    video = np.array(video)
    video = torch.tensor(video).permute(0, 3, 1, 2)
    video = video.float() / 255.0
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video


class _BaseAttnMapsCallback(PipelineCallback):
    def __init__(
        self,
        store: AttentionStore,
        guiding_phrases: list[GuidingPhrase],
        output_path: str,
        fps: float = 7.0,
    ):
        super().__init__()
        self._store = store
        self._guiding_phrases = guiding_phrases
        self._output_path = output_path
        self._fps = fps

        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Saving attention maps to {output_path}")

    @property
    def tensor_inputs(self) -> list[str]:
        return ["latents"]

    def _export_attention_maps(
        self,
        attention_maps: torch.Tensor,
        step_index: int,
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> None:
        for guiding_phrase in self._guiding_phrases:
            if len(guiding_phrase.indices) == 0:
                continue

            tokens_map = attention_maps[:, :, guiding_phrase.indices].mean(-1).reshape(lat_f, lat_h, lat_w, 1)
            att_pil = attn_to_pil(tokens_map)
            video_path = f"{self._output_path}/{guiding_phrase.phrase}_{step_index}.mp4"
            export_to_video(att_pil, video_path, fps=self._fps)


class SaveAttnMapsCallbackWAN(_BaseAttnMapsCallback):
    def __init__(
        self,
        store: AttentionStore,
        guiding_phrases: list[GuidingPhrase],
        latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        output_path: str,
        fps: float = 7.0,
    ):
        super().__init__(store, guiding_phrases, output_path, fps)
        self._latent_shape = latent_shape
        self._patch_size = patch_size

    def callback_fn(
        self,
        pipeline,
        step_index: int,
        timestep: float,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        latents = callback_kwargs["latents"]
        attn = self._store.get_avg_attention_across_layers()

        lat_shape = latents.shape[2:]
        ph, pw = self._patch_size[1], self._patch_size[2]
        attn_shape = (lat_shape[0], lat_shape[1] // ph, lat_shape[2] // pw)
        attn = attn.reshape(latents.shape[0], *attn_shape, attn.shape[-1])
        attn = attn.repeat_interleave(ph, dim=-3)
        attn = attn.repeat_interleave(pw, dim=-2)
        b, f, h, w, c = attn.shape
        attn = attn.reshape(b, f * h * w, c)

        self._export_attention_maps(attn, step_index, lat_shape[0], lat_shape[1], lat_shape[2])

        self._store.reset()
        return callback_kwargs


class SaveAttnMapsCallbackLTX(_BaseAttnMapsCallback):
    def __init__(
        self,
        store: AttentionStore,
        guiding_phrases: list[GuidingPhrase],
        video_shape: tuple[int, int, int],
        output_path: str,
        fps: float = 7.0,
    ):
        super().__init__(store, guiding_phrases, output_path, fps)
        self._video_shape = video_shape

    def callback_fn(
        self,
        pipeline,
        step_index: int,
        timestep: float,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        frames, height, width = self._video_shape
        lat_h = height // 32
        lat_w = width // 32
        lat_f = (frames - 1) // 8 + 1

        if self._store.accumulator is not None:
            attention_maps = self._store.get_avg_attention_across_layers()

            batch_size = attention_maps.shape[0]
            if batch_size == 2:
                attention_maps = attention_maps[-1:]

            self._export_attention_maps(attention_maps, step_index, lat_f, lat_h, lat_w)

        self._store.reset()
        return callback_kwargs
