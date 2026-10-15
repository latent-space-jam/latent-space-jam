from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from loguru import logger

from src.optimization import AttentionStore, compute_attention_weights


class AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        store: AttentionStore,
        name: str,
        store_image_attn: bool = False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0+ (scaled_dot_product_attention)")

        self.store = store
        self.name = name
        self.layer_idx = int(self.name.split(".")[2])
        self.store_image_attn = store_image_attn

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                x: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).permute(0, 2, 1, 3)
            value_img = value_img.unflatten(2, (attn.heads, -1)).permute(0, 2, 1, 3)
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        is_cross = getattr(attn, "is_cross_attention", False)
        if is_cross:
            attn_weight = compute_attention_weights(query, key, attention_mask)
            attn_weight = F.softmax(attn_weight, dim=-1)
            self.store(attn_weight)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def register_attention_hook(
    pipeline: WanPipeline | WanImageToVideoPipeline,
    store: AttentionStore,
    layers_to_hook: list[int] = [],
    cross_attention_only: bool = True,
) -> dict[str, any]:
    original_attn_processors = {}

    def _wrap_module(transformer, prefix: str):
        num_reg, total = 0, 0
        for name, module in transformer.named_modules():
            if isinstance(module, WanAttention):
                is_cross_attention = getattr(module, "is_cross_attention", False)
                if cross_attention_only and is_cross_attention:
                    if total in layers_to_hook:
                        full_name = f"{prefix}.{name}"
                        original_attn_processors[full_name] = module.processor
                        module.set_processor(
                            AttnProcessor(
                                store=store,
                                name=full_name,
                                store_image_attn=False,
                            )
                        )
                        num_reg += 1
                total += 1

        return num_reg, total

    if pipeline.transformer is not None:
        num_reg, total = _wrap_module(pipeline.transformer, "transformer")
        logger.info(f"Wrapped {num_reg}/{total} attention layers in transformer")

    if getattr(pipeline, "transformer_2", None) is not None:
        num_reg, total = _wrap_module(pipeline.transformer_2, "transformer_2")
        logger.info(f"Wrapped {num_reg}/{total} attention layers in transformer_2")

    return original_attn_processors


def unregister_attention_hook(
    pipeline: WanPipeline | WanImageToVideoPipeline,
    original_attn_processors: dict[str, WanAttention],
):
    def restore_module(transformer, prefix: str):
        num_restored = 0
        for name, module in transformer.named_modules():
            if isinstance(module, WanAttention):
                full_name = f"{prefix}.{name}"
                if full_name in original_attn_processors:
                    module.set_processor(original_attn_processors[full_name])
                    num_restored += 1
        return num_restored

    if pipeline.transformer is not None:
        num_restored = restore_module(pipeline.transformer, "transformer")
        logger.info(f"Restored {num_restored} attention layers in transformer")

    if getattr(pipeline, "transformer_2", None) is not None:
        num_restored = restore_module(pipeline.transformer_2, "transformer_2")
        logger.info(f"Restored {num_restored} attention layers in transformer_2")
