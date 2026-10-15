from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.attention import AttentionCallable

from src.optimization import AttentionStore


class PytorchAttentionWithStore(AttentionCallable):
    def __init__(self, store: AttentionStore):
        self.store = store

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        ik = torch.eye(k.shape[-2], dtype=k.dtype, device=k.device).cuda().expand(k.shape[0], k.shape[1], -1, -1)
        vik = torch.cat([v, ik], dim=-1)

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(q, k, vik, is_causal=False, attn_mask=None)
        out, attn_weights = sdpa_out.split([v.shape[-1], k.shape[-2]], dim=-1)
        self.store(attn_weights)

        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


def register_attention_hook(x0_model: X0Model, store: AttentionStore, layers_to_hook: list[int]) -> dict[str, Any]:
    original_attentions = {}
    registered = 0

    for idx, block in enumerate(x0_model.velocity_model.transformer_blocks):
        if idx in layers_to_hook and hasattr(block, "attn2"):
            full_name = f"block_{idx}_attn2"
            original_attentions[full_name] = block.attn2.attention_function
            block.attn2.attention_function = PytorchAttentionWithStore(store)
            registered += 1

    logger.info(f"Registered {registered}/{len(x0_model.velocity_model.transformer_blocks)} transformer blocks")
    return original_attentions


def unregister_attention_hook(x0_model: X0Model, original_attentions: dict[str, Any]) -> None:
    restored = 0

    for idx, block in enumerate(x0_model.velocity_model.transformer_blocks):
        full_name = f"block_{idx}_attn2"
        if full_name in original_attentions:
            block.attn2.attention_function = original_attentions[full_name]
            restored += 1

    logger.info(f"Unregistered {restored} attention hooks")
