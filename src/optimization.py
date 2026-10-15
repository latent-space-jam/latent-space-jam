import math
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol

import torch
import tqdm
from bitsandbytes.optim import AdamW8bit
from loguru import logger
from transformers import AutoTokenizer


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAMW8BIT = "adamw8bit"


@dataclass(frozen=True)
class GuidingPhrase:
    phrase: str
    indices: list[int]


def compute_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    scale_factor = 1.0 / math.sqrt(q.size(-1))
    attn_weight = torch.matmul(q, k.transpose(-1, -2)) * scale_factor

    if attention_mask is not None:
        attn_weight = attn_weight + attention_mask

    return attn_weight


class AttentionStore:
    def __init__(self):
        self.accumulator = None
        self.count = 0
        self.keep_heads = False

    def __call__(self, probs: torch.Tensor) -> None:
        if not self.keep_heads:
            probs = probs.mean(dim=1)

        if self.accumulator is None:
            self.accumulator = probs
        else:
            self.accumulator += probs
        self.count += 1

    def reset(self):
        self.accumulator = None
        self.count = 0

    def get_avg_attention_across_layers(self) -> torch.Tensor:
        return self.accumulator / self.count


class RandomizedAttentionStore(AttentionStore):
    def __init__(self, num_layers: int, total_layers: int, layer_range: List[int], seed: Optional[int] = None):
        super().__init__()
        self.num_layers = min(num_layers, total_layers)
        self.total_layers = total_layers
        self.seed = seed
        self.layer_range = layer_range
        self._rng = random.Random(seed)
        self._call_index = 0
        self._selected_layers = set()
        self.resample()

    def __call__(self, probs: torch.Tensor) -> None:
        if self._call_index in self._selected_layers:
            super().__call__(probs)
        self._call_index += 1

    def reset(self):
        super().reset()
        self._call_index = 0
        self.resample()

    def resample(self) -> List[int]:
        self._selected_layers = set(self._rng.sample(range(self.total_layers), self.num_layers))
        return sorted(self._selected_layers)

    def get_selected_layers(self) -> List[int]:
        return sorted(self._selected_layers)


class LayoutLoss(Protocol):
    def __call__(self, attn: torch.Tensor) -> torch.Tensor: ...


class LayoutKLLoss(LayoutLoss):
    def __init__(
        self,
        bboxes: torch.Tensor,
        indices: list[list[int]],
        eps: float = 1e-8,
        bboxes_weights: Optional[list[float]] = None,
    ):
        self.bboxes = bboxes
        self.indices = indices
        self.eps = eps
        self.bboxes_weights = bboxes_weights or [1.0] * len(bboxes)

    def __call__(self, attn: torch.Tensor) -> torch.Tensor:
        assert attn.ndim == 3, f"Attention weights must be 3D, got shape {attn.shape}"

        loss = 0.0
        bboxes = self.bboxes.to(attn.device)

        for bbox, tokens, weight in zip(bboxes, self.indices, self.bboxes_weights, strict=False):
            if len(tokens) == 0:
                continue

            bbox_flat = bbox.flatten()
            token_attn = attn[:, :, tokens].mean(dim=-1).flatten()

            p = token_attn + self.eps
            p = p / p.sum()

            q = bbox_flat + self.eps
            q = q / q.sum()

            loss += weight * (q * (q.log() - p.log())).sum()

        if not isinstance(loss, torch.Tensor):
            return torch.tensor(0.0, device=attn.device, dtype=attn.dtype, requires_grad=True)

        return loss


class ForwardFunc(Protocol):
    def __call__(self, latents: torch.Tensor) -> torch.Tensor: ...


class LatentRefiner:
    def __init__(
        self,
        loss_fn: LayoutLoss,
        num_refinements: int = 3,
        lr_start: float = 0.01,
        lr_end: float = 0.05,
        betas: tuple[float, float] = (0.4, 0.9),
        weight_decay: float = 0.0,
        optimizer_type: OptimizerType = OptimizerType.ADAMW,
    ):
        self.loss_fn = loss_fn
        self.num_refinements = num_refinements
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.betas = betas
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type

    def _create_optimizer(self, params: list, lr: float) -> torch.optim.Optimizer:
        match self.optimizer_type:
            case OptimizerType.ADAMW:
                return torch.optim.AdamW(
                    params,
                    lr=lr,
                    betas=self.betas,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.ADAMW8BIT:
                return AdamW8bit(
                    params,
                    lr=lr,
                    betas=self.betas,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def optimize(
        self,
        latents: torch.Tensor,
        forward_fn: ForwardFunc,
        store: AttentionStore,
        progress_bar: tqdm.tqdm | None = None,
    ) -> torch.Tensor:
        latents = latents.clone().detach().requires_grad_(True)

        optimizer = self._create_optimizer([latents], lr=self.lr_start)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_refinements, eta_min=self.lr_end)
        first_loss = None

        with torch.enable_grad():
            for i in range(self.num_refinements):
                latents = latents.requires_grad_(True)
                store.reset()

                _ = forward_fn(latents)

                attn = store.get_avg_attention_across_layers()
                loss = self.loss_fn(attn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if i == 0:
                    first_loss = loss.item()

                if progress_bar is not None:
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        loss=f"{first_loss:.2f}→{loss.item():.2f}",
                        grad=f"{latents.grad.norm().item():.2e}",
                        lr=f"{current_lr:.2e}",
                        refine=f"{i + 1}/{self.num_refinements}",
                    )

        store.reset()
        return latents.detach().clone()


def get_token_indices_for_phrase_indices(
    phrase_indices: list[list[int]],
    prompt: str,
    tokenizer: AutoTokenizer,
) -> list[list[int]]:
    word_char_spans = [(m.start(), m.end()) for m in re.finditer(r"\w+(?:[-']\w+)*", prompt)]

    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding["offset_mapping"]

    word_to_tokens: dict[int, list[int]] = {}
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for word_idx, (char_start, char_end) in enumerate(word_char_spans):
            if tok_start >= char_end:
                continue
            if tok_end <= char_start:
                break
            word_to_tokens.setdefault(word_idx, []).append(tok_idx)

    num_words = len(word_char_spans)
    result = []
    for word_group in phrase_indices:
        token_indices = []
        for word_idx in word_group:
            if word_idx >= num_words:
                logger.warning(f"Word index {word_idx} out of range for prompt with {num_words} words")
                continue
            token_indices.extend(word_to_tokens.get(word_idx, []))
        result.append(sorted(set(token_indices)))

    return result
