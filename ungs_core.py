from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sta_core_v2 import Rank8_Projection


class UNGSCore(nn.Module):
    """
    Minimal UNGS operator block for action-state generation.

    Operator semantics:
      N(h)        = -h + phi(h)
      E(h, N(h))  = fuse([h, N(h), h*N(h)])
      out         = norm(h + E + memory)

    It also returns differentiable axiom losses and cheap emergence metrics.
    """

    def __init__(self, dim: int, rank: int = 8, rel_threshold: float = 0.6):
        super().__init__()
        self.neg_proj = Rank8_Projection(dim, rank)
        self.enc_fuse = nn.Linear(dim * 3, dim, bias=True)
        self.rec_proj = nn.Linear(dim, dim, bias=True)
        self.mem_proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)

        self.memory = nn.Parameter(torch.zeros(dim))
        self.rel_threshold = rel_threshold

        nn.init.normal_(self.enc_fuse.weight, std=0.02)
        nn.init.zeros_(self.enc_fuse.bias)
        nn.init.normal_(self.rec_proj.weight, std=0.02)
        nn.init.zeros_(self.rec_proj.bias)
        nn.init.normal_(self.mem_proj.weight, std=0.02)
        nn.init.zeros_(self.mem_proj.bias)

    def _negation(self, h: torch.Tensor) -> torch.Tensor:
        return -h + self.neg_proj(h)

    def forward(
        self,
        h: torch.Tensor,
        *,
        compute_losses: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        n1 = self._negation(h)
        encapsulated = self.enc_fuse(torch.cat([h, n1, h * n1], dim=-1))

        mem = self.mem_proj(self.memory).view(1, 1, -1)
        out = self.norm(h + encapsulated + mem)

        losses: Dict[str, torch.Tensor] = {}
        if compute_losses:
            n2 = self._negation(n1)
            closure_loss = F.mse_loss(n2, -h)
            encapsulation_loss = F.mse_loss(self.rec_proj(encapsulated), h)
            # Compare global state summary against memory target to avoid batch broadcasting artifacts.
            self_ref_loss = F.mse_loss(out.mean(dim=(0, 1)), mem.view(-1))
            losses = {
                "closure": closure_loss,
                "encapsulation": encapsulation_loss,
                "self_ref": self_ref_loss,
            }

        metrics = self._compute_metrics(out, losses)
        return out, losses, metrics

    def _compute_metrics(
        self,
        out: torch.Tensor,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        with torch.no_grad():
            # Relation density on a sampled token subset to avoid quadratic blow-up.
            sample_t = min(64, out.size(1))
            sample = out[:, :sample_t, :]
            sample = F.normalize(sample, dim=-1)
            sim = torch.matmul(sample, sample.transpose(1, 2))

            eye = torch.eye(sample_t, device=out.device, dtype=torch.bool).unsqueeze(0)
            rel_mask = (~eye) & (sim > self.rel_threshold)
            denom = max(sample_t * sample_t - sample_t, 1)
            relation_density = float(rel_mask.float().sum().item() / (out.size(0) * denom))

            # Effective rank ratio as hierarchy compression proxy.
            flat = out.reshape(-1, out.size(-1))
            flat = flat[: min(flat.size(0), 512)]
            s = torch.linalg.svdvals(flat.float())
            p = s / s.sum().clamp(min=1e-8)
            entropy = -(p * torch.log(p.clamp(min=1e-8))).sum()
            effective_rank = float(torch.exp(entropy).item())
            hierarchy_ratio = effective_rank / float(out.size(-1))

            self_ref = float(losses.get("self_ref", torch.tensor(0.0, device=out.device)).item())
            self_ref_consistency = 1.0 / (1.0 + self_ref)

        return {
            "relation_density": relation_density,
            "hierarchy_ratio": hierarchy_ratio,
            "self_ref_consistency": self_ref_consistency,
        }


def ungs_total_loss(
    losses: Dict[str, torch.Tensor],
    *,
    closure_lambda: float,
    encapsulation_lambda: float,
    self_ref_lambda: float,
) -> torch.Tensor:
    if not losses:
        return torch.tensor(0.0)
    return (
        closure_lambda * losses["closure"]
        + encapsulation_lambda * losses["encapsulation"]
        + self_ref_lambda * losses["self_ref"]
    )
