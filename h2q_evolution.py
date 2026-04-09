"""
h2q_evolution.py — H2Q_Evolution_Engine
========================================
Ultra-compact, byte-level auto-regressive language model built entirely from
sta_core_v2 primitives.

Design constraints (from H2Q-MicroStream architecture):
  * VOCAB_SIZE = 256  (hard-locked to raw Unicode bytes; no BPE, no tokeniser)
  * All weight matrices use Rank-8 bottleneck projections throughout
  * Causal mask enforces the time-arrow constraint in every STA attention layer
  * Parameter budget: dim=128, num_layers=4 → ~100 K params < 1 MB

Architecture:
    nn.Embedding(256, dim)
    -> N × STA_Block
         pre-norm LayerNorm -> Stereographic_Attention_Layer_V2 (causal=True)
         pre-norm LayerNorm -> Rank8_FeedForward  (dim->rank->GELU->dim)
    -> nn.LayerNorm(dim)
    -> nn.Linear(dim, 256, bias=False)   [weight-tied to embedding]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sta_core_v2 import Rank8_Projection, Stereographic_Attention_Layer_V2

# ---------------------------------------------------------------------------
# Rank-8 Feed-Forward Block
# ---------------------------------------------------------------------------

class Rank8_FeedForward(nn.Module):
    """
    Feed-forward block constrained to Rank-8 bottleneck throughout.

    Computation:
        dim -> 8 -> GELU -> 8 -> dim

    Using two sequential Rank8_Projection modules keeps the entire
    intermediate representation within the 8 essential topological
    eigenmodes — consistent with the MicroStream "less is more" philosophy.

    Args:
        dim:  input/output feature dimension
        rank: bottleneck rank (default 8)
    """

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.up   = Rank8_Projection(dim, rank)
        self.down = Rank8_Projection(dim, rank)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


# ---------------------------------------------------------------------------
# STA Block (one transformer layer)
# ---------------------------------------------------------------------------

class STA_Block(nn.Module):
    """
    One transformer-style block using causal STA v2 attention + Rank-8 FF.

    Pre-norm residual layout:
        x -> LayerNorm -> STA_V2 (causal=True) -> + x
          -> LayerNorm -> Rank8_FeedForward     -> + x

    Args:
        dim:                 feature dimension
        rank:                Rank-8 bottleneck width
        shockwave_threshold: Lambda in [0, pi] for algebraic shockwave mask
        max_seq_len:         maximum sequence length passed to pos_enc
    """

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        shockwave_threshold: float = math.pi / 2,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Stereographic_Attention_Layer_V2(
            hidden_dim=dim,
            shockwave_threshold=shockwave_threshold,
            rank=rank,
            max_seq_len=max_seq_len,
            causal=True,          # enforce time-arrow causality
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = Rank8_FeedForward(dim, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, dim]

        Returns:
            [B, L, dim]
        """
        x = x + self.attn(self.norm1(x))   # causal STA residual
        x = x + self.ff(self.norm2(x))     # rank-8 FF residual
        return x


# ---------------------------------------------------------------------------
# H2Q_Evolution_Engine — the full model
# ---------------------------------------------------------------------------

class H2Q_Evolution_Engine(nn.Module):
    """
    H2Q Evolution Engine: a byte-stream auto-regressive model designed for
    continuous online (self-evolving) learning via Rolling Horizon Causal
    Validation.

    Key properties:
      * VOCAB = 256: language reduced to pure 1-D numeric signal (raw bytes).
        No BPE, no SentencePiece, no statistical tokeniser of any kind.
      * All weight matrices route through Rank-8 bottlenecks — the model is
        forced to compress all representations into 8 essential topological
        eigenmodes.
      * Weight tying: the output projection shares the embedding matrix,
        halving the parameter count of the byte interface.
      * get_topology_sparsity() exposes the mean shockwave-truncation fraction
        across all STA layers, enabling real-time monitoring of active noise
        suppression during the evolution loop.

    Default configuration (dim=128, num_layers=4, rank=8):
      * ~100 K trainable parameters  (<1 MB float32)
      * Suitable for CPU or low-VRAM GPU online learning

    Args:
        dim:                 hidden dimension (default 128)
        num_layers:          number of STA blocks (default 4)
        rank:                Rank-8 bottleneck width for all projections
        shockwave_threshold: Lambda in [0, pi] for algebraic shockwave mask
        max_seq_len:         maximum context length
    """

    VOCAB: int = 256

    def __init__(
        self,
        dim: int = 128,
        num_layers: int = 4,
        rank: int = 8,
        shockwave_threshold: float = math.pi / 2,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim         = dim
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len

        # ── Byte embedding: maps each of the 256 byte values to a vector ──────
        self.embedding = nn.Embedding(self.VOCAB, dim)

        # ── N × STA blocks (causal, rank-8 throughout) ────────────────────────
        self.blocks = nn.ModuleList(
            [
                STA_Block(
                    dim=dim,
                    rank=rank,
                    shockwave_threshold=shockwave_threshold,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

        # ── Final norm + output projection ────────────────────────────────────
        self.final_norm = nn.LayerNorm(dim)
        self.head       = nn.Linear(dim, self.VOCAB, bias=False)

        # Weight tying: output logit matrix shares the embedding weight tensor.
        # This enforces a consistent byte-space metric between input and output.
        self.head.weight = self.embedding.weight

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.ones_(block.norm1.weight)
            nn.init.zeros_(block.norm1.bias)
            nn.init.ones_(block.norm2.weight)
            nn.init.zeros_(block.norm2.bias)
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                  # [B, L]  long — byte indices [0, 255]
        targets: torch.Tensor = None,     # [B, L]  long — next-byte targets
    ):
        """
        Args:
            x:       [B, L]  integer byte indices in [0, 255]
            targets: [B, L]  integer targets for next-byte cross-entropy (optional)

        Returns:
            logits: [B, L, 256]
            loss:   scalar cross-entropy loss, or None if targets not provided
        """
        h = self.embedding(x)          # [B, L, dim]

        for block in self.blocks:
            h = block(h)               # [B, L, dim]  (causal, rank-8)

        h      = self.final_norm(h)    # [B, L, dim]
        logits = self.head(h)          # [B, L, 256]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.VOCAB),
                targets.view(-1),
            )

        return logits, loss

    # ── Topology diagnostics ──────────────────────────────────────────────────

    def get_topology_sparsity(self) -> float:
        """
        Return the mean algebraic shockwave sparsity across all STA layers.

        Sparsity = fraction of (query, key) pairs whose geodesic distance on
        S^D exceeds Lambda and were therefore hard-zeroed by the shockwave
        truncation mechanism.

        A rising sparsity value indicates the model is actively identifying and
        pruning low-relevance token connections — evidence that the Rank-8
        topological eigenmodes are converging to meaningful causal structure.

        Returns:
            float in [0, 1]
        """
        values = [block.attn.last_sparsity for block in self.blocks]
        return sum(values) / len(values) if values else 0.0

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_size_mb(self) -> float:
        """Return total size of trainable parameters in megabytes (float32)."""
        return (
            sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad)
            / 1e6
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,   # [1, L_prompt]  long
        new_bytes: int = 200,
    ) -> torch.Tensor:
        """
        Greedy / multinomial next-byte generation for inspection.

        Args:
            prompt:    [1, L_prompt]  seed byte indices
            new_bytes: number of bytes to generate

        Returns:
            [1, L_prompt + new_bytes]  long tensor
        """
        self.eval()
        idx = prompt
        for _ in range(new_bytes):
            idx_cond      = idx[:, -self.max_seq_len :]
            logits, _     = self(idx_cond)
            next_byte     = torch.multinomial(
                F.softmax(logits[:, -1, :], dim=-1), num_samples=1
            )
            idx = torch.cat([idx, next_byte], dim=1)
        return idx
