"""
TCRH-Layer: Topological Class Routing & Hashing Layer
=====================================================
A binary LSH-based sparse attention mechanism for the H2Q-MicroStream
architecture. Replaces O(N^2) dot-product attention with integer-comparison
and bitwise-Hamming filtering, yielding a sparse directed-acyclic-graph (DAG)
of token interactions.

Terminology map (problem-statement name → actual technique):
  "Locality-Sensitive Homotopy Hashing"   → Binary LSH via sign quantization
  "Chern Class Integer Tagging"           → Coarse bucket assignment (top-k bits)
  "Homotopy Bitwise Hashing"              → Hamming distance on binary codes
  "Categorical Commuting Routing"         → Bucket-grouped sparse routing
  "Shock Truncation"                      → Hard connectivity mask (0/1 weights)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Topological Hash Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Topological_Hash_Encoder(nn.Module):
    """
    Encodes a real-valued feature tensor into two discrete representations:

      1. ``hash_signatures`` (int8, shape [B, T, hash_dim]):
         Binary codes produced by sign-quantising a random linear projection
         of the input.  Equivalent to a random-hyperplane LSH.
         → "Locality-Sensitive Homotopy Hashing" in the problem statement.

      2. ``chern_tags`` (int32, shape [B, T]):
         Coarse bucket index obtained by interpreting the leading
         ``ceil(log2(num_buckets))`` bits of each binary code as an integer.
         Tokens that land in the same bucket are candidates for interaction.
         → "Chern Class Integer Tagging" in the problem statement.
    """

    def __init__(self, hidden_dim: int, hash_dim: int = 64, num_buckets: int = 16):
        super().__init__()
        self.hash_dim = hash_dim
        self.num_buckets = num_buckets

        # Number of bits used to form the Chern-class bucket index
        self.tag_bits = max(1, math.ceil(math.log2(max(num_buckets, 2))))

        # Fixed random projection matrix (not trained); kept as a buffer so it
        # moves with the module to the correct device.
        self.register_buffer("proj", torch.randn(hidden_dim, hash_dim) / (hidden_dim ** 0.5))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: float tensor of shape [B, T, D]

        Returns:
            hash_signatures : int8 tensor  [B, T, hash_dim]   (0 / 1 per bit)
            chern_tags       : int32 tensor [B, T]             (bucket index)
        """
        # ── Locality-Sensitive Homotopy Hashing (sign quantisation) ──────────
        # Project into hash space, then binarise.  Two tokens that were close
        # in the original space will tend to share more bits (low Hamming dist).
        projected = x @ self.proj.to(x.dtype)           # [B, T, hash_dim]
        hash_signatures = (projected > 0).to(torch.int8)  # [B, T, hash_dim]  ← LSH binarisation

        # ── Chern Class Integer Filter: bucket assignment ─────────────────────
        # Interpret the first `tag_bits` bits as a binary integer → bucket id.
        # Vectorised: multiply each bit column by its positional power of 2.
        powers = torch.arange(self.tag_bits, device=x.device, dtype=torch.int32)  # [tag_bits]
        chern_tags = (
            hash_signatures[:, :, :self.tag_bits].to(torch.int32) * (2 ** powers)
        ).sum(dim=-1)                                     # [B, T] ← Chern Class Integer Tag assembly

        return hash_signatures, chern_tags


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: TCRH Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class TCRH_Attention_Layer(nn.Module):
    """
    Topological Class Routing & Hashing Attention.

    Two-level sparse filtering:
      Level 1 – Chern Class Integer Filter:
        If chern_tag[i] ≠ chern_tag[j], the pair is topologically
        incompatible → weight set to 0 (shock truncation).            [line ~113]

      Level 2 – Homotopy Bitwise Hashing (Hamming distance):
        Among same-bucket pairs, compute Hamming distance between
        binary hash signatures.  Pairs whose distance exceeds the
        threshold Λ are also truncated to 0.                           [line ~121]

    Connected pairs (distance ≤ Λ) receive uniform weight (1 / |bucket|),
    which avoids any floating-point similarity computation entirely.

    For long sequences this class offers a *bucket-grouped* forward path
    (``use_bucket_mode=True``) that avoids materialising the full [B,T,T]
    interaction matrix, keeping memory O(N · bucket_size).
    """

    def __init__(
        self,
        hidden_dim: int,
        hash_dim: int = 64,
        num_buckets: int = 16,
        hamming_threshold: int = 8,
        use_bucket_mode: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim
        self.hamming_threshold = hamming_threshold
        self.use_bucket_mode = use_bucket_mode

        self.encoder = Topological_Hash_Encoder(hidden_dim, hash_dim, num_buckets)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    # ── pairwise mode (exact, O(N²) memory, suitable for T ≤ 4096) ──────────

    def _forward_pairwise(self, x: torch.Tensor, hash_sigs: torch.Tensor,
                          chern_tags: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        V = self.v_proj(x)                                # [B, T, D]

        # ── Level 1: Chern Class Integer Filter ──────────────────────────────
        c_q = chern_tags.unsqueeze(2)                     # [B, T, 1]
        c_k = chern_tags.unsqueeze(1)                     # [B, 1, T]
        chern_match = (c_q == c_k)                        # [B, T, T] ← Chern Class Integer Filter (line 113)

        # ── Level 2: Homotopy Bitwise Hashing (Hamming distance) ─────────────
        h_q = hash_sigs.unsqueeze(2).to(torch.int32)     # [B, T, 1, hash_dim]
        h_k = hash_sigs.unsqueeze(1).to(torch.int32)     # [B, 1, T, hash_dim]
        hamming_dist = (h_q != h_k).sum(dim=-1)          # [B, T, T] ← Homotopy Bitwise Hashing (line 121)

        # Shock truncation: connect only if BOTH filters pass
        connected = chern_match & (hamming_dist <= self.hamming_threshold)  # [B, T, T]

        # Uniform weights over connected neighbours (no float dot-product)
        weights = connected.float()
        norm = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = weights / norm                          # [B, T, T]

        out = weights @ V                                 # [B, T, D]
        return self.o_proj(out)

    # ── bucket mode (O(N · bucket_size) memory, suitable for long sequences) ─

    def _forward_buckets(self, x: torch.Tensor, hash_sigs: torch.Tensor,
                         chern_tags: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        V = self.v_proj(x)                                # [B, T, D]
        out = torch.zeros_like(V)

        for b in range(B):
            unique_tags = chern_tags[b].unique()          # per-sample unique buckets
            for tag in unique_tags:
                tag_val = tag.item()
                idx = (chern_tags[b] == tag_val).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue

                # ── Homotopy Bitwise Hashing within bucket ────────────────────
                h = hash_sigs[b][idx].to(torch.int32)    # [k, hash_dim]
                h_q = h.unsqueeze(1)                      # [k, 1, hash_dim]
                h_k = h.unsqueeze(0)                      # [1, k, hash_dim]
                hdist = (h_q != h_k).sum(dim=-1)          # [k, k] ← Homotopy Bitwise Hashing

                connected = (hdist <= self.hamming_threshold).float()  # [k, k]
                norm = connected.sum(dim=-1, keepdim=True).clamp(min=1.0)
                w = connected / norm                      # [k, k]

                out[b][idx] = w @ V[b][idx]               # [k, D]

        return self.o_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor [B, T, D]
        Returns:
            out: float tensor [B, T, D]
        """
        hash_sigs, chern_tags = self.encoder(x)

        if self.use_bucket_mode:
            return self._forward_buckets(x, hash_sigs, chern_tags)
        else:
            return self._forward_pairwise(x, hash_sigs, chern_tags)

    # Expose the last computed connectivity statistics for analysis
    @torch.no_grad()
    def connectivity_stats(self, x: torch.Tensor):
        """Return fraction of token pairs that are topologically connected."""
        hash_sigs, chern_tags = self.encoder(x)
        B, T, _ = x.shape

        c_q = chern_tags.unsqueeze(2)
        c_k = chern_tags.unsqueeze(1)
        chern_match = (c_q == c_k)

        h_q = hash_sigs.unsqueeze(2).to(torch.int32)
        h_k = hash_sigs.unsqueeze(1).to(torch.int32)
        hamming_dist = (h_q != h_k).sum(dim=-1)

        connected = chern_match & (hamming_dist <= self.hamming_threshold)
        frac = connected.float().mean().item()
        intercept_rate = 1.0 - frac
        return {
            "connected_fraction": frac,
            "bitwise_xor_intercept_rate": intercept_rate,
            "total_pairs": B * T * T,
            "connected_pairs": connected.sum().item(),
        }
