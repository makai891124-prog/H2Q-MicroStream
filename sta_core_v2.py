"""
sta_core_v2.py -- STA v2: Optimised Stereographic Topological Attention
========================================================================
Drop-in replacement for sta_core.py with three hardware-aware optimisations:

Optimisation 1 -- Algebraic Shockwave Truncation  (no arccos)
  The geodesic condition  arccos(<s_q,s_k>) > Lambda  is algebraically
  equivalent to  <s_q,s_k> < cos(Lambda)  for Lambda in [0, pi], because
  cosine is strictly monotone-decreasing on that interval.
  cos(Lambda) is pre-computed once in __init__ as self.cos_lambda and the
  forward pass uses a plain scalar comparison:
      mask = raw_inner < self.cos_lambda   (True => hard-zero)
  This eliminates every call to torch.acos / torch.arccos, saving the
  expensive transcendental evaluation on every (query, key) pair.

Optimisation 2 -- Rank-8 Projection  (low-rank bottleneck)
  Standard nn.Linear(D, D) has D^2 parameters and O(B*L*D^2) FLOPs.
  Rank8_Projection replaces it with  D -> 8 -> D  giving:
      params:  D*8 + 8*D = 16*D  vs  D*D   (98.4% reduction at D=1024)
      FLOPs:   O(B*L*16*D)       vs  O(B*L*D^2)
  The bottleneck forces the network to route information through 8 essential
  topological eigenmodes -- a structural regulariser matching the
  MicroStream philosophy of "less is more".

Optimisation 3 -- Batched Spherical Inner Product  (torch.bmm)
  Sphere inner product is computed as:
      raw_inner = torch.bmm(q_s, k_s.transpose(1, 2))  # [B, L, L]
  No explicit expansion or broadcasting materialises the full [B,L,L,D+1]
  tensor; torch.bmm routes directly to cuBLAS / MPS GEMM kernels.

All other components (inverse_stereo_project, stereo_project,
SphericalTopologicalEncoding) are mathematically identical to v1.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import binary_sta_cuda_ext
except Exception:
    binary_sta_cuda_ext = None

# ---------------------------------------------------------------------------
# Device selection (mirrors train_V2.0Q.py)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Stereographic projection helpers  (unchanged from v1)
# ---------------------------------------------------------------------------

def inverse_stereo_project(x: torch.Tensor):
    """
    Lift flat feature vectors from R^D to the unit hypersphere S^D in R^{D+1}
    via inverse stereographic projection.

        r2   = ||x||^2
        xi_i = 2*x_i / (r2 + 1)      [line A]  <- equatorial coords
        eta  = (r2 - 1) / (r2 + 1)   [line B]  <- polar axis

    ╔══════════════════════════════════════════════════════════╗
    ║  NORTH POLE WORMHOLE (lines A+B):                       ║
    ║  As |x| -> inf, eta -> 1 and xi_i -> 0.                 ║
    ║  All large-norm tokens collapse to the same geometric   ║
    ║  point (North Pole), chord-distance -> 0 between them   ║
    ║  regardless of their 1-D sequence separation.           ║
    ╚══════════════════════════════════════════════════════════╝

    Args:
        x: [..., D]  float32

    Returns:
        s:     [..., D+1]  unit sphere point  (xi_1, ..., xi_D, eta)
        omega: [..., 1]    conformal factor   Omega(x) = 2/(r2+1) in (0, 1]
    """
    r2    = x.pow(2).sum(dim=-1, keepdim=True)     # [..., 1]
    denom = r2 + 1.0                               # [..., 1]
    xi    = (2.0 * x) / denom                     # [..., D]  [line A]  xi_i = 2*x_i / (r2+1)
    eta   = (r2 - 1.0) / denom                    # [..., 1]  [line B]  eta  = (r2-1)/(r2+1)
    s     = torch.cat([xi, eta], dim=-1)           # [..., D+1]  <- NORTH POLE SHORTCUT
    omega = 2.0 / denom                            # [..., 1]  conformal / residual factor
    return s, omega


def stereo_project(s: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Collapse a unit hypersphere point back to flat R^D via forward
    stereographic projection from the North Pole N = (0,...,0,1).

        x_i = xi_i / (1 - eta)

    Args:
        s:   [..., D+1]  unit sphere point
        eps: numerical guard for eta -> 1

    Returns:
        x: [..., D]  flat space vector
    """
    xi  = s[..., :-1]   # [..., D]
    eta = s[..., -1:]   # [..., 1]
    return xi / (1.0 - eta + eps)


# ---------------------------------------------------------------------------
# Topological Loci Encoding  (unchanged from v1)
# ---------------------------------------------------------------------------

class SphericalTopologicalEncoding(nn.Module):
    """
    Position encoding on the unit hypersphere S^D via block-diagonal
    SO(D+1) Givens rotors (SU(2) analogue).

    Each position t uses angle theta_{t,k} = t * omega_k in rotation
    plane (2k, 2k+1), with RoPE-style log-spaced frequencies omega_k.
    Rotations preserve unit norm: the token stays on S^D.

    Args:
        sphere_dim:  D+1
        max_seq_len: maximum sequence length
    """

    def __init__(self, sphere_dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.sphere_dim = sphere_dim
        self.num_pairs  = sphere_dim // 2

        inv_freq = 1.0 / (
            10000.0 ** (
                torch.arange(0, self.num_pairs, dtype=torch.float32) / self.num_pairs
            )
        )
        # Persistent buffer: included in state_dict, not a gradient parameter
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, s: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s:         [B, L, D+1]
            positions: [L]  integer indices

        Returns:
            s_rot: [B, L, D+1]  (unit norm preserved)
        """
        np_ = self.num_pairs
        theta  = torch.outer(positions.float(), self.inv_freq)   # [L, np_]
        cos_t  = theta.cos().unsqueeze(0)                        # [1, L, np_]
        sin_t  = theta.sin().unsqueeze(0)                        # [1, L, np_]

        even_idx = torch.arange(0, 2 * np_, 2, device=s.device)
        odd_idx  = even_idx + 1

        s_even = s[:, :, even_idx]
        s_odd  = s[:, :, odd_idx]

        new_even = cos_t * s_even - sin_t * s_odd
        new_odd  = sin_t * s_even + cos_t * s_odd

        s_rot = s.clone()
        s_rot[:, :, even_idx] = new_even
        s_rot[:, :, odd_idx]  = new_odd
        return s_rot


# ---------------------------------------------------------------------------
# Optimisation 2: Rank-8 Projection
# ---------------------------------------------------------------------------

class Rank8_Projection(nn.Module):
    """
    Low-rank bottleneck projection: D -> 8 -> D.

    Replaces nn.Linear(D, D) which has D^2 parameters.
    Parameter count: 2 * D * 8 = 16 * D  (98.4% reduction at D=1024).

    The bottleneck forces routing through 8 essential topological
    eigenmodes, acting as a structural regulariser.

    Args:
        dim: feature dimension D
    """

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.Linear(rank, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class StereographicAttentionLayer(nn.Module):
    """
    Binary topological attention using fixed stereographic addressing planes.

    This layer avoids QK^T sequence correlation entirely. Instead it:
      1. Projects tokens onto fixed random planes and sign-quantises them.
      2. Computes structural similarity with chunked int8 XOR.
      3. Normalises the similarity rows into routing weights.
      4. Aggregates a value projection with those routing weights.

    During training the fixed planes evolve with a no-grad anchor-shift rule
    whenever their activation ratio collapses toward all-0 or all-1.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_planes: int = 128,
        chunk_size: int = 64,
        temperature: float = 1.0,
        routing_mode: str = "normalize",
        binary_backend: str = "packbits",
        fused_chunk_compute: bool = True,
        causal: bool = False,
        evolution_eta: float = 0.1,
        evolution_noise_std: float = 0.01,
        evolution_low_threshold: float = 0.1,
        evolution_high_threshold: float = 0.9,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_planes <= 0:
            raise ValueError("num_planes must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if routing_mode not in {"normalize", "softmax"}:
            raise ValueError("routing_mode must be 'normalize' or 'softmax'")
        if binary_backend not in {"int8", "packbits", "cuda_ext"}:
            raise ValueError("binary_backend must be 'int8', 'packbits' or 'cuda_ext'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_dim = hidden_dim
        self.num_planes = num_planes
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.routing_mode = routing_mode
        self.binary_backend = binary_backend
        self.fused_chunk_compute = fused_chunk_compute
        self.causal = causal
        self.evolution_eta = evolution_eta
        self.evolution_noise_std = evolution_noise_std
        self.evolution_low_threshold = evolution_low_threshold
        self.evolution_high_threshold = evolution_high_threshold
        self.eps = eps

        planes = torch.randn(num_planes, hidden_dim, dtype=torch.float32)
        planes = F.normalize(planes, dim=-1)
        self.register_buffer("addressing_planes", planes)
        self.register_buffer("last_activation_ratio", torch.zeros(num_planes, dtype=torch.float32))
        self.register_buffer(
            "pack_shifts",
            (1 << torch.arange(32, dtype=torch.int64)),
        )
        self.register_buffer(
            "popcount_lut",
            torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8),
        )

        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.last_sparsity: float = 0.0
        self.last_invalid_plane_count: int = 0
        self.last_routing_row_sum_mean: float = 0.0
        self.last_routing_density: float = 0.0
        self.cuda_ext_enabled: bool = False

        mode = os.environ.get("BINARY_STA_CUDA_EXT_MODE", "always").strip().lower()
        if mode not in {"always", "infer_long"}:
            mode = "always"
        self.cuda_ext_mode = mode
        self.cuda_ext_min_seq_len = max(1, int(os.environ.get("BINARY_STA_CUDA_EXT_MIN_SEQ", "256")))
        self.packbits_infer_cuda_ext = (
            os.environ.get("BINARY_STA_PACKBITS_INFER_CUDA_EXT", "0").strip() == "1"
        )

        if self.binary_backend in {"cuda_ext", "packbits"}:
            can_use = (
                binary_sta_cuda_ext is not None
                and torch.cuda.is_available()
                and binary_sta_cuda_ext.is_available()
            )
            if can_use and (self.binary_backend == "cuda_ext" or self.packbits_infer_cuda_ext):
                self.cuda_ext_enabled = True
            elif self.binary_backend == "cuda_ext":
                self.binary_backend = "packbits"

    def _effective_binary_backend(self, seq_len: int) -> str:
        backend = self.binary_backend
        if backend == "cuda_ext":
            if not self.cuda_ext_enabled:
                return "packbits"
            if self.cuda_ext_mode == "infer_long" and (self.training or seq_len < self.cuda_ext_min_seq_len):
                return "packbits"
            return "cuda_ext"
        if backend == "packbits" and self.packbits_infer_cuda_ext and self.cuda_ext_enabled:
            if (not self.training) and (seq_len >= self.cuda_ext_min_seq_len):
                return "cuda_ext"
        return backend

    def encode_topology(self, x: torch.Tensor):
        """
        Returns:
            projected: [B, L, K] float
            codes:     [B, L, K] int8 in {0, 1}
        """
        projected = F.linear(x, self.addressing_planes.to(dtype=x.dtype))
        codes = (projected > 0).to(torch.int8)
        return projected, codes

    @torch.no_grad()
    def evolution_step(self, x: torch.Tensor, codes: torch.Tensor) -> int:
        ratios = codes.float().mean(dim=(0, 1))
        invalid = (ratios < self.evolution_low_threshold) | (ratios > self.evolution_high_threshold)

        self.last_activation_ratio.copy_(ratios)
        invalid_count = int(invalid.sum().item())
        self.last_invalid_plane_count = invalid_count
        if invalid_count == 0:
            return 0

        centroid = x.mean(dim=(0, 1)).to(self.addressing_planes.dtype)
        updated = self.addressing_planes[invalid] - self.evolution_eta * centroid.unsqueeze(0)
        if self.evolution_noise_std > 0:
            updated = updated + torch.randn_like(updated) * self.evolution_noise_std
        updated = F.normalize(updated, dim=-1)
        new_planes = self.addressing_planes.clone()
        new_planes[invalid] = updated
        self.addressing_planes.copy_(new_planes)
        return invalid_count

    def _causal_mask(self, q_len: int, total_len: int, start: int, device: torch.device) -> torch.Tensor:
        q_positions = torch.arange(start, start + q_len, device=device).view(1, q_len, 1)
        k_positions = torch.arange(total_len, device=device).view(1, 1, total_len)
        return k_positions <= q_positions

    def _pack_codes_u32(self, codes: torch.Tensor):
        bsz, seq_len, num_planes = codes.shape
        num_words = (num_planes + 31) // 32
        padded_planes = num_words * 32
        if padded_planes != num_planes:
            codes = F.pad(codes, (0, padded_planes - num_planes), value=0)

        bits = codes.to(torch.int64).view(bsz, seq_len, num_words, 32)
        shifts = self.pack_shifts.to(device=codes.device)
        packed = (bits * shifts.view(1, 1, 1, 32)).sum(dim=-1, dtype=torch.int64)
        return packed, num_words

    def _popcount_u32_words(self, words: torch.Tensor) -> torch.Tensor:
        lut = self.popcount_lut.to(device=words.device)
        b0 = words & 0xFF
        b1 = (words >> 8) & 0xFF
        b2 = (words >> 16) & 0xFF
        b3 = (words >> 24) & 0xFF
        return lut[b0] + lut[b1] + lut[b2] + lut[b3]

    def _normalise_similarity(self, similarity: torch.Tensor, causal_mask: torch.Tensor | None) -> torch.Tensor:
        if self.routing_mode == "softmax":
            logits = similarity / self.temperature
            if causal_mask is not None:
                logits = logits.masked_fill(~causal_mask, float("-inf"))
            weights = F.softmax(logits, dim=-1)
            return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        weights = similarity
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return weights / denom

    def _chunk_similarity_int8(self, q_codes: torch.Tensor, full_codes: torch.Tensor, value_dtype: torch.dtype):
        xor_bits = torch.bitwise_xor(q_codes.unsqueeze(2), full_codes.unsqueeze(1))
        mismatch = xor_bits.sum(dim=-1, dtype=torch.int32)
        return (self.num_planes - mismatch).to(dtype=value_dtype) / float(self.num_planes)

    def _chunk_similarity_packbits(self, q_packed: torch.Tensor, full_packed: torch.Tensor, value_dtype: torch.dtype):
        xor_words = torch.bitwise_xor(q_packed.unsqueeze(2), full_packed.unsqueeze(1))
        word_pop = self._popcount_u32_words(xor_words)
        mismatch = word_pop.sum(dim=-1, dtype=torch.int32)
        return (self.num_planes - mismatch).to(dtype=value_dtype) / float(self.num_planes)

    def _chunked_context(self, codes: torch.Tensor, values: torch.Tensor, value_dtype: torch.dtype):
        _, seq_len, _ = codes.shape
        backend = self._effective_binary_backend(seq_len)
        full_codes = codes.contiguous()
        full_packed = None
        if backend == "packbits":
            full_packed, _ = self._pack_codes_u32(full_codes)

        if backend == "cuda_ext":
            full_packed, _ = self._pack_codes_u32(full_codes)
            use_values = values.contiguous()
            try:
                output = binary_sta_cuda_ext.fused_forward(
                    packed_codes=full_packed,
                    values=use_values,
                    num_planes=self.num_planes,
                    chunk_size=self.chunk_size,
                    routing_mode=self.routing_mode,
                    temperature=self.temperature,
                )
                # Extension path currently returns context only.
                return output, 1.0, 1.0
            except Exception:
                self.cuda_ext_enabled = False
                backend = "packbits"
                full_packed = self._pack_codes_u32(full_codes)[0]

        context_chunks = []
        chunk_count = 0
        density_acc = values.new_zeros(())
        row_sum_acc = values.new_zeros(())

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_codes = full_codes[:, start:end, :]

            if backend == "packbits":
                q_packed = full_packed[:, start:end, :]
                similarity = self._chunk_similarity_packbits(q_packed, full_packed, value_dtype)
            else:
                similarity = self._chunk_similarity_int8(q_codes, full_codes, value_dtype)

            causal_mask = None
            if self.causal:
                causal_mask = self._causal_mask(end - start, seq_len, start, codes.device)
                similarity = similarity.masked_fill(~causal_mask, 0.0)

            weights = self._normalise_similarity(similarity, causal_mask)
            context_chunks.append(torch.bmm(weights, values))

            density_acc = density_acc + (weights > 0).to(values.dtype).mean()
            row_sum_acc = row_sum_acc + weights.sum(dim=-1).mean()
            chunk_count += 1

        context = torch.cat(context_chunks, dim=1)
        inv = 1.0 / float(max(chunk_count, 1))
        row_sum_mean = float((row_sum_acc * inv).detach().item())
        density_mean = float((density_acc * inv).detach().item())
        return context, row_sum_mean, density_mean

    def _chunked_similarity(self, codes: torch.Tensor, value_dtype: torch.dtype) -> torch.Tensor:
        _, seq_len, _ = codes.shape
        backend = self._effective_binary_backend(seq_len)
        outputs = []
        full_codes = codes.contiguous()
        full_packed = None
        if backend in {"packbits", "cuda_ext"}:
            full_packed, _ = self._pack_codes_u32(full_codes)

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_codes = full_codes[:, start:end, :]
            if backend in {"packbits", "cuda_ext"}:
                q_packed = full_packed[:, start:end, :]
                similarity = self._chunk_similarity_packbits(q_packed, full_packed, value_dtype)
            else:
                similarity = self._chunk_similarity_int8(q_codes, full_codes, value_dtype)

            causal_mask = None
            if self.causal:
                causal_mask = self._causal_mask(end - start, seq_len, start, codes.device)
                similarity = similarity.masked_fill(~causal_mask, 0.0)

            weights = self._normalise_similarity(similarity, causal_mask)

            outputs.append(weights)

        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, L, D], got {tuple(x.shape)}")
        if x.size(-1) != self.hidden_dim:
            raise ValueError(
                f"hidden dim mismatch: expected {self.hidden_dim}, got {x.size(-1)}"
            )

        _, codes = self.encode_topology(x)
        values = self.v_proj(x)

        if self.fused_chunk_compute:
            output, row_sum_mean, density_mean = self._chunked_context(codes, values, x.dtype)
            self.last_routing_row_sum_mean = row_sum_mean
            self.last_routing_density = density_mean
        else:
            routing = self._chunked_similarity(codes, x.dtype)
            output = torch.bmm(routing, values)
            self.last_routing_row_sum_mean = float(routing.sum(dim=-1).mean().item())
            self.last_routing_density = float((routing > 0).float().mean().item())

        self.last_sparsity = 1.0 - self.last_routing_density

        if self.training:
            self.evolution_step(x.detach(), codes.detach())

        return self.o_proj(output)


# ---------------------------------------------------------------------------
# Stereographic_Attention_Layer_V2  (optimised STA operator)
# ---------------------------------------------------------------------------

class Stereographic_Attention_Layer_V2(nn.Module):
    """
    Optimised Stereographic Topological Attention (STA v2).

    Identical mathematical semantics to Stereographic_Attention_Layer in
    sta_core.py, with three hardware-level optimisations:

      1. No arccos: Shockwave mask is  raw_inner < cos_lambda  (pure scalar
         comparison, no transcendental function in the forward pass).
      2. Rank-8 projections for Q, K, V, O  (D->8->D bottleneck).
      3. torch.bmm for the [B,L,L] spherical inner product.

    Pipeline
    --------
    Input  [B, L, D]
        -> Rank8_Projection (q, k, v)
        -> inverse_stereo_project -> S^D
        -> SphericalTopologicalEncoding (Givens rotors)
        -> torch.bmm  -> raw_inner [B,L,L]           Opt 3
        -> conformal modulation  Omega(q)*Omega(k)   <- 球极残差调制
        -> algebraic shockwave mask  raw_inner < cos_lambda  Opt 1
        -> masked softmax
        -> context = bmm(attn, v)
        -> Rank8_Projection (output)
    Output [B, L, D]

    Args:
        hidden_dim:          D; flat feature dimension (must be >= 2)
        shockwave_threshold: Lambda in [0, pi]; geodesic distance cutoff.
                             Precomputed as cos_lambda = cos(Lambda).
        rank:                bottleneck rank for Rank8_Projection (default 8)
        max_seq_len:         maximum sequence length
        causal:              if True, apply an upper-triangular causal mask to
                             scores before shockwave truncation, enforcing the
                             time-arrow constraint (token i cannot attend to j>i).
    """

    def __init__(
        self,
        hidden_dim: int,
        shockwave_threshold: float = math.pi / 2,
        rank: int = 8,
        max_seq_len: int = 8192,
        causal: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sphere_dim = hidden_dim + 1
        self.causal     = causal

        # Opt 1: precompute cos(Lambda) -- no arccos in forward pass
        self.cos_lambda = math.cos(shockwave_threshold)

        # Opt 2: Rank-8 bottleneck projections
        self.q_proj = Rank8_Projection(hidden_dim, rank)
        self.k_proj = Rank8_Projection(hidden_dim, rank)
        self.v_proj = Rank8_Projection(hidden_dim, rank)
        self.o_proj = Rank8_Projection(hidden_dim, rank)

        # Topological position encoding (unchanged from v1)
        self.pos_enc = SphericalTopologicalEncoding(self.sphere_dim, max_seq_len)

        self.last_sparsity: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]  float32

        Returns:
            out: [B, L, D]  float32
        """
        B, L, D = x.shape

        # ── Rank-8 projections in flat space  (Opt 2) ────────────────────
        q = self.q_proj(x)   # [B, L, D]
        k = self.k_proj(x)   # [B, L, D]
        v = self.v_proj(x)   # [B, L, D]

        # ── Inverse stereographic projection: R^D -> S^D ─────────────────
        # ← NORTH POLE WORMHOLE: large-norm tokens collapse to eta~1
        q_s, omega_q = inverse_stereo_project(q)   # [B,L,D+1], [B,L,1]
        k_s, omega_k = inverse_stereo_project(k)   # [B,L,D+1], [B,L,1]

        # ── Topological position encoding (SU(2) rotors on S^D) ──────────
        positions = torch.arange(L, device=x.device, dtype=torch.long)
        q_s = self.pos_enc(q_s, positions)   # [B, L, D+1]
        k_s = self.pos_enc(k_s, positions)   # [B, L, D+1]

        # ── Spherical inner product via torch.bmm  (Opt 3) ───────────────
        # <s_q, s_k> in R^{D+1} = cos(geodesic); result in [-1, 1]
        raw_inner = torch.bmm(q_s, k_s.transpose(1, 2))   # [B, L, L]

        # ── Conformal residual modulation: Omega(q) * Omega(k) ───────────
        # <- 球极残差调制: amplitude-modulate scores by the conformal distortion
        conformal = omega_q * omega_k.transpose(1, 2)     # [B, L, L]  (broadcast)
        scores    = raw_inner * conformal                  # <- 球极残差调制 (this line)

        # ── Causal mask (time-arrow enforcement) ─────────────────────────────
        # Applied to scores (post-conformal) so the conformal factor cannot
        # leak information from future tokens across the causal boundary.
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )                                                          # True  => future token => zero
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # ── Algebraic Shockwave Truncation: raw_inner < cos(Lambda)  (Opt 1)
        # Mathematically: arccos(inner) > Lambda  <=>  inner < cos(Lambda)
        # cos is monotone-decreasing on [0,pi]; no transcendental ops needed.
        shock_mask = raw_inner < self.cos_lambda           # True => geodesic > Lambda => zero
        self.last_sparsity = shock_mask.float().mean().item()

        # Hard-zero: assign -inf so softmax gives exactly 0 for masked pairs
        scores = scores.masked_fill(shock_mask, float("-inf"))

        # ── Scale & masked softmax ────────────────────────────────────────
        scores       = scores / math.sqrt(self.sphere_dim)
        attn_weights = F.softmax(scores, dim=-1)           # [B, L, L]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # ── Value aggregation & output projection ─────────────────────────
        context = torch.bmm(attn_weights, v)               # [B, L, D]
        return self.o_proj(context)
