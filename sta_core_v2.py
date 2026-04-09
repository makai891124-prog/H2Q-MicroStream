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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """

    def __init__(
        self,
        hidden_dim: int,
        shockwave_threshold: float = math.pi / 2,
        rank: int = 8,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sphere_dim = hidden_dim + 1

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
