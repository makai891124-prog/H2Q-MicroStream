"""
sta_core.py -- STA: Stereographic Topological Attention
========================================================
A PyTorch operator that replaces the flat-Euclidean attention mechanism
with a full non-Euclidean pipeline on the N-dimensional unit hypersphere S^D.

Mathematical pipeline
---------------------
1. Inverse Stereographic Projection  R^D -> S^D
   Every flat feature vector x in R^D is lifted to a unit point on S^D:
       xi_i = 2*x_i / (|x|^2 + 1)          (flat -> sphere equatorial belt)
       eta   = (|x|^2 - 1) / (|x|^2 + 1)  (north/south axis: eta->1 as |x|->inf)
   All vectors with large Euclidean norm map to near the North Pole (eta~1)
   regardless of their sequence position: the "topological wormhole" that
   short-circuits long-range dependencies.

2. Topological Loci Encoding  (SU(2) Rotors on S^D)
   Position index t is encoded as a block-diagonal rotation
   R_t in SO(D+1) composed of Givens rotations in each pair of sphere
   dimensions (2k, 2k+1) with angle theta_{t,k} = t * omega_k.
   omega_k are learnable frequencies initialised like RoPE.

3. Spherical inner product & conformal residual modulation
   Attention score <- <s_q, s_k>_sphere * Omega(q) * Omega(k)
   <s_q, s_k> = s_q . s_k in R^{D+1} = cos(geodesic angle)
   Omega(x) = 2/(|x|^2+1): the conformal factor / observation residual.

4. Shockwave Truncation (hard threshold Lambda on geodesic distance)
   geodesic(i,j) = arccos(<s_i,s_j>) > Lambda  =>  score = -inf (hard zero)
   Eliminates topologically disconnected pairs before softmax.

5. Value aggregation & Forward Stereographic Projection  S^D -> R^D
   Masked softmax over surviving scores, aggregate real-space Value
   vectors, apply output projection.

Style
-----
Follows the nn.Module / DEVICE conventions from train_V2.0Q.py.
No new dependencies beyond existing torch / numpy in requirements.txt.
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
# Stereographic projection helpers
# ---------------------------------------------------------------------------

def inverse_stereo_project(x: torch.Tensor):
    """
    Lift flat feature vectors from R^D to the unit hypersphere S^D in R^{D+1}
    via inverse stereographic projection (pole at South Pole = (0,...,0,-1)).

    Formulae:
        r2   = ||x||^2
        xi_i = 2*x_i / (r2 + 1)      <- equatorial coords   [line A]
        eta  = (r2 - 1) / (r2 + 1)   <- polar axis          [line B]

    Property verified: sum(xi_i^2) + eta^2 = 1  (exactly on the unit sphere).

    ╔═══════════════════════════════════════════════════════════╗
    ║  NORTH POLE WORMHOLE (lines A+B):                        ║
    ║  As |x| -> inf, eta -> 1 and xi_i -> 0.                  ║
    ║  All large-norm tokens collapse to the same point        ║
    ║  (North Pole), giving chord-distance -> 0 between them   ║
    ║  regardless of their 1-D sequence separation.            ║
    ╚═══════════════════════════════════════════════════════════╝

    Args:
        x: [..., D]  float32

    Returns:
        s:     [..., D+1]  unit sphere point  (xi_1, ..., xi_D, eta)
        omega: [..., 1]    conformal factor   Omega(x) = 2/(r2+1) in (0,1]
    """
    r2 = x.pow(2).sum(dim=-1, keepdim=True)         # [..., 1]  squared norm
    denom = r2 + 1.0                                 # [..., 1]
    xi  = (2.0 * x) / denom                         # [..., D]  [line A]  xi_i = 2*x_i / (r2+1)
    eta = (r2 - 1.0) / denom                        # [..., 1]  [line B]  eta  = (r2-1)/(r2+1)
    s     = torch.cat([xi, eta], dim=-1)             # [..., D+1]  <- NORTH POLE SHORTCUT
    omega = 2.0 / denom                              # [..., 1]  conformal / residual factor
    return s, omega


def stereo_project(s: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Collapse a unit hypersphere point back to flat R^D via forward
    stereographic projection from the North Pole N = (0,...,0,1).

    Formula:  x_i = xi_i / (1 - eta)

    Args:
        s:   [..., D+1]  unit sphere point
        eps: numerical guard for eta -> 1 (North Pole singularity)

    Returns:
        x: [..., D]  flat space vector
    """
    xi  = s[..., :-1]   # [..., D]
    eta = s[..., -1:]   # [..., 1]
    return xi / (1.0 - eta + eps)


# ---------------------------------------------------------------------------
# Topological Loci Encoding
# ---------------------------------------------------------------------------

class SphericalTopologicalEncoding(nn.Module):
    """
    Position encoding on the unit hypersphere S^D using block-diagonal
    SO(D+1) rotations composed of 2-D Givens rotors -- the real-valued
    analogue of SU(2) rotors acting on the sphere.

    Each position t is mapped to a rotation R_t whose k-th Givens block
    (in sphere dimensions (2k, 2k+1)) uses angle:

        theta_{t,k} = t * omega_k

    where omega_k are learnable frequencies initialised like RoPE:
        omega_k = 1 / 10000^(k / num_pairs).

    Applying R_t "guides" the sphere point to a position-specific locus
    without leaving S^D (rotations preserve unit norm).

    Args:
        sphere_dim:  D+1 (number of real coordinates on S^D)
        max_seq_len: maximum supported sequence length
    """

    def __init__(self, sphere_dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.sphere_dim = sphere_dim
        self.num_pairs  = sphere_dim // 2   # number of 2-D rotation planes

        # Frequencies initialised like RoPE (log-spaced): omega_k = 1/10000^(k/num_pairs)
        # Registered as a persistent buffer (included in state_dict, not a trainable param)
        inv_freq = 1.0 / (
            10000.0 ** (
                torch.arange(0, self.num_pairs, dtype=torch.float32) / self.num_pairs
            )
        )
        self.register_buffer("inv_freq", inv_freq)   # [num_pairs], persistent buffer (saved in state_dict, not a gradient parameter)

    def forward(self, s: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Rotate sphere coordinates s by position-dependent Givens rotations.

        Args:
            s:         [B, L, D+1]  sphere coordinates
            positions: [L]          integer position indices 0..L-1

        Returns:
            s_rot: [B, L, D+1]  rotated sphere coordinates (unit norm preserved)
        """
        B, L, Sd = s.shape
        np_ = self.num_pairs

        # theta[t, k] = t * omega_k   [L, num_pairs]
        theta = torch.outer(positions.float(), self.inv_freq)   # [L, np_]
        cos_t = theta.cos()   # [L, np_]
        sin_t = theta.sin()   # [L, np_]

        # Index arrays for even/odd sphere dimensions in each pair
        even_idx = torch.arange(0, 2 * np_, 2, device=s.device)   # [np_]
        odd_idx  = even_idx + 1                                     # [np_]

        # Extract paired sphere dimensions
        s_even = s[:, :, even_idx]   # [B, L, np_]
        s_odd  = s[:, :, odd_idx]    # [B, L, np_]

        # Broadcast angles: [1, L, np_]
        c  = cos_t.unsqueeze(0)
        st = sin_t.unsqueeze(0)

        # Apply Givens rotation in each pair  (SU(2) rotor analogue)
        new_even = c * s_even - st * s_odd   # [B, L, np_]
        new_odd  = st * s_even + c * s_odd   # [B, L, np_]

        # Write rotated pairs back; remaining dims (if Sd is odd) are unchanged
        s_rot = s.clone()
        s_rot[:, :, even_idx] = new_even
        s_rot[:, :, odd_idx]  = new_odd
        return s_rot


# ---------------------------------------------------------------------------
# Stereographic_Attention_Layer  (full STA operator)
# ---------------------------------------------------------------------------

class Stereographic_Attention_Layer(nn.Module):
    """
    Stereographic Topological Attention (STA) layer.

    Pipeline
    --------
    Input  [B, L, D]  (float32)
        down  q_proj / k_proj / v_proj   -> Q, K, V  [B, L, D]
        down  inverse_stereo_project     -> Q_s, K_s  [B, L, D+1]  + Omega [B, L, 1]
        down  SphericalTopologicalEncod. -> Q_s, K_s with position-rotor applied
        down  spherical inner product    -> scores = Q_s @ K_s^T  [B, L, L]
        down  conformal modulation       -> scores *= Omega(q) * Omega(k)
        down  Shockwave Truncation       -> geodesic > Lambda => -inf
        down  masked softmax             -> attn_weights  [B, L, L]
        down  weighted sum of V          -> context  [B, L, D]
        down  o_proj                     -> output  [B, L, D]
    Output [B, L, D]  (float32)

    Args:
        hidden_dim:           D; flat feature dimension (must be >= 2)
        shockwave_threshold:  Lambda; geodesic distance cutoff in [0, pi]
                              Default pi/2 (= 1.5708): keep only the closer
                              hemisphere of interactions.
        max_seq_len:          maximum sequence length supported
    """

    def __init__(
        self,
        hidden_dim: int,
        shockwave_threshold: float = math.pi / 2,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.hidden_dim          = hidden_dim
        self.sphere_dim          = hidden_dim + 1        # S^D lives in R^{D+1}
        self.shockwave_threshold = shockwave_threshold

        # Learnable real-valued projections (mirrors BalancedHamiltonLayer style)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Topological position encoding on S^D
        self.pos_enc = SphericalTopologicalEncoding(self.sphere_dim, max_seq_len)

        # Exposed for logging / analysis
        self.last_sparsity: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]  float32

        Returns:
            out: [B, L, D]  float32
        """
        B, L, D = x.shape

        # ── Linear projections in flat space ────────────────────────────
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

        # ── Spherical inner product: <s_q, s_k> = cos(geodesic) ──────────
        # Standard dot product in R^{D+1} on the unit sphere equals cos(theta)
        raw_inner = torch.bmm(q_s, k_s.transpose(1, 2))   # [B, L, L] in [-1, 1]

        # ── Conformal residual modulation: Omega(q) * Omega(k) ───────────
        # ← SPHERICAL RESIDUAL AMPLITUDE MODULATION (observation distortion)
        omega_q_sq = omega_q.squeeze(-1)   # [B, L]
        omega_k_sq = omega_k.squeeze(-1)   # [B, L]
        conformal = omega_q_sq.unsqueeze(2) * omega_k_sq.unsqueeze(1)   # [B,L,L]
        scores = raw_inner * conformal      # <- "球极残差调制" (conformal modulation line)

        # ── Shockwave Truncation: geodesic > Lambda => hard zero ──────────
        # Geodesic angle theta = arccos(<s_i, s_j>); clamp to valid arccos range
        geodesic = torch.acos(raw_inner.clamp(-1.0 + 1e-7, 1.0 - 1e-7))   # [B,L,L]
        shock_mask = geodesic <= self.shockwave_threshold                   # True=keep

        # Record sparsity (fraction of pairs zeroed by Shockwave Truncation)
        self.last_sparsity = 1.0 - shock_mask.float().mean().item()

        # Apply hard-zero mask via -inf so softmax assigns exactly 0 probability
        scores = scores.masked_fill(~shock_mask, float("-inf"))             # hard zero

        # ── Scale & masked softmax ────────────────────────────────────────
        scale  = math.sqrt(self.sphere_dim)
        scores = scores / scale
        attn_weights = F.softmax(scores, dim=-1)             # [B, L, L]
        # Guard: rows where every key was masked become NaN -> set to 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # ── Value aggregation ─────────────────────────────────────────────
        context = torch.bmm(attn_weights, v)   # [B, L, D]

        # ── Output projection ─────────────────────────────────────────────
        out = self.o_proj(context)
        return out
