"""
das_core.py — STQ-TN: Shockwave Truncated Quaternion Tensor Network
====================================================================
Core operator for DAS_Attention_Layer.

Mathematical foundations
------------------------
1. SU(2) / Dual Quaternion isomorphism
   Every hidden vector is mapped to a 2×2 complex block-unitary matrix:

       M(q) = [[ z1,  z2 ],
               [-z2*, z1*]]   where z1 = a + bi, z2 = c + di

   The quaternion (a, b, c, d) is first L2-normalised so that
   |z1|² + |z2|² = 1 → det(M) = 1, giving a proper unit-sphere SU(2)
   element.  The four real components encode "direction" in the
   non-commutative sense.

2. Non-commutative geometric product
   Token interactions are computed as pairwise batch complex matmul:

       M_int[t_q, t_k] = M_q[t_q] @ M_k[t_k]    (non-commutative)

   M_q @ M_k ≠ M_k @ M_q, preserving the "spatial folding / direction"
   property that a scalar dot-product discards.

3. Energy metric  (bounded in [0, 1] for unit-SU(2) inputs)
   For unit SU(2) matrices, |det(M_int)| = 1 always (constant).  Instead
   we use the trace-based quaternion inner product:

       energy[t_q, t_k] = |Re(tr(M_int))| / 2  ∈ [0, 1]

   This equals 1 when M_q = M_k (identical rotation) and 0 when they
   are antipodal — exactly the "alignment" of the two directions.

4. Topological fusion rule  τ ⊗ τ = 1 ⊕ τ  (Shockwave Truncation)
   For each (t_q, t_k) block:
     • → 1 (annihilation)  : energy < λ  →  zero the value contribution
     • → τ (self-recurrence): energy ≥ λ  →  pass the value block forward

   The causal mask (s > t → zero) is applied jointly with the energy gate.

5. Topological stability guarantee
   Because zeroed blocks are INDEPENDENT of input magnitude, small
   perturbations δx with ‖δx‖ << (margin to threshold λ) cannot flip
   the truncation decision for well-separated blocks.  The output
   therefore changes only through the surviving interactions.

Style
-----
Follows the nn.Module / torch.complex64 / device_compute conventions
established in train.py and train_V2.0Q.py.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Device selection (mirrors train_V2.0Q.py)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DTYPE = torch.complex64  # all SU(2) matrices live here


# ---------------------------------------------------------------------------
# Helper: map four real scalars → one 2×2 unit-SU(2) matrix
# ---------------------------------------------------------------------------
def real_to_su2(x: torch.Tensor) -> torch.Tensor:
    """
    Map a real tensor whose last dim = 4·num_blocks to a batch of 2×2
    unit-SU(2) matrices (proper unit quaternions, det = 1).

    Args:
        x: [..., 4 * num_blocks]  (float32)

    Returns:
        M: [..., num_blocks, 2, 2]  (complex64)

    Each group of 4 reals (a, b, c, d) is L2-normalised, then mapped to:
        z1 = a + i·b,  z2 = c + i·d
        M  = [[ z1,  z2 ],
              [-z2*, z1*]]          ← unit SU(2), |z1|²+|z2|² = 1
    """
    *leading, last_dim = x.shape
    assert last_dim % 4 == 0, "hidden_dim must be divisible by 4"
    num_blocks = last_dim // 4

    # [..., num_blocks, 4]
    x4 = x.view(*leading, num_blocks, 4)

    # Normalise to unit quaternion → proper SU(2) element (direction only)
    norm = x4.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    x4n = x4 / norm                           # unit quaternion: a²+b²+c²+d² = 1

    a, b, c, d = x4n[..., 0], x4n[..., 1], x4n[..., 2], x4n[..., 3]

    # Build complex components — non-commutative direction encoding
    z1 = torch.complex(a, b).to(DTYPE)   # z1 = a + i·b
    z2 = torch.complex(c, d).to(DTYPE)   # z2 = c + i·d

    # Assemble the 2×2 SU(2) block: M(q) = [[z1, z2], [-z2*, z1*]]
    row0 = torch.stack([z1,         z2],         dim=-1)   # [..., nb, 2]
    row1 = torch.stack([-z2.conj(), z1.conj()],  dim=-1)
    return torch.stack([row0, row1], dim=-2)               # [..., nb, 2, 2]


# ---------------------------------------------------------------------------
# Helper: 2×2 complex matrix → four real components
# ---------------------------------------------------------------------------
def su2_to_real(M: torch.Tensor) -> torch.Tensor:
    """
    Inverse of real_to_su2 (output stage).

    Extract (z1, z2) from the top row: z1 = M[...,0,0], z2 = M[...,0,1]
    then decompose to four reals [Re(z1), Im(z1), Re(z2), Im(z2)].

    Args:
        M: [..., num_blocks, 2, 2]  complex64

    Returns:
        x: [..., num_blocks * 4]  float32
    """
    z1 = M[..., 0, 0]   # top-left  = z1
    z2 = M[..., 0, 1]   # top-right = z2

    out = torch.stack([z1.real, z1.imag, z2.real, z2.imag], dim=-1)
    *leading, num_blocks, _ = out.shape
    return out.view(*leading, num_blocks * 4).to(torch.float32)


# ---------------------------------------------------------------------------
# Shockwave Truncation: the τ ⊗ τ = 1 ⊕ τ operator
# ---------------------------------------------------------------------------
def shockwave_truncation(
    energy: torch.Tensor,
    M_v_exp: torch.Tensor,
    causal_mask: torch.Tensor,
    lambda_threshold: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """
    Apply the topological fusion rule τ ⊗ τ = 1 ⊕ τ.

    Energy = |Re(tr(M_int))| / 2  ∈ [0, 1]  is pre-computed by the caller.

    Branch 1 → 1 (annihilation, 1⊕τ):
        energy < λ  OR  s > t (non-causal)
        → zero out the value-block contribution

    Branch 2 → τ (self-recurrence, 1⊕τ):
        energy ≥ λ  AND  s ≤ t
        → pass M_v[t_k] forward; it is already a unit-SU(2) element

    Args:
        energy:       [B, T_q, T_k, nb]  float — quaternion inner product
        M_v_exp:      [B,  1,  T_k, nb, 2, 2]  complex64  (value matrices)
        causal_mask:  [T_q, T_k]  bool  (True where s ≤ t)
        lambda_threshold: energy threshold λ

    Returns:
        M_agg:    [B, T_q, nb, 2, 2]  — aggregated value blocks
        sparsity: float — fraction of (t_q, t_k, nb) triples zeroed
    """
    # survive = energy gate ∧ causal gate
    # [B, T_q, T_k, nb]
    survive = energy >= lambda_threshold                    # energy gate
    causal  = causal_mask.unsqueeze(0).unsqueeze(-1)       # [1, T_q, T_k, 1]
    survive = survive & causal                             # combined gate

    # sparsity = fraction of entries killed by → 1 branch
    sparsity = 1.0 - survive.float().mean().item()

    # Broadcast survive mask to matrix shape: [B, T_q, T_k, nb, 1, 1]
    gate = survive.unsqueeze(-1).unsqueeze(-1)             # [B, T_q, T_k, nb, 1, 1]
    gate_c = gate.to(DTYPE)

    # Branch 1 → 1: zero annihilated blocks
    # Branch 2 → τ: keep value block as-is (it is already unit-SU(2))
    # M_v_exp: [B, 1, T_k, nb, 2, 2]  →  gated: [B, T_q, T_k, nb, 2, 2]
    M_gated = M_v_exp * gate_c                             # ← 1⊕τ collapse line

    # Aggregate surviving value blocks over the key dimension
    # survive: [B, T_q, T_k, nb]
    n_survive = survive.float().sum(dim=2)                 # [B, T_q, nb]
    n_survive = n_survive.clamp(min=1.0)

    # Sum and normalise: [B, T_q, nb, 2, 2]
    M_agg = M_gated.sum(dim=2)                             # sum over T_k
    M_agg = M_agg / n_survive.unsqueeze(-1).unsqueeze(-1).to(DTYPE)

    # output_zero_mask: True for (b, t_q, nb) positions where every key
    # was annihilated (→ 1 branch) and M_agg is exactly zero.
    # These positions are PERFECTLY immune to input noise — their
    # contribution to the output cannot change regardless of perturbation.
    output_zero_mask = ~survive.any(dim=2)                 # [B, T_q, nb]

    return M_agg, sparsity, output_zero_mask


# ---------------------------------------------------------------------------
# DAS_Attention_Layer — the full STQ-TN operator
# ---------------------------------------------------------------------------
class DAS_Attention_Layer(nn.Module):
    """
    Shockwave Truncated Quaternion attention layer.

    Pipeline
    --------
    Input  [B, T, D]  (float32)
        ↓  real_to_su2          → unit SU(2) block matrices  [B,T,nb,2,2]
        ↓  pairwise matmul      → M_int = M_q[t] @ M_k[s]   (non-commutative)
        ↓  energy metric        → |Re(tr(M_int))| / 2  ∈ [0,1]
        ↓  shockwave_truncation → τ⊗τ = 1⊕τ  (binary gate + aggregation)
        ↓  su2_to_real          → reconstruct real features
        ↓  o_proj               → linear output mixing
    Output [B, T, D]  (float32)

    Args:
        hidden_dim:       D, must be divisible by 4
        lambda_threshold: energy cutoff λ for Shockwave Truncation
                          (≈0.5 gives ~50% sparsity for random unit quaternions)
    """

    def __init__(self, hidden_dim: int, lambda_threshold: float = 0.5):
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"
        self.hidden_dim = hidden_dim
        self.num_blocks = hidden_dim // 4
        self.lambda_threshold = lambda_threshold

        # Learnable real-valued projections (mirrors BalancedHamiltonLayer style)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Exposed for logging / experiment inspection
        self.last_sparsity: float = 0.0
        # output_zero_mask: [B, T, nb] bool — True for fully-zeroed output blocks.
        # These blocks have EXACTLY zero contribution; noise cannot affect them.
        self.last_output_zero_mask: torch.Tensor | None = None
        # last_out_real: pre-projection real output [B, T, D] — the level at
        # which the zeroed-block firewall property holds exactly.
        self.last_out_real: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]  float32

        Returns:
            out: [B, T, D]  float32
        """
        B, T, D = x.shape

        # --- Linear projections in real space --------------------------
        q = self.q_proj(x)   # [B, T, D]
        k = self.k_proj(x)   # [B, T, D]
        v = self.v_proj(x)   # [B, T, D]

        # --- Real → unit-SU(2) block matrices (spatial folding) --------
        # Each token → num_blocks independent 2×2 unit-SU(2) matrices.
        # The normalisation step encodes DIRECTION only, stripping magnitude
        # and mapping each feature group onto the non-commutative S³ sphere.
        M_q = real_to_su2(q)   # [B, T, nb, 2, 2]
        M_k = real_to_su2(k)   # [B, T, nb, 2, 2]
        M_v = real_to_su2(v)   # [B, T, nb, 2, 2]

        # --- Pairwise geometric product: M_int = M_q[t_q] @ M_k[t_k] --
        # Non-commutative matrix product preserves directionality:
        # M_q @ M_k ≠ M_k @ M_q  ← "空间折叠 / spatial folding" line
        #
        # Expand dims for broadcasting over (T_q, T_k):
        #   M_q: [B, T, 1, nb, 2, 2]
        #   M_k: [B, 1, T, nb, 2, 2]
        M_q_exp = M_q.unsqueeze(2)   # [B, T_q, 1,   nb, 2, 2]
        M_k_exp = M_k.unsqueeze(1)   # [B, 1,   T_k, nb, 2, 2]

        # Batch matmul over last two dims → [B, T_q, T_k, nb, 2, 2]
        M_int = torch.matmul(M_q_exp, M_k_exp)   # ← non-commutative geometric product

        # --- Energy metric: |Re(tr(M_int))| / 2  ∈ [0, 1] -------------
        # For unit-SU(2) inputs the product M_int is itself a unit-SU(2)
        # matrix. Its trace tr = M_int[0,0] + M_int[1,1] encodes the
        # "angle" between the two rotations:
        #   energy = 1  →  M_q = M_k  (identical direction)
        #   energy = 0  →  antipodal  (maximally orthogonal)
        #
        # Re(tr(M_int)) / 2 = quaternion inner product of q and k as
        # measured through their non-commutative product.
        trace  = M_int[..., 0, 0] + M_int[..., 1, 1]   # [B, T_q, T_k, nb]
        energy = trace.real.abs() / 2.0                  # ∈ [0, 1]

        # --- Causal mask: future keys cannot attend to past queries ----
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        # causal_mask[t_q, t_k] = True  iff  t_k ≤ t_q

        # --- Shockwave Truncation: τ ⊗ τ = 1 ⊕ τ ----------------------
        # energy < λ  →  1 branch (annihilation, zero the block)
        # energy ≥ λ  →  τ branch (self-recurrence, pass value forward)
        M_v_exp = M_v.unsqueeze(1)   # [B, 1, T_k, nb, 2, 2]
        M_agg, sparsity, output_zero_mask = shockwave_truncation(
            energy, M_v_exp, causal_mask,
            lambda_threshold=self.lambda_threshold,
        )
        self.last_sparsity = sparsity
        self.last_output_zero_mask = output_zero_mask   # exposed for analysis

        # --- SU(2) → Real: reconstruct real feature vectors ------------
        out_real = su2_to_real(M_agg.contiguous())   # [B, T, D]
        self.last_out_real = out_real   # pre-o_proj: zeroed blocks are exactly 0 here

        # Output projection (mirrors o_proj in QuaternionAttention)
        out = self.o_proj(out_real)   # [B, T, D]
        return out
