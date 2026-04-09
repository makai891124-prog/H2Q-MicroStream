"""
autonomous_evolution_daemon.py — Autonomous Evolution Daemon
=============================================================
Upgrades rolling_horizon_eval.py to an industrial-grade daemon capable of
T → ∞ (infinite-time) operation with no human intervention.

Three guarantees are continuously asserted:

  1. VRAM Topological Lockdown
       Delta_VRAM  := current_VRAM − initial_VRAM  must stay < 1 MB.
       Strict garbage collection after every chunk: empty_cache + graph-ref
       deletion.  Validation phase is isolated under torch.no_grad() so zero
       gradient accumulation can leak into the EMA.

  2. Dynamic Homology Checkpointing
       Model snapshots are saved on "topological phase-transition" events,
       NOT on a fixed epoch schedule.  A phase transition is detected when:
         (a) the moving-window slope of Causal_Loss → 0   (plateau condition)
         AND
         (b) Topo_Sparsity shows a step-change exceeding SPARSITY_STEP_THRESH.
       Each checkpoint is saved as a timestamped .pt file (physical invariant
       snapshot).

  3. Asymptotic Metric Monitor
       Delta_VRAM    — memory leak detector (must be < 1 MB)
       SVD_Entropy   — Shannon entropy of singular-value energy distribution
                       across all Rank-8 projection matrices; guards against
                       white-noise collapse (all singular values equal)
       Loss_Asymptote — asymptotic limit fitted to the last ASYMPTOTE_WINDOW
                        chunks via a simple linear regression on 1/step

Usage:
    python autonomous_evolution_daemon.py --source path/to/stream.txt

    # Fixed step count (useful for 48-hour pressure tests):
    python autonomous_evolution_daemon.py --source stream.txt --steps 200000

    # Auto-download TinyStories if no source given:
    python autonomous_evolution_daemon.py
"""

import argparse
import gc
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import torch
import torch.nn.functional as F

from h2q_evolution import H2Q_Evolution_Engine

# ─── Configuration ────────────────────────────────────────────────────────────

VOCAB_SIZE  = 256
DIM         = 128
NUM_LAYERS  = 4
RANK        = 8
SEQ_LEN     = 1024
BATCH_SIZE  = 1

LR           = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0

SHOCKWAVE_THRESHOLD = math.pi / 2

PRINT_EVERY       = 50          # print metrics every N steps
EMA_ALPHA         = 0.99        # EMA decay for Causal_Loss smoothing
CHECKPOINT_DIR    = "checkpoints_daemon"

# ── Phase-transition detection ────────────────────────────────────────────────
SLOPE_WINDOW          = 200    # steps used to estimate the Loss slope
SLOPE_NEAR_ZERO_THRESH = 1e-4  # |slope| below this → plateau condition met
SPARSITY_STEP_THRESH  = 0.02   # step-change in Topo_Sparsity that counts
SPARSITY_EMA_ALPHA    = 0.95   # EMA for smooth Topo_Sparsity baseline

# ── Asymptotic metric windows ─────────────────────────────────────────────────
ASYMPTOTE_WINDOW   = 1000   # chunks used to fit Loss_Asymptote
SVD_CHECK_EVERY    = 500    # steps between SVD_Entropy computations
VRAM_ASSERT_THRESH = 1e6    # 1 MB — Delta_VRAM hard assertion limit (bytes)

# ── Fallback download ─────────────────────────────────────────────────────────
FALLBACK_URL  = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-train.txt"
)
FALLBACK_PATH = os.path.join("data_tinystories", "TinyStories-train.txt")

# ─── Device ───────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _vram_bytes() -> int:
    """Return GPU memory allocated in bytes, or 0 on non-CUDA devices."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0


def _vram_str(bytes_: int) -> str:
    return f"{bytes_ / 1e9:.4f} GB" if torch.cuda.is_available() else "N/A"


def _download_fallback(url: str, dest: str) -> None:
    try:
        import requests
    except ImportError:
        print("[daemon] ERROR: 'requests' package required for auto-download.")
        print("  Install with:  pip install requests")
        sys.exit(1)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[daemon] Downloading fallback dataset → {dest}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                sys.stdout.write(".")
                sys.stdout.flush()
    print("\n[daemon] Download complete.")


# ─── Step 3 helper: SVD Entropy ───────────────────────────────────────────────

def _compute_svd_entropy(model: H2Q_Evolution_Engine) -> float:
    """
    Compute the Shannon entropy of the singular-value energy distribution
    across all Rank-8 projection matrices in the model.

    For each Rank8_Projection we have two Linear layers (down/up).  We form
    the effective (dim × dim) product matrix  W = W_up @ W_down  and take its
    SVD.  Singular-value energy is  sigma_i^2 / sum(sigma^2).

    A value near  log2(rank)  (≈ 3 bits for rank=8) means all singular values
    carry equal energy → white-noise collapse.
    A value near 0 means energy is concentrated in a single eigenmode.
    Healthy convergence sits between these extremes with a stable, sub-maximal
    entropy.

    Returns:
        float  Shannon entropy in bits (log2 base)
    """
    entropies = []
    with torch.no_grad():
        for block in model.blocks:
            for ff_proj in (block.ff.up, block.ff.down):
                # Each Rank8_Projection.proj is nn.Sequential(Linear, Linear)
                w_down = ff_proj.proj[0].weight  # [rank, dim]
                w_up   = ff_proj.proj[1].weight  # [dim,  rank]
                # Effective projection: [dim, dim]
                w_eff  = w_up @ w_down            # [dim, rank] × [rank, dim]
                # SVD — we only need singular values
                sv = torch.linalg.svdvals(w_eff)  # [dim] — singular values of [dim, dim] effective matrix
                energy = sv.pow(2)
                total  = energy.sum()
                if total > 0:
                    p = energy / total
                    # Clip to avoid log(0)
                    p = p.clamp(min=1e-12)
                    h = -(p * torch.log2(p)).sum().item()
                    entropies.append(h)
    return float(sum(entropies) / len(entropies)) if entropies else 0.0


# ─── Step 3 helper: Loss Asymptote ────────────────────────────────────────────

def _fit_asymptote(loss_history: deque) -> float:
    """
    Fit the asymptotic limit of the loss curve using the last
    ASYMPTOTE_WINDOW raw (non-EMA) causal loss values.

    Model:  L(t) ≈ C + a / t
    Linearise:  L(t) − L̄  ≈  a · (1/t − 1/t̄)
    The intercept C is estimated as  L̄ − a · (1/t̄).

    When fewer than 2 points are available, returns the current mean.

    Returns:
        float  estimated asymptote C
    """
    n = len(loss_history)
    if n < 2:
        return sum(loss_history) / max(n, 1)

    # x_i = 1 / (relative step index from 1..n)
    xs = [1.0 / (i + 1) for i in range(n)]
    ys = list(loss_history)

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    num   = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, ys))
    denom = sum((xi - x_mean) ** 2 for xi in xs)

    if abs(denom) < 1e-15:
        return y_mean

    a = num / denom
    c = y_mean - a * x_mean
    return c


# ─── Step 2 helper: Phase-Transition Detector ─────────────────────────────────

class PhaseTransitionDetector:
    """
    Detects a topological phase transition using two concurrent signals:

      (a) Loss slope:  linear regression over the last SLOPE_WINDOW raw losses.
          |slope| < SLOPE_NEAR_ZERO_THRESH  → plateau.

      (b) Sparsity step: EMA of Topo_Sparsity is tracked; a step-change
          (|current − EMA| > SPARSITY_STEP_THRESH) signals a phase transition.

    A checkpoint is triggered when BOTH conditions fire simultaneously.
    The detector enforces a minimum gap of SLOPE_WINDOW steps between two
    consecutive checkpoints to avoid bursting saves.
    """

    def __init__(self):
        self._loss_window    = deque(maxlen=SLOPE_WINDOW)
        self._sparsity_ema   = None
        self._last_ckpt_step = -SLOPE_WINDOW  # allow first detection immediately

    def update(self, step: int, loss: float, sparsity: float) -> bool:
        """
        Feed the latest metrics.  Returns True if a phase transition is detected
        and a checkpoint should be saved NOW.
        """
        self._loss_window.append(loss)

        # Update Sparsity EMA
        if self._sparsity_ema is None:
            self._sparsity_ema = sparsity
        prev_ema = self._sparsity_ema
        self._sparsity_ema = (
            SPARSITY_EMA_ALPHA * self._sparsity_ema
            + (1.0 - SPARSITY_EMA_ALPHA) * sparsity
        )

        # Need a full window before we can estimate slope
        if len(self._loss_window) < SLOPE_WINDOW:
            return False

        # Enforce minimum gap between checkpoints
        if step - self._last_ckpt_step < SLOPE_WINDOW:
            return False

        # (a) Estimate loss slope via linear regression over the window
        n  = len(self._loss_window)
        xs = list(range(n))
        ys = list(self._loss_window)
        x_mean = (n - 1) / 2.0
        y_mean = sum(ys) / n
        num    = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denom  = sum((x - x_mean) ** 2 for x in xs)
        slope  = (num / denom) if abs(denom) > 1e-15 else 0.0

        plateau_cond  = abs(slope) < SLOPE_NEAR_ZERO_THRESH

        # (b) Sparsity step-change
        sparsity_step_cond = abs(sparsity - prev_ema) > SPARSITY_STEP_THRESH

        if plateau_cond and sparsity_step_cond:
            self._last_ckpt_step = step
            return True

        return False


# ─── Autonomous Evolution Daemon ──────────────────────────────────────────────

class AutonomousEvolutionDaemon:
    """
    Industrial-grade daemon for infinite-time self-evolution of H2Q_Evolution_Engine.

    Implements Rolling Horizon Causal Validation (predict-then-learn) with:
      * VRAM Topological Lockdown (Step 1)
      * Dynamic Homology Checkpointing (Step 2)
      * Asymptotic Metric Monitor (Step 3)

    Args:
        source_path: path to a file used as the raw byte stream
    """

    def __init__(self, source_path: str):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = H2Q_Evolution_Engine(
            dim=DIM,
            num_layers=NUM_LAYERS,
            rank=RANK,
            shockwave_threshold=SHOCKWAVE_THRESHOLD,
            max_seq_len=SEQ_LEN,
        ).to(DEVICE)

        # ── Optimiser ─────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        # ── Diagnostics ───────────────────────────────────────────────────────
        param_count = self.model.count_parameters()
        param_mb    = self.model.parameter_size_mb()
        print("═" * 70)
        print("  Autonomous Evolution Daemon — H2Q_Evolution_Engine  T→∞")
        print("═" * 70)
        print(f"  Parameters      : {param_count:,}  ({param_mb:.2f} MB)")
        print(f"  Device          : {DEVICE}")
        print(
            f"  dim={DIM}  layers={NUM_LAYERS}  rank={RANK}  seq_len={SEQ_LEN}"
        )
        print(f"  Checkpoint dir  : {CHECKPOINT_DIR}/")
        print(f"  VRAM assertion  : Delta_VRAM < {VRAM_ASSERT_THRESH/1e6:.1f} MB")
        print("═" * 70)
        print()

        # ── Step 1: Record initial VRAM baseline ─────────────────────────────
        # Force a stable baseline after model allocation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._initial_vram: int = _vram_bytes()

        # ── Byte stream ───────────────────────────────────────────────────────
        self._file       = open(source_path, "rb")
        self._stream_pos = 0

        # ── EMA of Causal_Loss ────────────────────────────────────────────────
        self._ema_loss: float | None = None

        # ── Step 3: raw loss ring-buffer for asymptote fitting ────────────────
        self._raw_loss_history: deque = deque(maxlen=ASYMPTOTE_WINDOW)

        # ── Step 2: Phase-transition detector ─────────────────────────────────
        self._phase_detector = PhaseTransitionDetector()
        self._checkpoint_count = 0

    # ── Byte stream ───────────────────────────────────────────────────────────

    def _next_chunk(self) -> torch.Tensor:
        needed = SEQ_LEN + 1
        raw    = self._file.read(needed)
        if len(raw) < needed:
            self._file.seek(0)
            self._stream_pos = 0
            raw = raw + self._file.read(needed - len(raw))
        self._stream_pos += needed
        return torch.tensor(list(raw), dtype=torch.long, device=DEVICE)

    # ── Step 2: Save checkpoint ───────────────────────────────────────────────

    def _save_checkpoint(self, step: int, ema_loss: float, sparsity: float) -> str:
        ts   = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = f"phase_snapshot_{ts}_step{step:08d}.pt"
        path = os.path.join(CHECKPOINT_DIR, name)
        torch.save(
            {
                "step":        step,
                "ema_loss":    ema_loss,
                "sparsity":    sparsity,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            },
            path,
        )
        self._checkpoint_count += 1
        return path

    # ── Step 3: Delta_VRAM assertion ──────────────────────────────────────────

    def _assert_vram_invariant(self, step: int) -> int:
        """
        Assert Delta_VRAM < VRAM_ASSERT_THRESH.
        Logs a warning (does NOT abort) so the daemon can continue running
        while the operator investigates.
        Returns Delta_VRAM in bytes.
        """
        delta = _vram_bytes() - self._initial_vram
        if delta >= VRAM_ASSERT_THRESH:
            print(
                f"\n[daemon][WARN] step={step}  "
                f"Delta_VRAM={delta/1e6:.3f} MB  "
                f"EXCEEDS {VRAM_ASSERT_THRESH/1e6:.1f} MB threshold! "
                f"Possible memory leak detected."
            )
        return delta

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 0) -> None:
        """
        Execute the autonomous T→∞ Rolling Horizon Causal Validation loop.

        Args:
            max_steps: 0 = run indefinitely until KeyboardInterrupt.
        """
        header = (
            f"{'Step':>8}  "
            f"{'Stream_Pos':>12}  "
            f"{'Causal_Loss':>12}  "
            f"{'Bits/byte':>10}  "
            f"{'Topo_Sparsity':>14}  "
            f"{'Delta_VRAM':>12}  "
            f"{'VRAM':>12}"
        )
        print(header)
        print("─" * len(header))

        step    = 0
        t_start = time.time()

        # Cached SVD entropy (recomputed every SVD_CHECK_EVERY steps)
        svd_entropy: float = 0.0

        try:
            while True:
                if max_steps > 0 and step >= max_steps:
                    break

                # ══════════════════════════════════════════════════════════════
                # ROLLING HORIZON BOUNDARY — step T
                # ══════════════════════════════════════════════════════════════

                chunk = self._next_chunk()
                x = chunk[:SEQ_LEN].unsqueeze(0)    # [1, SEQ_LEN]
                y = chunk[1:SEQ_LEN + 1].unsqueeze(0)  # [1, SEQ_LEN]

                # ──────────────────────────────────────────────────────────────
                # STEP 1-A: EVALUATE under strict no_grad isolation
                #   No gradient is allowed to accumulate in the eval phase.
                # ──────────────────────────────────────────────────────────────
                self.model.eval()
                with torch.no_grad():
                    logits, _ = self.model(x)
                    causal_loss = F.cross_entropy(
                        logits.view(-1, VOCAB_SIZE),
                        y.view(-1),
                    ).item()
                # ── Causal_Loss recorded on unseen data ───────────────────────

                # Free logits tensor immediately (no backward needed from eval)
                del logits

                # EMA update
                if self._ema_loss is None:
                    self._ema_loss = causal_loss
                else:
                    self._ema_loss = (
                        EMA_ALPHA * self._ema_loss
                        + (1.0 - EMA_ALPHA) * causal_loss
                    )
                self._raw_loss_history.append(causal_loss)

                # ──────────────────────────────────────────────────────────────
                # STEP 1-B: EVOLVE — gradient update
                # ──────────────────────────────────────────────────────────────
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)  # zero & free grad buffers
                _, train_loss = self.model(x, targets=y)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

                # ── STEP 1-C: VRAM Topological Lockdown ──────────────────────
                # Explicitly delete all non-leaf computation graph references
                # and trigger CUDA memory compaction after every chunk.
                del train_loss, x, y, chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                step += 1

                # ── Topo_Sparsity (cheap — just reads cached last_sparsity) ──
                topo_sparsity = self.model.get_topology_sparsity()

                # ──────────────────────────────────────────────────────────────
                # STEP 2: Dynamic Homology Checkpointing
                # ──────────────────────────────────────────────────────────────
                if self._phase_detector.update(step, causal_loss, topo_sparsity):
                    ckpt_path = self._save_checkpoint(
                        step, self._ema_loss, topo_sparsity
                    )
                    print(
                        f"\n[daemon][PHASE] step={step:,}  "
                        f"Phase transition detected!  "
                        f"Snapshot #{self._checkpoint_count} → {ckpt_path}"
                    )

                # ──────────────────────────────────────────────────────────────
                # STEP 3-A: SVD_Entropy (periodic)
                # ──────────────────────────────────────────────────────────────
                if step % SVD_CHECK_EVERY == 0:
                    svd_entropy = _compute_svd_entropy(self.model)

                # ──────────────────────────────────────────────────────────────
                # Print metrics every PRINT_EVERY steps
                # ──────────────────────────────────────────────────────────────
                if step % PRINT_EVERY == 0:
                    # STEP 3-B: Delta_VRAM assertion
                    delta_vram = self._assert_vram_invariant(step)

                    # STEP 3-C: Loss_Asymptote
                    loss_asymptote = _fit_asymptote(self._raw_loss_history)

                    bits_per_byte  = self._ema_loss / math.log(2)
                    elapsed        = time.time() - t_start
                    steps_per_sec  = step / max(elapsed, 1e-6)

                    print(
                        f"{step:>8}  "
                        f"{self._stream_pos:>12,}  "
                        f"{self._ema_loss:>12.4f}  "
                        f"{bits_per_byte:>10.4f}  "
                        f"{topo_sparsity * 100:>13.1f}%  "
                        f"{delta_vram/1e6:>+10.3f}MB  "
                        f"{_vram_str(_vram_bytes()):>12}  "
                        f"svd_H={svd_entropy:.3f}bits  "
                        f"asymptote={loss_asymptote:.4f}  "
                        f"({steps_per_sec:.1f} step/s)"
                    )

        except KeyboardInterrupt:
            print(f"\n[daemon] Interrupted at step {step}.")

        finally:
            self._file.close()
            elapsed = time.time() - t_start
            print()
            print("═" * 70)
            print(f"  Daemon completed {step:,} evolution steps in {elapsed:.1f}s")
            print(f"  Checkpoints saved : {self._checkpoint_count}")
            if self._ema_loss is not None:
                final_bits = self._ema_loss / math.log(2)
                print(f"  Final Causal_Loss (EMA) : {self._ema_loss:.4f}")
                print(f"  Final Bits/byte         : {final_bits:.4f}")
                print(
                    f"  Final Topo_Sparsity     : "
                    f"{self.model.get_topology_sparsity() * 100:.1f}%"
                )
                if len(self._raw_loss_history) >= 2:
                    print(
                        f"  Loss_Asymptote (fitted) : "
                        f"{_fit_asymptote(self._raw_loss_history):.4f}"
                    )
                final_svd = _compute_svd_entropy(self.model)
                print(f"  SVD_Entropy (final)     : {final_svd:.4f} bits")
                delta_final = _vram_bytes() - self._initial_vram
                print(f"  Delta_VRAM (final)      : {delta_final/1e6:+.4f} MB")
            print("═" * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous Evolution Daemon — H2Q_Evolution_Engine  T→∞\n"
            "Rolling Horizon Causal Validation with VRAM lockdown, "
            "phase-transition checkpointing, and asymptotic metric monitoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Path to a UTF-8 text or binary file used as the byte stream. "
            "If omitted, TinyStories-train.txt is auto-downloaded."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum evolution steps (default 0 = run until KeyboardInterrupt).",
    )
    args = parser.parse_args()

    source = args.source
    if source is None:
        if not os.path.exists(FALLBACK_PATH):
            _download_fallback(FALLBACK_URL, FALLBACK_PATH)
        source = FALLBACK_PATH

    if not os.path.isfile(source):
        print(f"[daemon] ERROR: source file not found: {source}")
        sys.exit(1)

    file_size_mb = os.path.getsize(source) / 1e6
    print(f"[daemon] Byte stream source : {source}  ({file_size_mb:.1f} MB)")
    print(f"[daemon] Device             : {DEVICE}")
    print()

    daemon = AutonomousEvolutionDaemon(source)
    daemon.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
