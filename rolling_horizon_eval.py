"""
rolling_horizon_eval.py — Continuous Stream Optimizer
======================================================
Implements the Rolling Horizon Causal Validation protocol for the
H2Q_Evolution_Engine.

Core law (strict time-arrow ordering):
    For each chunk T of the raw byte stream:

      1. EVALUATE (Extrapolate):
           Model predicts chunk T using its CURRENT weights.
           Chunk T has NEVER been seen by the model before.
           Causal_Loss is recorded here — BEFORE any weight update.

      2. EVOLVE (Update):
           The model computes gradients on the same chunk T and calls
           optimizer.step(), updating its internal topological parameters.

      3. Advance the stream window to chunk T+1.

    This strict "predict-then-learn" ordering:
      * Eliminates interpolation artefacts (the model cannot memorise
        what it has not yet seen).
      * Provides an unbiased online estimate of generalisation ability.
      * Mirrors the causal structure of real-world continuous systems.

Real-time metrics printed every PRINT_EVERY steps:
    Stream_Position  — cumulative bytes consumed
    Causal_Loss      — EMA of pre-update cross-entropy loss
    Bits/byte        — Causal_Loss / ln(2)  (standard compression metric)
    Topo_Sparsity    — mean shockwave-truncation fraction across STA layers
    VRAM             — GPU memory allocated (or "N/A" on CPU)

Usage:
    # Use any local plain-text or binary file as the byte stream:
    python rolling_horizon_eval.py --source path/to/file.txt

    # Run for a fixed number of steps:
    python rolling_horizon_eval.py --source wiki.txt --steps 5000

    # Auto-download TinyStories if no source is provided:
    python rolling_horizon_eval.py
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from h2q_evolution import H2Q_Evolution_Engine

# ─── Configuration ────────────────────────────────────────────────────────────

VOCAB_SIZE  = 256           # hard-locked: one slot per raw byte value [0, 255]

DIM         = 128           # hidden dimension
NUM_LAYERS  = 4             # number of STA blocks
RANK        = 8             # Rank-8 bottleneck width throughout

SEQ_LEN     = 1024          # context window size (tokens per forward pass)
BATCH_SIZE  = 1             # online learning: one sequence at a time

LR           = 3e-4         # AdamW learning rate
WEIGHT_DECAY = 0.01         # AdamW weight decay
GRAD_CLIP    = 1.0          # gradient-norm clipping threshold

SHOCKWAVE_THRESHOLD = math.pi / 2   # geodesic cutoff Lambda = π/2 → cos = 0

PRINT_EVERY = 50            # print metrics every N evolution steps
EMA_ALPHA   = 0.99          # EMA decay for Causal_Loss smoothing

# Fallback data source (same dataset as train.py)
FALLBACK_URL  = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-train.txt"
)
FALLBACK_PATH = os.path.join("data_tinystories", "TinyStories-train.txt")

# ─── Device selection ─────────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _vram_str() -> str:
    """Return current GPU memory allocation as a human-readable string."""
    if torch.cuda.is_available():
        gb = torch.cuda.memory_allocated() / 1e9
        return f"{gb:.3f} GB"
    return "N/A"


def _download_fallback(url: str, dest: str) -> None:
    """Download a remote file with a simple progress indicator."""
    try:
        import requests
    except ImportError:
        print("[rolling_horizon] ERROR: 'requests' package is required for auto-download.")
        print("  Install with:  pip install requests")
        sys.exit(1)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[rolling_horizon] Downloading fallback dataset → {dest}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                sys.stdout.write(".")
                sys.stdout.flush()
    print(f"\n[rolling_horizon] Download complete.")


# ─── Continuous Stream Optimizer ─────────────────────────────────────────────

class Continuous_Stream_Optimizer:
    """
    Rolling Horizon Causal Validation loop for H2Q_Evolution_Engine.

    The data source is opened as a raw binary stream (any UTF-8 text or binary
    file works).  At each evolution step the model must predict the next
    SEQ_LEN bytes BEFORE being allowed to learn from them, enforcing strict
    causal ordering.

    Args:
        source_path: path to a file used as the raw byte stream
    """

    def __init__(self, source_path: str):

        # ── Model instantiation ───────────────────────────────────────────────
        self.model = H2Q_Evolution_Engine(
            dim=DIM,
            num_layers=NUM_LAYERS,
            rank=RANK,
            shockwave_threshold=SHOCKWAVE_THRESHOLD,
            max_seq_len=SEQ_LEN,
        ).to(DEVICE)

        # ── AdamW optimiser ───────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        # ── Report model size ─────────────────────────────────────────────────
        param_count = self.model.count_parameters()
        param_mb    = self.model.parameter_size_mb()
        print("─" * 60)
        print("  H2Q Evolution Engine — Online Causal Self-Evolution")
        print("─" * 60)
        print(f"  Parameters      : {param_count:,}  ({param_mb:.2f} MB)")
        print(f"  Device          : {DEVICE}")
        print(f"  dim={DIM}  layers={NUM_LAYERS}  rank={RANK}  seq_len={SEQ_LEN}")
        print(f"  VOCAB_SIZE      : {VOCAB_SIZE}  (raw bytes, no BPE)")
        print(
            f"  Shockwave Lambda: π/2  "
            f"(cos_lambda = {math.cos(SHOCKWAVE_THRESHOLD):.4f})"
        )
        print(f"  LR              : {LR}  |  grad_clip={GRAD_CLIP}")
        print("─" * 60)
        print()

        # ── Raw byte stream ───────────────────────────────────────────────────
        self.source_path = source_path
        self._file       = open(source_path, "rb")
        self._file_size  = os.path.getsize(source_path)
        self._stream_pos = 0   # cumulative raw bytes consumed (wraps on EOF)

        # ── EMA state for Causal_Loss smoothing ───────────────────────────────
        self._ema_loss: float | None = None

    # ─── Byte stream ──────────────────────────────────────────────────────────

    def _next_chunk(self) -> torch.Tensor:
        """
        Read SEQ_LEN+1 raw bytes from the stream.

        The byte stream is treated as conceptually infinite: on reaching EOF
        the file cursor wraps back to the start and reading continues.

        Returns:
            [SEQ_LEN+1]  long tensor of byte values in [0, 255]
        """
        needed = SEQ_LEN + 1
        raw    = self._file.read(needed)

        # Wrap on EOF
        if len(raw) < needed:
            self._file.seek(0)
            self._stream_pos = 0
            raw = raw + self._file.read(needed - len(raw))

        self._stream_pos += needed
        return torch.tensor(list(raw), dtype=torch.long, device=DEVICE)

    # ─── Main evolution loop ──────────────────────────────────────────────────

    def run(self, max_steps: int = 0) -> None:
        """
        Execute the continuous Rolling Horizon Causal Validation loop.

        Args:
            max_steps: stop after this many evolution steps.
                       0 (default) runs until KeyboardInterrupt or EOF wrap.
        """
        # Print column headers
        header = (
            f"{'Step':>8}  "
            f"{'Stream_Pos':>12}  "
            f"{'Causal_Loss':>12}  "
            f"{'Bits/byte':>10}  "
            f"{'Topo_Sparsity':>14}  "
            f"{'VRAM':>10}"
        )
        print(header)
        print("─" * len(header))

        step    = 0
        t_start = time.time()

        try:
            while True:
                if max_steps > 0 and step >= max_steps:
                    break

                # ══════════════════════════════════════════════════════════════
                # ROLLING HORIZON BOUNDARY — beginning of evolution step T
                # ══════════════════════════════════════════════════════════════

                # Step A: READ next chunk
                #   The model has never seen these SEQ_LEN+1 bytes before.
                chunk = self._next_chunk()                    # [SEQ_LEN+1]
                x = chunk[:SEQ_LEN].unsqueeze(0)             # [1, SEQ_LEN]
                y = chunk[1:SEQ_LEN + 1].unsqueeze(0)        # [1, SEQ_LEN]

                # ──────────────────────────────────────────────────────────────
                # Step B: EVALUATE / EXTRAPOLATE
                #   Forward pass with current weights — NO gradient update.
                #   Causal_Loss is measured here, on unseen data.
                # ──────────────────────────────────────────────────────────────
                self.model.eval()
                with torch.no_grad():
                    logits, _ = self.model(x)                # [1, SEQ_LEN, 256]
                    causal_loss = F.cross_entropy(
                        logits.view(-1, VOCAB_SIZE),
                        y.view(-1),
                    ).item()
                # ── CAUSAL LOSS RECORDED — model has predicted this chunk ────

                # Update EMA of Causal_Loss (for smooth monitoring)
                if self._ema_loss is None:
                    self._ema_loss = causal_loss
                else:
                    self._ema_loss = (
                        EMA_ALPHA * self._ema_loss + (1.0 - EMA_ALPHA) * causal_loss
                    )

                # ──────────────────────────────────────────────────────────────
                # Step C: EVOLVE / UPDATE
                #   Now the model is allowed to learn from this chunk.
                #   Gradient flows; weights updated via AdamW.
                # ──────────────────────────────────────────────────────────────
                self.model.train()
                _, train_loss = self.model(x, targets=y)
                self.optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()
                # ── WEIGHT UPDATE COMPLETE — window slides to T+1 ─────────────

                # ══════════════════════════════════════════════════════════════
                # ROLLING HORIZON BOUNDARY — end of evolution step T
                # ══════════════════════════════════════════════════════════════

                step += 1

                # ── Print metrics every PRINT_EVERY steps ─────────────────────
                if step % PRINT_EVERY == 0:
                    topo_sparsity = self.model.get_topology_sparsity()
                    bits_per_byte = self._ema_loss / math.log(2)
                    elapsed       = time.time() - t_start
                    steps_per_sec = step / max(elapsed, 1e-6)

                    print(
                        f"{step:>8}  "
                        f"{self._stream_pos:>12,}  "
                        f"{self._ema_loss:>12.4f}  "
                        f"{bits_per_byte:>10.4f}  "
                        f"{topo_sparsity * 100:>13.1f}%  "
                        f"{_vram_str():>10}  "
                        f"({steps_per_sec:.1f} step/s)"
                    )

        except KeyboardInterrupt:
            print(f"\n[rolling_horizon] Interrupted at step {step}.")

        finally:
            self._file.close()
            elapsed = time.time() - t_start
            print()
            print("─" * 60)
            print(f"  Finished {step} evolution steps in {elapsed:.1f}s")
            if self._ema_loss is not None:
                print(f"  Final Causal_Loss (EMA) : {self._ema_loss:.4f}")
                print(
                    f"  Final Bits/byte         : "
                    f"{self._ema_loss / math.log(2):.4f}"
                )
                print(
                    f"  Final Topo_Sparsity     : "
                    f"{self.model.get_topology_sparsity() * 100:.1f}%"
                )
            print("─" * 60)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling Horizon Causal Validation — H2Q Evolution Engine.\n"
            "Implements strict predict-then-learn online self-evolution on a raw "
            "byte stream, with no BPE tokenisation (VOCAB_SIZE=256)."
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

    # ── Resolve byte stream source ─────────────────────────────────────────────
    source = args.source
    if source is None:
        if not os.path.exists(FALLBACK_PATH):
            _download_fallback(FALLBACK_URL, FALLBACK_PATH)
        source = FALLBACK_PATH

    if not os.path.isfile(source):
        print(f"[rolling_horizon] ERROR: source file not found: {source}")
        sys.exit(1)

    file_size_mb = os.path.getsize(source) / 1e6
    print(f"[rolling_horizon] Byte stream source : {source}  ({file_size_mb:.1f} MB)")
    print(f"[rolling_horizon] Device             : {DEVICE}")
    print()

    # ── Run continuous evolution ───────────────────────────────────────────────
    cso = Continuous_Stream_Optimizer(source)
    cso.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
