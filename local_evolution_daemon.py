"""
local_evolution_daemon.py
=========================
Production-style local daemon for continuous H2Q evolution on CUDA.

Core guarantees:
1) Ada optimization enabled: torch.set_float32_matmul_precision('high').
2) Absolute device pinning: model and tensors stay on cuda:0 only.
3) Infinite byte-stream loop with seamless EOF wraparound via seek(0).
4) No graph accumulation: optimizer.zero_grad(set_to_none=True) immediately
   after optimizer.step().
5) Periodic VRAM cleanup: gc.collect + torch.cuda.empty_cache every N steps.
6) Physical telemetry every N steps into evolution_telemetry.csv.
7) Topology phase save when EMA loss hits new low and sparsity > 50%.
"""

import argparse
import csv
import gc
import math
import os
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F

from h2q_evolution import H2Q_Evolution_Engine
from sta_core_v2 import Rank8_Projection


# Tensor Core friendly matmul behavior.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def require_cuda0() -> torch.device:
    """Hard fail unless CUDA is available; lock runtime to cuda:0."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. This daemon is hard-locked to cuda:0 and has no CPU/MPS fallback."
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def vram_allocated_mb(device: torch.device) -> float:
    return torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)


def vram_reserved_mb(device: torch.device) -> float:
    return torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)


class ContinuousByteStream:
    """Read fixed-size byte chunks forever; wrap to file start at EOF."""

    def __init__(self, path: str, chunk_bytes: int):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Source file not found: {path}")
        if chunk_bytes <= 1:
            raise ValueError("chunk_bytes must be > 1")

        self.path = path
        self.chunk_bytes = chunk_bytes
        self.file = open(path, "rb")
        self.file_size = os.path.getsize(path)
        self.wrap_count = 0

    def next_chunk(self, device: torch.device) -> torch.Tensor:
        needed = self.chunk_bytes
        chunks = []
        remaining = needed

        while remaining > 0:
            data = self.file.read(remaining)
            if data:
                chunks.append(data)
                remaining -= len(data)
                continue

            # EOF reached: rewind seamlessly.
            self.file.seek(0)
            self.wrap_count += 1

        raw = b"".join(chunks)
        byte_tensor = torch.tensor(list(raw), dtype=torch.long, device=device)
        return byte_tensor

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


class TelemetryCSV:
    """Append-only telemetry sink with automatic header creation."""

    FIELDNAMES = [
        "T_Step",
        "Causal_Loss_EMA",
        "Topology_Sparsity",
        "SVD_Entropy",
        "VRAM_Allocated_MB",
        "StepPerSec",
    ]

    def __init__(self, path: str):
        self.path = path
        exists = os.path.exists(path)
        self.fp = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDNAMES)
        if not exists or os.path.getsize(path) == 0:
            self.writer.writeheader()
            self.fp.flush()

    def write(
        self,
        step: int,
        causal_loss_ema: float,
        topology_sparsity: float,
        svd_entropy: float,
        vram_allocated_mb_value: float,
        step_per_sec: float,
    ) -> None:
        self.writer.writerow(
            {
                "T_Step": step,
                "Causal_Loss_EMA": f"{causal_loss_ema:.8f}",
                "Topology_Sparsity": f"{topology_sparsity:.8f}",
                "SVD_Entropy": f"{svd_entropy:.8f}",
                "VRAM_Allocated_MB": f"{vram_allocated_mb_value:.4f}",
                "StepPerSec": f"{step_per_sec:.6f}",
            }
        )
        self.fp.flush()

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass


def compute_svd_entropy(model: H2Q_Evolution_Engine) -> float:
    """
    Compute Shannon entropy over singular-value energy distributions for all
    Rank8_Projection effective matrices W_eff = W_up @ W_down.
    """
    entropy_values = []
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, Rank8_Projection):
                continue

            w_down = module.proj[0].weight.float()  # [rank, dim]
            w_up = module.proj[1].weight.float()    # [dim, rank]
            w_eff = w_up @ w_down                   # [dim, dim]

            singular_vals = torch.linalg.svdvals(w_eff)
            energy = singular_vals.pow(2)
            total = energy.sum()
            if total <= 0:
                continue

            probs = (energy / total).clamp(min=1e-12)
            entropy = -(probs * torch.log2(probs)).sum().item()
            entropy_values.append(entropy)

    if not entropy_values:
        return 0.0
    return float(sum(entropy_values) / len(entropy_values))


class LocalEvolutionDaemon:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = require_cuda0()

        self.model = H2Q_Evolution_Engine(
            dim=args.dim,
            num_layers=args.num_layers,
            rank=args.rank,
            shockwave_threshold=args.shockwave_threshold,
            max_seq_len=args.seq_len,
        ).to(self.device)

        self._eager_model = self.model
        if args.compile:
            # Windows + Triton often needs serialized Inductor compilation and
            # a stable cache directory to avoid rename/collision issues.
            if os.name == "nt":
                if args.inductor_cache_dir:
                    cache_root = args.inductor_cache_dir
                    os.makedirs(cache_root, exist_ok=True)
                    cache_dir = os.path.join(cache_root, f"pid_{os.getpid()}")
                    os.makedirs(cache_dir, exist_ok=True)
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                if args.inductor_compile_threads > 0:
                    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(args.inductor_compile_threads)
            try:
                self.model = torch.compile(self.model, mode=args.compile_mode)
                self._compile_smoke_test()
                print(f"[opt] torch.compile enabled (mode={args.compile_mode})")
            except Exception as err:
                self.model = self._eager_model
                print(f"[opt] torch.compile unavailable, continue without compile: {err}")

        adamw_kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        if self.device.type == "cuda":
            adamw_kwargs["fused"] = True

        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **adamw_kwargs)
        self.optimizer.zero_grad(set_to_none=True)

        self.stream = ContinuousByteStream(args.source, args.seq_len + 1)
        self.telemetry = TelemetryCSV(args.telemetry_csv)

        self.step = 0
        self._steps_this_run = 0  # 本次运行步数（与 checkpoint 绝对步数分离）
        self.ema_loss: Optional[float] = None
        self.best_ema_loss = float("inf")
        self.last_svd_entropy = 0.0
        self.start_time = time.time()

        self.initial_vram_allocated = vram_allocated_mb(self.device)

        if self.args.resume:
            self._try_resume(self.args.resume)

    def _compile_smoke_test(self) -> None:
        """Run a tiny forward pass once so compile errors surface before main loop."""
        self.model.eval()
        with torch.no_grad():
            x = torch.zeros((1, self.args.seq_len), dtype=torch.long, device=self.device)
            _ = self.model(x)

    def _print_boot_banner(self) -> None:
        gpu_name = torch.cuda.get_device_name(self.device)
        props = torch.cuda.get_device_properties(self.device)
        total_vram_mb = props.total_memory / (1024.0 * 1024.0)

        print("=" * 92)
        print("H2Q Local Evolution Daemon :: Logical Stress + Cognitive Correction Mode")
        print("=" * 92)
        print(f"Device                : {self.device} :: {gpu_name}")
        print(f"Total VRAM            : {total_vram_mb:.2f} MB")
        print(f"Initial VRAM Alloc    : {self.initial_vram_allocated:.2f} MB")
        print(f"Byte source           : {self.args.source}")
        print(f"Byte source size      : {self.stream.file_size / (1024.0 * 1024.0):.2f} MB")
        print(
            f"Model config          : dim={self.args.dim}, layers={self.args.num_layers}, "
            f"rank={self.args.rank}, seq_len={self.args.seq_len}"
        )
        print(f"Optimizer             : AdamW(lr={self.args.lr}, wd={self.args.weight_decay})")
        print(f"Telemetry             : every {self.args.telemetry_every} steps -> {self.args.telemetry_csv}")
        print(f"Cache cleanup         : every {self.args.cache_clear_every} steps")
        print(f"Best topology path    : {self.args.best_checkpoint}")
        print("Loop mode             : while True (unbounded unless --max-steps is set)")
        print("=" * 92)
        print()

    def _save_best_topology(self, sparsity: float) -> None:
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "reason": "best_topology",
            "T_Step": self.step,
            "Causal_Loss_EMA": self.ema_loss,
            "best_ema_loss": self.best_ema_loss,
            "SVD_Entropy": self.last_svd_entropy,
            "wrap_count": self.stream.wrap_count,
            "Topology_Sparsity": sparsity,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "dim": self.args.dim,
                "num_layers": self.args.num_layers,
                "rank": self.args.rank,
                "seq_len": self.args.seq_len,
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay,
                "shockwave_threshold": self.args.shockwave_threshold,
            },
        }
        torch.save(payload, self.args.best_checkpoint)

    def _save_runtime_snapshot(self, path: str, reason: str) -> str:
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "reason": reason,
            "T_Step": self.step,
            "Causal_Loss_EMA": self.ema_loss,
            "best_ema_loss": self.best_ema_loss,
            "SVD_Entropy": self.last_svd_entropy,
            "wrap_count": self.stream.wrap_count,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "dim": self.args.dim,
                "num_layers": self.args.num_layers,
                "rank": self.args.rank,
                "seq_len": self.args.seq_len,
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay,
                "shockwave_threshold": self.args.shockwave_threshold,
            },
        }
        torch.save(payload, path)
        return path

    def _save_emergency_snapshot(self, reason: str) -> str:
        return self._save_runtime_snapshot(self.args.emergency_checkpoint, reason)

    def _try_resume(self, path: str) -> None:
        if not os.path.isfile(path):
            print(f"[resume] checkpoint not found, skip: {path}")
            return

        print(f"[resume] loading checkpoint: {path}")
        # Prefer safer loading mode on recent PyTorch versions.
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        model_state = ckpt.get("model_state")
        optim_state = ckpt.get("optimizer_state")
        if model_state is not None:
            self.model.load_state_dict(model_state)
        if optim_state is not None:
            self.optimizer.load_state_dict(optim_state)

        self.step = int(ckpt.get("T_Step", ckpt.get("step", 0)))
        loaded_ema = ckpt.get("Causal_Loss_EMA", ckpt.get("ema_loss"))
        if loaded_ema is not None:
            self.ema_loss = float(loaded_ema)
        loaded_best = ckpt.get("best_ema_loss")
        if loaded_best is not None:
            self.best_ema_loss = float(loaded_best)

        print(
            f"[resume] restored T={self.step}, ema={self.ema_loss}, best_ema={self.best_ema_loss}"
        )

    def _update_ema(self, value: float) -> float:
        if self.ema_loss is None:
            self.ema_loss = value
        else:
            self.ema_loss = self.args.ema_alpha * self.ema_loss + (1.0 - self.args.ema_alpha) * value
        return self.ema_loss

    def _heartbeat_print(
        self,
        causal_loss: float,
        topology_sparsity: float,
        vram_alloc_mb: float,
        vram_res_mb: float,
    ) -> None:
        elapsed = max(time.time() - self.start_time, 1e-6)
        steps_per_sec = self.step / elapsed
        bits_per_byte = (self.ema_loss / math.log(2)) if self.ema_loss is not None else float("nan")
        delta_vram = vram_alloc_mb - self.initial_vram_allocated

        print(
            f"[breath] T={self.step:>9} | loss(raw)={causal_loss:>8.5f} | "
            f"loss(ema)={self.ema_loss:>8.5f} | bpb={bits_per_byte:>7.4f} | "
            f"sparsity={topology_sparsity * 100:>6.2f}% | svd_H={self.last_svd_entropy:>6.3f} | "
            f"vram_alloc={vram_alloc_mb:>8.2f}MB (delta {delta_vram:+7.2f}) | "
            f"vram_reserved={vram_res_mb:>8.2f}MB | wraps={self.stream.wrap_count} | "
            f"speed={steps_per_sec:>7.2f} step/s"
        )

    def run(self) -> None:
        self._print_boot_banner()
        shutdown_reason = "normal_exit"
        amp_dtype = torch.bfloat16 if self.args.autocast_bf16 else None

        try:
            while True:
                # 使用本次运行步数计数（而非 checkpoint 恢复的绝对步数），
                # 确保 --resume 后仍能正确训练 --max-steps 步。
                if self.args.max_steps > 0 and self._steps_this_run >= self.args.max_steps:
                    print(f"[daemon] max steps reached: {self.args.max_steps}")
                    break

                chunk = self.stream.next_chunk(self.device)
                x = chunk[: self.args.seq_len].unsqueeze(0)
                y = chunk[1 : self.args.seq_len + 1].unsqueeze(0)

                # Predict on unseen chunk before training this chunk.
                self.model.eval()
                with torch.no_grad():
                    if amp_dtype is None:
                        logits, _ = self.model(x)
                    else:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            logits, _ = self.model(x)
                    causal_loss = F.cross_entropy(
                        logits.view(-1, self.model.VOCAB),
                        y.view(-1),
                    ).item()

                self._update_ema(causal_loss)

                # Evolve on the same chunk.
                self.model.train()
                if amp_dtype is None:
                    _, train_loss = self.model(x, targets=y)
                else:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, train_loss = self.model(x, targets=y)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Keep references short-lived.
                del logits, train_loss, x, y, chunk

                self.step += 1
                self._steps_this_run += 1

                topology_sparsity = float(self.model.get_topology_sparsity())

                if self.step % self.args.svd_every == 0:
                    with torch.no_grad():
                        self.last_svd_entropy = compute_svd_entropy(self.model)

                if self.step % self.args.cache_clear_every == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.step % self.args.telemetry_every == 0:
                    vram_alloc_mb = vram_allocated_mb(self.device)
                    vram_res_mb = vram_reserved_mb(self.device)
                    elapsed = max(time.time() - self.start_time, 1e-6)
                    step_per_sec = self.step / elapsed

                    with torch.no_grad():
                        self.last_svd_entropy = compute_svd_entropy(self.model)

                    self.telemetry.write(
                        step=self.step,
                        causal_loss_ema=float(self.ema_loss),
                        topology_sparsity=topology_sparsity,
                        svd_entropy=self.last_svd_entropy,
                        vram_allocated_mb_value=vram_alloc_mb,
                        step_per_sec=step_per_sec,
                    )

                    self._heartbeat_print(
                        causal_loss=causal_loss,
                        topology_sparsity=topology_sparsity,
                        vram_alloc_mb=vram_alloc_mb,
                        vram_res_mb=vram_res_mb,
                    )

                    # Topological phase save gate.
                    if self.ema_loss is not None:
                        if self.ema_loss < self.best_ema_loss and topology_sparsity > 0.5:
                            self.best_ema_loss = self.ema_loss
                            self._save_best_topology(topology_sparsity)
                            print(
                                "[phase-save] new best topology captured :: "
                                f"T={self.step} | ema={self.ema_loss:.6f} | "
                                f"sparsity={topology_sparsity * 100:.2f}% -> {self.args.best_checkpoint}"
                            )

                if self.step % self.args.print_every == 0 and self.step % self.args.telemetry_every != 0:
                    vram_alloc_mb = vram_allocated_mb(self.device)
                    vram_res_mb = vram_reserved_mb(self.device)
                    self._heartbeat_print(
                        causal_loss=causal_loss,
                        topology_sparsity=topology_sparsity,
                        vram_alloc_mb=vram_alloc_mb,
                        vram_res_mb=vram_res_mb,
                    )

        except KeyboardInterrupt:
            shutdown_reason = "keyboard_interrupt"
            print(f"\n[daemon] interrupted by user at T={self.step}")
            snap = self._save_emergency_snapshot("keyboard_interrupt")
            print(f"[daemon] emergency snapshot saved: {snap}")
        except RuntimeError as err:
            msg = str(err).lower()
            if "out of memory" in msg:
                shutdown_reason = "cuda_oom"
                print(f"\n[daemon][OOM] CUDA OOM at T={self.step}: {err}")
                snap = self._save_emergency_snapshot("cuda_oom")
                print(f"[daemon] emergency snapshot saved: {snap}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise
        finally:
            final_snapshot_path = ""
            if self.args.final_checkpoint:
                final_snapshot_path = self._save_runtime_snapshot(
                    self.args.final_checkpoint,
                    shutdown_reason,
                )

            self.stream.close()
            self.telemetry.close()
            elapsed = time.time() - self.start_time
            print("\n" + "=" * 92)
            print("Daemon shutdown summary")
            print(f"T_Step reached         : {self.step}")
            print(f"Elapsed seconds        : {elapsed:.2f}")
            print(f"Final EMA loss         : {self.ema_loss if self.ema_loss is not None else float('nan'):.6f}")
            print(f"Final SVD entropy      : {self.last_svd_entropy:.6f}")
            print(f"Final VRAM alloc (MB)  : {vram_allocated_mb(self.device):.2f}")
            print(f"Final VRAM reserv (MB) : {vram_reserved_mb(self.device):.2f}")
            print(f"File wrap count        : {self.stream.wrap_count}")
            if final_snapshot_path:
                print(f"Final checkpoint       : {final_snapshot_path}")
            print("=" * 92)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="24/7 local H2Q evolution daemon on cuda:0 with topology-aware checkpointing.",
    )
    parser.add_argument("--source", type=str, required=True, help="Path to a large local .txt/.bin byte source")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--shockwave-threshold", type=float, default=math.pi / 2)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.99)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for potential speedup (PyTorch 2.x).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--inductor-cache-dir",
        type=str,
        default=".inductor_cache",
        help="Inductor cache directory (recommended on Windows when --compile is enabled).",
    )
    parser.add_argument(
        "--inductor-compile-threads",
        type=int,
        default=1,
        help="Set TORCHINDUCTOR_COMPILE_THREADS when --compile is enabled.",
    )
    parser.add_argument(
        "--autocast-bf16",
        action="store_true",
        help="Enable bfloat16 autocast on CUDA for faster training.",
    )

    parser.add_argument("--telemetry-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--svd-every", type=int, default=1000)
    parser.add_argument("--cache-clear-every", type=int, default=10000)

    parser.add_argument("--telemetry-csv", type=str, default="evolution_telemetry.csv")
    parser.add_argument("--best-checkpoint", type=str, default="h2q_evolution_best_topology.pt")
    parser.add_argument("--emergency-checkpoint", type=str, default="h2q_evolution_emergency.pt")
    parser.add_argument(
        "--final-checkpoint",
        type=str,
        default="h2q_evolution_last.pt",
        help="Path for the always-saved end-of-run checkpoint used to resume the next cycle.",
    )
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means infinite while True loop")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    daemon = LocalEvolutionDaemon(args)
    daemon.run()


if __name__ == "__main__":
    main()
