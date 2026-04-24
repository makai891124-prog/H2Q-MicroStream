"""
train_evolution_loop.py
=======================
Chronos-AGI self-evolution training skeleton.

Pipeline:
1) Generate: model proposes trajectories (gravitational beam search).
2) Collapse: keep stable candidates using higher-order tracker signals.
3) Back-Alignment: update model with teacher-forcing style objective.

Notes:
- This is a runnable research scaffold, not a full PPO/DPO trainer.
- PPO/DPO/Shadow-Hamiltonian-Loss are intentionally left as extension hooks.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from chronos_agi import HigherOrderStateTracker, beam_search_generate
from h2q_evolution import H2Q_Evolution_Engine


@dataclass
class LoopConfig:
    source: str
    dim: int = 128
    num_layers: int = 4
    rank: int = 8
    seq_len: int = 128
    lr: float = 3e-4
    iterations: int = 20
    prompt_bytes: int = 32
    generate_bytes: int = 48
    beam_width: int = 4


class BytePromptPool:
    """从字节文件中抽样 prompt，作为 Generate 阶段起点。"""

    def __init__(self, source_path: str):
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"source not found: {source_path}")
        self.source_path = source_path
        self.data = open(source_path, "rb").read()
        if len(self.data) < 64:
            raise ValueError("source is too small for prompt sampling")

    def sample_prompt(self, prompt_bytes: int, device: torch.device) -> torch.Tensor:
        max_start = max(1, len(self.data) - prompt_bytes - 1)
        s = random.randint(0, max_start)
        raw = self.data[s : s + prompt_bytes]
        x = torch.tensor(list(raw), dtype=torch.long, device=device).unsqueeze(0)
        return x


class EvolutionLoop:
    """
    自进化训练闭环。

    阶段定义：
    - Generate：生成候选序列（用引力 Beam Search）。
    - Collapse：根据高阶状态特征筛选“可对齐”样本。
    - Back-Alignment：把样本转为 next-token 监督目标，执行一次更新。
    """

    def __init__(self, cfg: LoopConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.model = H2Q_Evolution_Engine(
            dim=cfg.dim,
            num_layers=cfg.num_layers,
            rank=cfg.rank,
            max_seq_len=cfg.seq_len,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.tracker = HigherOrderStateTracker()
        self.pool = BytePromptPool(cfg.source)

    def generate_phase(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成阶段：采样 prompt 并扩展为候选轨迹。"""
        prompt = self.pool.sample_prompt(self.cfg.prompt_bytes, self.device)
        candidate = beam_search_generate(
            self.model,
            prompt,
            beam_width=self.cfg.beam_width,
            max_new_tokens=self.cfg.generate_bytes,
            temperature=1.0,
            top_k=16,
        )
        return prompt, candidate

    def collapse_phase(self, candidate: torch.Tensor) -> bool:
        """
        坍缩阶段：按高阶特征做启发式筛选。

        简单规则：
        - jerk/snap 能量不能过高（避免轨迹剧烈抖动）；
        - attractor 分数不能过低（避免轨迹持续漂移）。
        """
        self.model.eval()
        with torch.no_grad():
            _, _, features = self.model.forward_features(candidate, return_hidden_states=False)
            hidden = features["final_hidden"]  # [1,T,D]

        tracked = self.tracker.track(hidden)
        jerk_energy = tracked["jerk"].pow(2).mean().item()
        snap_energy = tracked["snap"].pow(2).mean().item()
        attractor_last = tracked["attractor_score"][:, -1].mean().item()

        keep = (jerk_energy < 2.0) and (snap_energy < 2.0) and (attractor_last > 0.10)

        print(
            f"[collapse] keep={keep} jerk={jerk_energy:.4f} snap={snap_energy:.4f} "
            f"attractor={attractor_last:.4f}"
        )
        return keep

    def back_alignment_phase(self, seq: torch.Tensor) -> float:
        """
        反向对齐阶段：
        用 teacher-forcing 的 next-token 目标执行一次基础优化。

        这里是可执行骨架：
        - PPO update hook（未实现）
        - DPO update hook（未实现）
        - Shadow Hamiltonian Loss hook（未实现）
        """
        if seq.size(1) < 2:
            return 0.0

        x = seq[:, :-1]
        y = seq[:, 1:]

        self.model.train()
        logits, loss = self.model(x, targets=y)
        if loss is None:
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def run(self) -> None:
        accepted = 0
        total_loss = 0.0

        for it in range(1, self.cfg.iterations + 1):
            # 1) Generate
            prompt, candidate = self.generate_phase()
            print(f"\n[iter {it}] prompt_len={prompt.size(1)} candidate_len={candidate.size(1)}")

            # 2) Collapse
            keep = self.collapse_phase(candidate)
            if not keep:
                print("[iter] candidate dropped by collapse phase")
                continue

            # 3) Back-Alignment
            loss = self.back_alignment_phase(candidate)
            accepted += 1
            total_loss += loss
            print(f"[back-align] loss={loss:.6f}")

        mean_loss = total_loss / max(accepted, 1)
        print("\n" + "=" * 72)
        print("Evolution loop finished")
        print(f"iterations={self.cfg.iterations} accepted={accepted} mean_loss={mean_loss:.6f}")
        print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chronos-AGI self-evolution training skeleton")
    p.add_argument("--source", type=str, default="data/open_corpus/open_corpus.bin")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--prompt-bytes", type=int, default=32)
    p.add_argument("--generate-bytes", type=int, default=48)
    p.add_argument("--beam-width", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = LoopConfig(
        source=args.source,
        dim=args.dim,
        num_layers=args.num_layers,
        rank=args.rank,
        seq_len=args.seq_len,
        lr=args.lr,
        iterations=args.iterations,
        prompt_bytes=args.prompt_bytes,
        generate_bytes=args.generate_bytes,
        beam_width=args.beam_width,
    )

    loop = EvolutionLoop(cfg, device)
    loop.run()


if __name__ == "__main__":
    main()
