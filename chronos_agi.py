"""
chronos_agi.py
==============
Chronos-AGI research module for H2Q-MicroStream.

This file intentionally keeps the implementation lightweight and modular:
1) Module 1: Higher-order latent state tracking via finite differences.
2) Module 2: Gravitational beam search with topology-inspired pruning.

The design goal is minimal intrusion: existing model/train scripts can keep
using their current path, while research code can opt into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TrackerConfig:
    """高阶状态追踪配置。"""

    eps: float = 1e-6
    attractor_window: int = 8
    curvature_scale: float = 1.0


class HigherOrderStateTracker:
    """
    Module 1: 高阶状态追踪器。

    说明（工程近似）：
    - 不做完整 Hessian / Koopman 真分解，而是使用隐藏状态轨迹的有限差分。
    - 对 hidden states 序列进行 1~4 阶差分，分别对应 velocity/acceleration/jerk/snap。
    - 用速度方向变化和加速度范数构造曲率代理。
    - 用“最近窗口中心距离”的负指数相似度构造吸引子分数。

    输入约定：
    - states: [T, D] 或 [B, T, D]。内部统一按 [B, T, D] 处理。
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()

    def _ensure_btd(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 2:
            return states.unsqueeze(0)
        if states.dim() != 3:
            raise ValueError(f"states must be [T,D] or [B,T,D], got {tuple(states.shape)}")
        return states

    def _finite_diff(self, x: torch.Tensor) -> torch.Tensor:
        # 有限差分：x[:, 1:] - x[:, :-1]
        return x[:, 1:, :] - x[:, :-1, :]

    def _pad_to(self, x: torch.Tensor, target_t: int) -> torch.Tensor:
        # 将短序列在时间维右侧补零到 target_t，便于与原始序列对齐。
        if x.size(1) >= target_t:
            return x[:, :target_t, :]
        pad_t = target_t - x.size(1)
        return F.pad(x, (0, 0, 0, pad_t), mode="constant", value=0.0)

    def _safe_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, dim=-1).clamp_min(self.config.eps)

    def _curvature_proxy(self, velocity: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
        """
        曲率代理：
        - 用相邻速度方向夹角变化（1 - cos）与加速度范数组合。
        - 输出 [B, T]，值越大表示轨迹弯折越明显。
        """
        v = velocity
        a = acceleration

        # 方向变化项：1 - cos(v_t, v_{t+1})
        v0 = v[:, :-1, :]
        v1 = v[:, 1:, :]
        cos = torch.sum(v0 * v1, dim=-1) / (self._safe_norm(v0) * self._safe_norm(v1))
        dir_change = (1.0 - cos.clamp(-1.0, 1.0)).clamp_min(0.0)

        # 对齐回 T 维
        dir_change = self._pad_to(dir_change.unsqueeze(-1), v.size(1)).squeeze(-1)

        # 加速度强度项
        acc_mag = self._safe_norm(a)

        return dir_change * self.config.curvature_scale + acc_mag

    def _attractor_score(self, states: torch.Tensor) -> torch.Tensor:
        """
        吸引子分数（工程代理）：
        - 取最近 W 步状态均值作为局部吸引子中心。
        - 当前状态与中心越接近，分数越高。
        - 输出 [B, T]，范围近似 (0, 1]。
        """
        b, t, d = states.shape
        w = max(2, min(self.config.attractor_window, t))

        out = []
        for i in range(t):
            s = max(0, i - w + 1)
            center = states[:, s : i + 1, :].mean(dim=1)  # [B, D]
            dist = self._safe_norm(states[:, i, :] - center)  # [B]
            score = torch.exp(-dist)
            out.append(score)

        return torch.stack(out, dim=1)  # [B, T]

    def track(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算高阶状态特征。

        Returns dict keys:
        - states: [B,T,D]
        - velocity, acceleration, jerk, snap: [B,T,D]
        - curvature_proxy: [B,T]
        - attractor_score: [B,T]
        """
        x = self._ensure_btd(states)
        t = x.size(1)

        v = self._finite_diff(x)
        a = self._finite_diff(v)
        j = self._finite_diff(a)
        s = self._finite_diff(j)

        # 对齐回 [B, T, D]
        v = self._pad_to(v, t)
        a = self._pad_to(a, t)
        j = self._pad_to(j, t)
        s = self._pad_to(s, t)

        curvature = self._curvature_proxy(v, a)
        attractor = self._attractor_score(x)

        return {
            "states": x,
            "velocity": v,
            "acceleration": a,
            "jerk": j,
            "snap": s,
            "curvature_proxy": curvature,
            "attractor_score": attractor,
        }


@dataclass
class BeamConfig:
    """引力 Beam Search 配置。"""

    beam_width: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 16

    # 评分项权重
    jerk_penalty_weight: float = 0.15
    snap_penalty_weight: float = 0.10
    attractor_bonus_weight: float = 0.20
    repeat_penalty_weight: float = 0.05
    length_penalty_alpha: float = 0.7


class GravitationalBeamSearch:
    """
    Module 2: 引力寻迹剪枝 Beam Search。

    评分函数（每条 beam）：
    score = log_prob
            - w_j * jerk_energy
            - w_s * snap_energy
            + w_a * attractor_score
            - w_r * repetition_penalty
            - length_penalty

    设计说明：
    - 该实现是研究骨架，强调可运行与可解释；
    - 不覆盖主模型现有 generate 行为，而是提供独立可选入口。
    """

    def __init__(
        self,
        tracker: Optional[HigherOrderStateTracker] = None,
        config: Optional[BeamConfig] = None,
    ):
        self.tracker = tracker or HigherOrderStateTracker()
        self.config = config or BeamConfig()

    def _length_penalty(self, length: int) -> float:
        # 类似 NMT 常见长度惩罚，防止 beam 过长偏置。
        alpha = self.config.length_penalty_alpha
        return ((5.0 + float(length)) / 6.0) ** alpha

    def _repeat_penalty(self, seq: torch.Tensor) -> float:
        # 简单重复惩罚：统计最近窗口中重复 token 比例。
        if seq.numel() < 4:
            return 0.0
        window = seq[-32:]
        uniq = torch.unique(window).numel()
        rep = 1.0 - float(uniq) / float(window.numel())
        return max(0.0, rep)

    def _model_forward(self, model, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一取 logits 与 final_hidden。
        - 优先使用 model.forward_features。
        - 回退到 model(seq) 时，final_hidden 用零张量占位。
        """
        if hasattr(model, "forward_features"):
            logits, _, features = model.forward_features(seq, return_hidden_states=False)
            final_hidden = features["final_hidden"]
            return logits, final_hidden

        logits, _ = model(seq)
        b, t, v = logits.shape
        # 回退路径无法拿到真实 hidden states，用 0 占位，保证代码可运行。
        final_hidden = torch.zeros((b, t, 1), device=logits.device, dtype=logits.dtype)
        return logits, final_hidden

    def search(self, model, prompt: torch.Tensor) -> torch.Tensor:
        """
        执行引力 Beam Search。

        Args:
            model: 自回归模型（支持 forward_features 更佳）
            prompt: [1, L]

        Returns:
            [1, L + max_new_tokens] 的最佳序列。
        """
        if prompt.dim() != 2 or prompt.size(0) != 1:
            raise ValueError("prompt must be shaped as [1, L]")

        device = prompt.device
        beams: List[Tuple[torch.Tensor, float]] = [(prompt.clone(), 0.0)]

        model.eval()
        for _ in range(self.config.max_new_tokens):
            candidates: List[Tuple[torch.Tensor, float]] = []

            for seq, score in beams:
                logits, hidden = self._model_forward(model, seq)
                next_logits = logits[:, -1, :] / max(self.config.temperature, 1e-6)

                # top-k 裁剪，避免分支爆炸
                k = min(self.config.top_k, next_logits.size(-1))
                topv, topi = torch.topk(next_logits, k=k, dim=-1)
                log_probs = F.log_softmax(topv, dim=-1)  # [1, k]

                # 追踪高阶状态（基于当前序列 hidden）
                tracked = self.tracker.track(hidden.squeeze(0))  # allow [T,D]
                jerk_energy = tracked["jerk"].pow(2).mean().item()
                snap_energy = tracked["snap"].pow(2).mean().item()
                attractor = tracked["attractor_score"][:, -1].mean().item()

                for j in range(k):
                    token = topi[0, j].view(1, 1)
                    token_logp = float(log_probs[0, j].item())

                    new_seq = torch.cat([seq, token.to(device)], dim=1)

                    repeat_pen = self._repeat_penalty(new_seq[0])
                    length_pen = self._length_penalty(new_seq.size(1))

                    raw = score + token_logp
                    gravity = (
                        raw
                        - self.config.jerk_penalty_weight * jerk_energy
                        - self.config.snap_penalty_weight * snap_energy
                        + self.config.attractor_bonus_weight * attractor
                        - self.config.repeat_penalty_weight * repeat_pen
                    )

                    final_score = gravity / max(length_pen, 1e-6)
                    candidates.append((new_seq, final_score))

            # 硬剪枝：仅保留 top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[: self.config.beam_width]

            if not beams:
                break

        # 返回最高分 beam
        best_seq = max(beams, key=lambda x: x[1])[0]
        return best_seq


def beam_search_generate(
    model,
    prompt: torch.Tensor,
    beam_width: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 16,
) -> torch.Tensor:
    """便捷函数：使用默认追踪器执行引力 Beam Search 生成。"""
    cfg = BeamConfig(
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    searcher = GravitationalBeamSearch(config=cfg)
    return searcher.search(model, prompt)
