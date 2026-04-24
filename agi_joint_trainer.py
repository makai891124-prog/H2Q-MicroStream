"""
agi_joint_trainer.py  —  联合加速 AGI 训练系统
================================================
整合所有现有数学加速能力，DeepSeek 外部督导联合训练。

数学加速方法清单 (Math Acceleration Inventory):
─────────────────────────────────────────────────────
① STA-v2: 代数冲击波截断 (Algebraic Shockwave Truncation)
   - arccos  →  raw_inner < cos(λ)，完全消除超越函数
   - Rank-8 投影: D→8→D，98.4% 参数节省 @D=1024
   - torch.bmm 球面内积，直接映射 cuBLAS GEMM 核
   - SO(D+1) Givens 旋转子球面位置编码 (SU(2)类RoPE)

② TCRH: 拓扑类路由与哈希 (Topological Class Routing & Hashing)
   - 二进制 LSH 签名 (符号量化)：整数比较替代浮点相似度
   - Chern 类整数桶路由：O(1) 兼容性过滤
   - Homotopy Hamming 位运算距离过滤：无 dot-product
   - 因果掩码感知路由 + Rank-8 V/O 投影

③ Hamilton 四元数运算 (Quaternion Hamilton Operations)
   - WaveStructureBank: 正交秩-8 四元数因子库 (共享正交基)
   - BalancedHamiltonLayer: 表示四元数乘积 O(8·D) vs 矩阵乘 O(D²)
   - ortho_loss: 强制因子组正交性 (代数结构正则化)

④ TF32 张量核心 (TensorFloat-32 Tensor Core Acceleration)
   - torch.set_float32_matmul_precision('high')
   - allow_tf32=True for matmul + cudnn

⑤ 梯度检查点 (Gradient Checkpointing)
   - torch.utils.checkpoint: 以重算换显存，支持更大批次

⑥ 异步双流数据加载 (Async Dual-Stream Data Pipeline)
   - 后台线程 + Queue 流水线
   - DeepSeek 注入热交换 (Injection hot-swap)

⑦ 权重绑定 (Weight Tying)
   - 输出投影共享嵌入矩阵 (embedding == head.weight)

⑧ DeepSeek 外部督导 (External Supervision)
   - 每 N chunk 生成模型样本 → DeepSeek 纠正 → 注入训练流

架构: HybridAcceleratedBlock 交替使用
  偶数层 → STA-v2  (球面拓扑 + 冲击波截断)
  奇数层 → CausalTCRH (整数路由 + Hamming 滤波)
  全部层 → Hamilton FF + 梯度检查点

数据: FineWeb-Edu_Full (E:\\Datasets)
"""

from __future__ import annotations

import math
import os
import sys
import gc
import csv
import time
import random
import glob
import threading
import queue
import shutil
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# §1  TF32 张量核心加速  (加速④)
# ══════════════════════════════════════════════════════════════════════════════
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ══════════════════════════════════════════════════════════════════════════════
# §2  导入路径 — 引入所有现有加速模块
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "H2Q-Single"))

from sta_core_v2 import (
    Rank8_Projection,
    StereographicAttentionLayer,
    Stereographic_Attention_Layer_V2,
    SphericalTopologicalEncoding,
    inverse_stereo_project,
)
from tcrh_layer import Topological_Hash_Encoder

try:
    from deepseek_supervisor import DeepSeekSupervisor
    _SUPERVISOR_OK = True
except ImportError:
    _SUPERVISOR_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# §3  设备锁定 (cuda:0 hard-lock)
# ══════════════════════════════════════════════════════════════════════════════
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required — this trainer is hard-locked to cuda:0.")

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

# ══════════════════════════════════════════════════════════════════════════════
# §4  配置
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # 模型架构
    "dim": 768,
    "factor_size": 32,   # Hamilton 分块尺寸; dim / factor_size 须被 4 整除
    "fixed_rank": 8,     # Rank-8 瓶颈 (STA-v2 + TCRH + Hamilton 全部使用)
    "depth": 12,         # 总层数 (6 STA-v2 + 6 TCRH 交替)
    "seq_len": 128,
    "batch_size": 24,
    "dropout_rate": 0.1,
    "axiom_lambda": 0.1,  # Hamilton ortho_loss 权重

    # STA-v2 冲击波截断阈值 (λ ∈ [0, π])
    "shockwave_threshold": math.pi / 2,
    "sta_variant": "sta_v2",
    "binary_num_planes": 128,
    "binary_chunk_size": 64,
    "binary_routing_mode": "normalize",
    "binary_backend": "packbits",
    "binary_fused_chunk_compute": True,

    # TCRH 哈希超参
    "hash_dim": 64,
    "num_buckets": 8,
    "hamming_thresh": 8,

    # 优化器
    "lr": 3e-4,
    "weight_decay": 0.02,
    "grad_clip": 1.0,

    # 训练
    "total_chunks": 200000,
    "chunk_size_mb": 10,

    # 路径
    "source_dir": r"E:\Datasets\FineWeb-Edu_Full",
    "buffer_dir": r"D:\H2Q_Cache_Zone",
    "checkpoint_path": "agi_joint.pt",
    "best_model_path": "agi_joint_best.pt",
    "telemetry_csv": "agi_joint_telemetry.csv",

    # DeepSeek 督导 (加速⑧)
    "supervise_every": 10,
    "supervise_gen_tokens": 256,

    # 评估窗口倍率：eval_limit = eval_window_multiplier * seq_len
    "eval_window_multiplier": 1000,
}

# ══════════════════════════════════════════════════════════════════════════════
# §5  Hamilton 四元数组件  (加速③)
#     device-agnostic 版: 不依赖全局 device 变量，随模型 .to(device) 迁移
# ══════════════════════════════════════════════════════════════════════════════

class WaveStructureBank(nn.Module):
    """
    正交秩-8 四元数因子库 (所有 BalancedHamiltonLayer 共享此库)。

    factors_A: [rank, 4, sub_blocks, sub_blocks]
      — 四个四元数分量 (r, i, j, k) 各自的 sub_blocks×sub_blocks 正交矩阵
    加速: 强迫信息通过 rank=8 个正交本征模流动 (结构正则化)
    """

    def __init__(self, num_blocks: int, rank: int):
        super().__init__()
        assert num_blocks % 4 == 0, f"num_blocks={num_blocks} 须被 4 整除"
        self.sub_blocks = num_blocks // 4
        self.rank = rank
        self.factors_A = nn.Parameter(
            torch.zeros(rank, 4, self.sub_blocks, self.sub_blocks)
        )
        with torch.no_grad():
            for r in range(rank):
                c = torch.randn(4, self.sub_blocks, self.sub_blocks)
                for comp in range(4):
                    nn.init.orthogonal_(c[comp])
                self.factors_A.data[r] = c * ((r + 1) ** -0.5)

    def get_factors(self) -> torch.Tensor:
        return self.factors_A


class BalancedHamiltonLayer(nn.Module):
    """
    Hamilton 四元数积替代标准线性层。

    加速原理:
      rank=8 可分离四元数乘积: O(8·D) 参数 vs 标准 nn.Linear O(D²)
      einsum 路径直接映射 cuBLAS GEMM，充分利用 TF32 张量核心

    Args:
        dim:         输入/输出维度 D (= 4 * sub_blocks * factor_size)
        factor_size: 子块尺寸
        bank:        共享的 WaveStructureBank
        rank:        秩 (一般 = fixed_rank = 8)
    """

    def __init__(self, dim: int, factor_size: int, bank: WaveStructureBank, rank: int):
        super().__init__()
        self.dim = dim
        self.factor_size = factor_size
        self.bank = bank
        self.factors_B = nn.Parameter(torch.zeros(rank, factor_size, factor_size))
        self.bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            for r in range(rank):
                b = torch.randn(factor_size, factor_size)
                nn.init.orthogonal_(b)
                self.factors_B.data[r] = b * ((r + 1) ** -0.5)

    def _construct_hamilton(self, A: torch.Tensor) -> torch.Tensor:
        """将 factors_A 展开为 [rank, 4·sub, 4·sub] 四元数乘法矩阵。"""
        r, i, j, k = A[:, 0], A[:, 1], A[:, 2], A[:, 3]   # 各 [rank, sub, sub]
        row0 = torch.cat([ r, -i, -j, -k], dim=2)
        row1 = torch.cat([ i,  r, -k,  j], dim=2)
        row2 = torch.cat([ j,  k,  r, -i], dim=2)
        row3 = torch.cat([ k, -j,  i,  r], dim=2)
        return torch.cat([row0, row1, row2, row3], dim=1)   # [rank, 4·sub, 4·sub]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        sub = self.bank.sub_blocks
        # x 重塑为四元数分块格式
        x_flat = x.reshape(B * T, 4 * sub, self.factor_size)       # [N, 4·sub, fs]
        A = self.bank.get_factors().to(dtype=x.dtype)
        B_f = self.factors_B.to(dtype=x.dtype)
        # ① 乘入 factors_B: [rank, N, 4·sub, fs]
        wav = torch.einsum("nsi,rji->rnsj", x_flat, B_f)
        # ② 乘入 Hamilton 矩阵: [N, 4·sub, fs]
        ham = self._construct_hamilton(A)
        out = torch.einsum("rnsj,rks->nkj", wav, ham)
        return out.reshape(B, T, D) + self.bias

    def ortho_loss(self) -> torch.Tensor:
        """正交性正则损失 — 强制 factors_B 保持正交结构。"""
        dev = self.factors_B.device
        loss = torch.tensor(0.0, device=dev)
        for p in self.factors_B:
            pf = p.float()
            loss = loss + torch.norm(pf.t() @ pf - torch.eye(pf.shape[1], device=dev))
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# §6  因果 TCRH 注意力  (加速②)
#     Rank-8 V/O 投影 + 因果掩码 + LSH + Hamming 位运算
# ══════════════════════════════════════════════════════════════════════════════

class CausalTCRH_Attention(nn.Module):
    """
    带因果掩码的拓扑类路由哈希注意力。

    三级加速过滤 (全部基于整数/位运算, 无浮点 dot-product):
      Level 1 — Chern 整数桶过滤  (chern_tag[i] != chern_tag[j] → hard-0)
      Level 2 — Homotopy Hamming 过滤 (hamming_dist > thresh → hard-0)
      Level 3 — 因果过滤            (j > i → hard-0)

    连通 token 对采用均匀权重 (1/|bucket|), 完全绕过 softmax。
    V/O 投影统一使用 Rank-8 瓶颈 (与 STA-v2 一致)。

    复杂度: O(N·hash_dim) 哈希 + O(N²) 桶掩码 (稀疏)
    """

    def __init__(
        self,
        dim: int,
        hash_dim: int = 64,
        num_buckets: int = 8,
        hamming_thresh: int = 8,
    ):
        super().__init__()
        self.hamming_thresh = hamming_thresh
        self.encoder = Topological_Hash_Encoder(dim, hash_dim, num_buckets)
        # Rank-8 瓶颈投影 (加速①② 共用 Rank-8 设计)
        self.v_proj = Rank8_Projection(dim, 8)
        self.o_proj = Rank8_Projection(dim, 8)
        self._last_connectivity: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        hash_sigs, chern_tags = self.encoder(x)           # LSH 量化
        V = self.v_proj(x)                                # [B, T, D]

        # ── Level 1: Chern 整数桶过滤 ─────────────────────────────────────
        c_q = chern_tags.unsqueeze(2)                     # [B, T, 1]
        c_k = chern_tags.unsqueeze(1)                     # [B, 1, T]
        chern_match = (c_q == c_k)                        # [B, T, T]

        # ── Level 2: Homotopy Hamming 位运算过滤 ─────────────────────────
        h_q = hash_sigs.unsqueeze(2).to(torch.int32)     # [B, T, 1, h]
        h_k = hash_sigs.unsqueeze(1).to(torch.int32)     # [B, 1, T, h]
        hamming = (h_q != h_k).sum(dim=-1)               # [B, T, T]
        connected = chern_match & (hamming <= self.hamming_thresh)

        # ── Level 3: 因果掩码 (时间箭头约束) ─────────────────────────────
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        connected = connected & ~causal.unsqueeze(0)     # [B, T, T]

        # 记录连通率 (遥测)
        self._last_connectivity = connected.float().mean().item()

        # 均匀权重 (无 softmax)
        w = connected.float()
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = w @ V                                       # [B, T, D]
        return self.o_proj(out)

    def get_connectivity(self) -> float:
        return self._last_connectivity


# ══════════════════════════════════════════════════════════════════════════════
# §7  混合加速块 (HybridAcceleratedBlock)
#     偶数层 → STA-v2 (冲击波截断 + Rank-8 + bmm + 球面编码)
#     奇数层 → CausalTCRH (整数路由 + Hamming + 因果掩码)
#     全层   → Hamilton 四元数 FF + 梯度检查点  (加速③⑤)
# ══════════════════════════════════════════════════════════════════════════════

class HybridAcceleratedBlock(nn.Module):
    """
    交替使用两种加速注意力的变换器块。

    加速方法组合:
      Self-Attn:  偶数层 STA-v2 | 奇数层 CausalTCRH
      FF:         BalancedHamiltonLayer (Hamilton 积) × 2
      Checkpoint: 注意力 + FF 均使用 torch.utils.checkpoint
    """

    def __init__(
        self,
        dim: int,
        factor_size: int,
        bank: WaveStructureBank,
        rank: int,
        layer_idx: int,
        seq_len: int,
        shockwave_threshold: float,
        hash_dim: int,
        num_buckets: int,
        hamming_thresh: int,
        sta_variant: str,
        binary_num_planes: int,
        binary_chunk_size: int,
        binary_routing_mode: str,
        binary_backend: str,
        binary_fused_chunk_compute: bool,
    ):
        super().__init__()
        self.use_sta = (layer_idx % 2 == 0)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

        if self.use_sta:
            if sta_variant == "binary_sta":
                self.attn = StereographicAttentionLayer(
                    hidden_dim=dim,
                    num_planes=binary_num_planes,
                    chunk_size=binary_chunk_size,
                    routing_mode=binary_routing_mode,
                    binary_backend=binary_backend,
                    fused_chunk_compute=binary_fused_chunk_compute,
                    causal=True,
                )
            elif sta_variant == "sta_v2":
                # 加速①: STA-v2 (代数冲击波截断 + Rank-8 + bmm + 球面编码)
                self.attn = Stereographic_Attention_Layer_V2(
                    hidden_dim=dim,
                    shockwave_threshold=shockwave_threshold,
                    rank=rank,
                    max_seq_len=seq_len,
                    causal=True,
                )
            else:
                raise ValueError(f"unsupported sta_variant: {sta_variant}")
        else:
            # 加速②: CausalTCRH (LSH + Chern + Hamming + 因果掩码)
            self.attn = CausalTCRH_Attention(dim, hash_dim, num_buckets, hamming_thresh)

        # 加速③: Hamilton 四元数前馈 (两层 Hamilton 积)
        self.ff1 = BalancedHamiltonLayer(dim, factor_size, bank, rank)
        self.ff2 = BalancedHamiltonLayer(dim, factor_size, bank, rank)
        self.act = nn.GELU()

    # ── 检查点函数 (norm 在 checkpoint 内部以最大化显存节省) ───────────────
    def _attn_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm_attn(x))

    def _ff_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(self.act(self.ff1(self.norm_ff(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 加速⑤: 梯度检查点 — 以重算换显存
        x = x + cp.checkpoint(self._attn_fn, x, use_reentrant=False)
        x = x + cp.checkpoint(self._ff_fn,  x, use_reentrant=False)
        return x

    def ortho_loss(self) -> torch.Tensor:
        return self.ff1.ortho_loss() + self.ff2.ortho_loss()

    def get_attn_stat(self) -> float:
        """STA-v2 → shockwave 稀疏率; TCRH → 连通率 (取负以统一方向)。"""
        if self.use_sta:
            return float(getattr(self.attn, "last_sparsity", 0.0))
        else:
            return float(self.attn.get_connectivity())


# ══════════════════════════════════════════════════════════════════════════════
# §8  AGI 加速变换器  (全加速方法组合)
# ══════════════════════════════════════════════════════════════════════════════

class AGI_Accelerated_Transformer(nn.Module):
    """
    整合所有 8 种加速方法的 AGI 字节级自回归语言模型。

    架构规格:
      - VOCAB = 256 (原始字节, 无 tokenizer)
      - Embedding (256, dim) — 与 output head 权重绑定 (加速⑦)
      - depth × HybridAcceleratedBlock
          偶数层: STA-v2 (加速①)
          奇数层: CausalTCRH (加速②)
          全部层: Hamilton FF (加速③) + GradCheckpoint (加速⑤)
      - LayerNorm → head (weight-tied, 加速⑦)
    """

    VOCAB: int = 256

    def __init__(self, config: dict):
        super().__init__()
        dim = config["dim"]
        fs  = config["factor_size"]
        rank = config["fixed_rank"]
        depth = config["depth"]
        seq_len = config["seq_len"]

        # WaveStructureBank: 全维度 Hamilton FF
        # 约束: dim = 4 * sub_blocks * factor_size → num_blocks = dim // factor_size
        num_blocks = dim // fs
        assert num_blocks % 4 == 0, (
            f"dim/factor_size={num_blocks} 须整除 4 (dim={dim}, factor_size={fs})"
        )
        self.bank = WaveStructureBank(num_blocks, rank)

        # 加速⑦: 嵌入 + 权重绑定
        self.emb = nn.Embedding(self.VOCAB, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.drop = nn.Dropout(config["dropout_rate"])

        # 交替 STA-v2 / CausalTCRH + Hamilton FF + GradCheckpoint
        self.layers = nn.ModuleList([
            HybridAcceleratedBlock(
                dim=dim,
                factor_size=fs,
                bank=self.bank,
                rank=rank,
                layer_idx=i,
                seq_len=seq_len,
                shockwave_threshold=config["shockwave_threshold"],
                hash_dim=config["hash_dim"],
                num_buckets=config["num_buckets"],
                hamming_thresh=config["hamming_thresh"],
                sta_variant=config["sta_variant"],
                binary_num_planes=config["binary_num_planes"],
                binary_chunk_size=config["binary_chunk_size"],
                binary_routing_mode=config["binary_routing_mode"],
                binary_backend=config["binary_backend"],
                binary_fused_chunk_compute=config["binary_fused_chunk_compute"],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.VOCAB, bias=False)
        self.head.weight = self.emb.weight   # ← 权重绑定 (加速⑦)

        self.axiom_lambda = config["axiom_lambda"]
        self._seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = x.shape
        h = self.drop(self.emb(x) + self.pos[:, :T, :])

        ortho = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            h = layer(h)
            ortho = ortho + layer.ortho_loss()

        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, self.VOCAB), targets.reshape(-1))
            loss = ce + self.axiom_lambda * ortho * 0.01

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, new_tokens: int) -> torch.Tensor:
        for _ in range(new_tokens):
            idx_cond = idx[:, -self._seq_len:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def accel_stats(self) -> dict:
        """收集所有加速层的实时遥测数据。"""
        sta_sp, tcrh_conn = [], []
        for l in self.layers:
            s = l.get_attn_stat()
            if l.use_sta:
                sta_sp.append(s)
            else:
                tcrh_conn.append(s)
        with torch.no_grad():
            ol = sum(l.ortho_loss().item() for l in self.layers)
        return {
            "sta_sparsity_mean":   sum(sta_sp) / max(len(sta_sp), 1),
            "tcrh_conn_mean":      sum(tcrh_conn) / max(len(tcrh_conn), 1),
            "ortho_loss":          ol,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §9  加速遥测 CSV
# ══════════════════════════════════════════════════════════════════════════════

class AccelTelemetry:
    FIELDS = [
        "timestamp", "chunk", "train_loss", "val_loss",
        "sta_sparsity", "tcrh_connectivity", "ortho_loss",
        "tokens_per_sec", "vram_alloc_gb",
    ]

    def __init__(self, path: str):
        self.path = path
        exists = os.path.exists(path)
        self.fp = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDS)
        if not exists or os.path.getsize(path) == 0:
            self.writer.writeheader()
            self.fp.flush()

    def write(self, **kwargs):
        row = {f: kwargs.get(f, "") for f in self.FIELDS}
        self.writer.writerow(row)
        self.fp.flush()

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# §10  异步双流数据加载器  (加速⑥ — 来自 qrlm_m4_evo.py)
# ══════════════════════════════════════════════════════════════════════════════

class AsyncBufferedLoader:
    """
    FineWeb 异步双流加载器。
    加速⑥: 后台线程预加载 + Queue 流水线 + DeepSeek Injection 热交换。
    """

    def __init__(self, config: dict, resume_file_index: int = 0):
        self.chunk_size = config["chunk_size_mb"] * 1024 * 1024
        self.batch_size = config["batch_size"]
        self.source_dir = config["source_dir"]
        self.buffer_dir = config["buffer_dir"]

        self.injection_dir = os.path.join(self.buffer_dir, "Injection")
        os.makedirs(self.injection_dir, exist_ok=True)

        print(f"[Loader] 扫描数据源: {self.source_dir}")
        self.file_list = sorted(
            glob.glob(os.path.join(self.source_dir, "**/*.parquet"), recursive=True)
        )
        if not self.file_list:
            print(f"[Loader] 错误: 未在 {self.source_dir} 找到 .parquet 文件")
            sys.exit(1)
        print(f"[Loader] 发现 {len(self.file_list)} 个数据文件")

        self.current_file_index = int(resume_file_index)
        self.queue: queue.Queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.buffer_integers: list = []

        self.loader_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.loader_thread.start()

    def _clean_buffer_dir(self):
        for f in os.listdir(self.buffer_dir):
            fp = os.path.join(self.buffer_dir, f)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass

    def _ingest_injection_files(self):
        inj_files = sorted(glob.glob(os.path.join(self.injection_dir, "*.parquet")))
        if not inj_files:
            return False
        print(f"[Loader] 💉 注入 {len(inj_files)} 个 DeepSeek 意识碎片...")
        for f in inj_files:
            try:
                df = pd.read_parquet(f, columns=["text"])
                try:
                    os.remove(f)
                except Exception:
                    pass
                for text in df["text"].dropna().astype(str).tolist():
                    b = text.encode("utf-8", errors="ignore") + b"\0"
                    self.buffer_integers.extend(b)
            except Exception as e:
                print(f"[Loader] 注入读取失败: {e}")
        return True

    def _process_parquet(self, file_path: str):
        try:
            df = pd.read_parquet(file_path, columns=["text"])
            try:
                os.remove(file_path)
            except Exception:
                pass
            for idx, text in enumerate(df["text"].dropna().astype(str).tolist()):
                if self.stop_event.is_set():
                    break
                if idx % 1000 == 0:
                    self._ingest_injection_files()
                b = text.encode("utf-8", errors="ignore") + b"\0"
                self.buffer_integers.extend(b)
                if len(self.buffer_integers) >= self.chunk_size:
                    t = torch.tensor(self.buffer_integers[: self.chunk_size], dtype=torch.long)
                    self.queue.put(t)
                    self.buffer_integers = self.buffer_integers[self.chunk_size :]
        except Exception as e:
            print(f"[Loader] 文件处理错误 ({os.path.basename(file_path)}): {e}")

    def _background_worker(self):
        if self.current_file_index == 0:
            self._clean_buffer_dir()
        while not self.stop_event.is_set():
            self._ingest_injection_files()
            if self.current_file_index >= len(self.file_list):
                print("[Loader] 数据集循环重置...")
                self.current_file_index = 0
            src = self.file_list[self.current_file_index]
            buf = os.path.join(self.buffer_dir, os.path.basename(src))
            try:
                if not os.path.exists(buf):
                    shutil.copy2(src, buf)
                self._process_parquet(buf)
                self.current_file_index += 1
            except Exception as e:
                print(f"[Loader] 主循环错误: {e}")
                self.current_file_index += 1
                time.sleep(1)

    def load_next_chunk(self) -> Optional[torch.Tensor]:
        try:
            data = self.queue.get(timeout=120)
        except queue.Empty:
            print("[Loader] 数据加载超时")
            return None
        num_batches = len(data) // self.batch_size
        valid_len = num_batches * self.batch_size
        if valid_len == 0:
            return self.load_next_chunk()
        return data[:valid_len].view(self.batch_size, num_batches).contiguous().to(DEVICE)

    def decode(self, token_ids: list) -> str:
        valid = bytes([i for i in token_ids if 0 < i < 256])
        return valid.decode("utf-8", errors="ignore")

    def get_bookmark(self) -> int:
        return self.current_file_index

    def stop(self):
        self.stop_event.set()


# ══════════════════════════════════════════════════════════════════════════════
# §11  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def get_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(DEVICE) / 1024 ** 3
    return 0.0


def sanitize_state(sd: dict) -> dict:
    return {
        (k[10:] if k.startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }


def print_boot_banner(model: AGI_Accelerated_Transformer, cfg: dict):
    props = torch.cuda.get_device_properties(DEVICE)
    total_vram = props.total_memory / 1024 ** 3
    param_count = sum(p.numel() for p in model.parameters())
    n_sta  = sum(1 for l in model.layers if l.use_sta)
    n_tcrh = sum(1 for l in model.layers if not l.use_sta)

    print("=" * 90)
    print("AGI 联合加速训练系统  /  AGI Joint Accelerated Trainer")
    print("=" * 90)
    print(f"  GPU          : {torch.cuda.get_device_name(DEVICE)}")
    print(f"  VRAM 总量    : {total_vram:.1f} GB")
    print(f"  参数量       : {param_count:,}")
    print(f"  架构         : {cfg['depth']} 层 ({n_sta} STA-v2 + {n_tcrh} CausalTCRH) × Hamilton FF")
    print(f"  dim/rank     : {cfg['dim']} / {cfg['fixed_rank']}")
    print(f"  seq_len      : {cfg['seq_len']}")
    print(f"  batch_size   : {cfg['batch_size']}")
    print()
    print("  已启用加速方法:")
    print(f"    ① STA-v2       冲击波截断 λ={cfg['shockwave_threshold']:.4f} ({n_sta} 层)")
    print(f"    ② CausalTCRH   LSH+Hamming hash_dim={cfg['hash_dim']} bucket={cfg['num_buckets']} ({n_tcrh} 层)")
    print(f"    ③ Hamilton FF  Rank-{cfg['fixed_rank']} 四元数积 factor_size={cfg['factor_size']}")
    print(f"    ④ TF32         matmul+cudnn allow_tf32=True")
    print(f"    ⑤ GradCkpt     torch.utils.checkpoint (attn+ff 全部)")
    print(f"    ⑥ AsyncLoader  FineWeb 双流 chunk={cfg['chunk_size_mb']}MB + Injection 热交换")
    print(f"    ⑦ WeightTie    embedding == head.weight")
    print(f"    ⑧ DeepSeek     督导注入 every={cfg['supervise_every']} chunks")
    print("=" * 90)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §12  联合训练主循环
# ══════════════════════════════════════════════════════════════════════════════

def train_joint(cfg: dict):
    resume_file_index = 0
    chunk_counter     = 0
    best_loss         = float("inf")

    # ── 模型初始化 ──────────────────────────────────────────────────────────
    model = AGI_Accelerated_Transformer(cfg).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ── 恢复检查点 ──────────────────────────────────────────────────────────
    ckpt_path = cfg["checkpoint_path"]
    if os.path.exists(ckpt_path):
        print(f"[Train] 恢复存档: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(sanitize_state(ckpt["model"]))
            opt.load_state_dict(ckpt["optimizer"])
            chunk_counter     = ckpt.get("chunk_counter", 0)
            best_loss         = ckpt.get("best_loss", float("inf"))
            saved_offset      = ckpt.get("dataset_offset", 0)
            resume_file_index = saved_offset if saved_offset < 500000 else 0
            print(f"[Train] ✅ 存档加载 chunk={chunk_counter}, best_loss={best_loss:.4f}")
        except Exception:
            print("[Train] 存档加载失败，从头训练")
            traceback.print_exc()
    else:
        print("[Train] 未找到存档，从头开始")

    for pg in opt.param_groups:
        pg["lr"] = cfg["lr"]

    print_boot_banner(model, cfg)

    # ── 数据加载器 (加速⑥) ─────────────────────────────────────────────────
    loader = AsyncBufferedLoader(cfg, resume_file_index)

    # ── DeepSeek 督导器 (加速⑧) ────────────────────────────────────────────
    supervisor = None
    if _SUPERVISOR_OK and cfg.get("supervise_every", 0) > 0:
        inj_dir = os.path.join(cfg["buffer_dir"], "Injection")
        supervisor = DeepSeekSupervisor(
            injection_dir=inj_dir,
            every_n_chunks=cfg["supervise_every"],
            gen_tokens=cfg.get("supervise_gen_tokens", 256),
        )
        print(f"[Train] DeepSeek 督导已启用 (every {cfg['supervise_every']} chunks)")
    else:
        print("[Train] DeepSeek 督导未启用")

    # ── 遥测 ────────────────────────────────────────────────────────────────
    telemetry = AccelTelemetry(cfg["telemetry_csv"])

    print("[Train] 等待初始数据 ...")
    current_data = loader.load_next_chunk()
    if current_data is None:
        print("[Train] 数据加载失败，退出")
        return

    print(f"[Train] 🚀 开始训练 chunk={chunk_counter}/{cfg['total_chunks']}")

    try:
        model.train()
        while chunk_counter < cfg["total_chunks"]:
            t0 = time.time()

            # ── 预取下一块 ─────────────────────────────────────────────────
            future_data = loader.load_next_chunk()
            if future_data is None:
                print("[Train] 数据耗尽，结束训练")
                break

            seq_len = cfg["seq_len"]

            # ── 验证阶段 (Rolling Horizon: 在未见数据上评估) ───────────────
            model.eval()
            val_accum, val_steps = 0.0, 0
            eval_limit = min(
                future_data.size(1),
                cfg.get("eval_window_multiplier", 1000) * seq_len,
            )
            with torch.no_grad():
                for i in range(0, eval_limit, seq_len):
                    if i + seq_len + 1 > future_data.size(1):
                        break
                    vx = future_data[:, i      : i + seq_len    ]
                    vy = future_data[:, i + 1  : i + seq_len + 1]
                    _, vl = model(vx, vy)
                    val_accum += vl.item()
                    val_steps += 1
            avg_val = val_accum / max(val_steps, 1)
            model.train()

            # ── 训练阶段 ───────────────────────────────────────────────────
            train_accum, train_steps = 0.0, 0
            chunk_t0 = time.time()

            for i in range(0, current_data.size(1), seq_len):
                if i + seq_len + 1 > current_data.size(1):
                    break
                x = current_data[:, i      : i + seq_len    ]
                y = current_data[:, i + 1  : i + seq_len + 1]

                # TF32 直接前传 + 反传 (加速④)
                _, loss = model(x, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)

                train_accum += loss.item()
                train_steps += 1

                if train_steps % 50 == 0:
                    tps = (cfg["batch_size"] * seq_len) / max(time.time() - chunk_t0, 1e-6)
                    sys.stdout.write(
                        f"\r  [训练] step={train_steps} loss={loss.item():.4f} "
                        f"speed={int(tps)} tok/s vram={get_vram_gb():.2f}GB"
                    )
                    sys.stdout.flush()

            avg_train = train_accum / max(train_steps, 1)
            tokens_per_sec = (
                cfg["batch_size"] * seq_len * train_steps / max(time.time() - chunk_t0, 1e-6)
            )
            current_data = future_data
            chunk_counter += 1

            # ── 加速统计 ───────────────────────────────────────────────────
            stats = model.accel_stats()
            total_time = time.time() - t0

            print(
                f"\n{'='*70}\n"
                f"  Chunk {chunk_counter:5d} | train={avg_train:.4f} val={avg_val:.4f} "
                f"diff={avg_val-avg_train:+.4f}\n"
                f"  STA稀疏率={stats['sta_sparsity_mean']*100:.1f}%  "
                f"TCRH连通率={stats['tcrh_conn_mean']*100:.1f}%  "
                f"ortho_loss={stats['ortho_loss']:.3f}\n"
                f"  速度={int(tokens_per_sec)} tok/s  "
                f"VRAM={get_vram_gb():.2f}GB  "
                f"时间={total_time:.1f}s  "
                f"文件索引={loader.get_bookmark()}"
            )

            # ── 遥测写入 ───────────────────────────────────────────────────
            telemetry.write(
                timestamp=datetime.utcnow().isoformat() + "Z",
                chunk=chunk_counter,
                train_loss=f"{avg_train:.6f}",
                val_loss=f"{avg_val:.6f}",
                sta_sparsity=f"{stats['sta_sparsity_mean']:.4f}",
                tcrh_connectivity=f"{stats['tcrh_conn_mean']:.4f}",
                ortho_loss=f"{stats['ortho_loss']:.4f}",
                tokens_per_sec=f"{tokens_per_sec:.0f}",
                vram_alloc_gb=f"{get_vram_gb():.3f}",
            )

            # ── 检查点保存 ─────────────────────────────────────────────────
            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "chunk_counter": chunk_counter},
                    cfg["best_model_path"],
                )
            torch.save(
                {
                    "chunk_counter":  chunk_counter,
                    "model":          model.state_dict(),
                    "optimizer":      opt.state_dict(),
                    "best_loss":      best_loss,
                    "dataset_offset": loader.get_bookmark(),
                },
                cfg["checkpoint_path"],
            )

            # ── DeepSeek 督导注入 (加速⑧) ──────────────────────────────────
            if supervisor is not None:
                supervisor.maybe_supervise(chunk_counter, model, loader, DEVICE)

            # ── Thought Stream 生成 (每 5 chunk 一次) ──────────────────────
            if chunk_counter % 5 == 0:
                print("\n[思维流] 自由联想生成:")
                model.eval()
                with torch.no_grad():
                    seed_str = random.choice(["The ", "Why ", "If ", "It is ", "I "])
                    ctx = torch.tensor(
                        [list(seed_str.encode("utf-8"))], dtype=torch.long, device=DEVICE
                    )
                    out = model.generate(ctx, 300)
                    print(f"诱导词: [{seed_str.strip()}]")
                    print(loader.decode(out[0].tolist()))
                    print("-" * 60)
                model.train()

            # ── 周期 VRAM 清理 ─────────────────────────────────────────────
            if chunk_counter % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n[Train] 用户中断，紧急保存...")
    except Exception:
        print("\n[Train] 严重错误:")
        traceback.print_exc()
    finally:
        # 保存当前状态
        try:
            torch.save(
                {
                    "chunk_counter":  chunk_counter,
                    "model":          model.state_dict(),
                    "optimizer":      opt.state_dict(),
                    "best_loss":      best_loss,
                    "dataset_offset": loader.get_bookmark(),
                },
                cfg["checkpoint_path"],
            )
            print(f"[Train] 状态已保存: {cfg['checkpoint_path']}")
        except Exception:
            print("[Train] 紧急保存失败")
        loader.stop()
        telemetry.close()
        print(
            f"\n[Train] 完成 — chunk={chunk_counter}, best_loss={best_loss:.4f}, "
            f"VRAM={get_vram_gb():.2f}GB"
        )


# ══════════════════════════════════════════════════════════════════════════════
# §13  命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AGI 联合加速训练系统 — 整合 STA-v2 + TCRH + Hamilton + TF32 + GradCkpt + DeepSeek"
    )
    p.add_argument("--dim",           type=int,   default=CONFIG["dim"])
    p.add_argument("--depth",         type=int,   default=CONFIG["depth"])
    p.add_argument("--seq-len",       type=int,   default=CONFIG["seq_len"])
    p.add_argument("--batch-size",    type=int,   default=CONFIG["batch_size"])
    p.add_argument("--lr",            type=float, default=CONFIG["lr"])
    p.add_argument("--total-chunks",  type=int,   default=CONFIG["total_chunks"])
    p.add_argument("--chunk-size-mb", type=int,   default=CONFIG["chunk_size_mb"])
    p.add_argument("--source-dir",    type=str,   default=CONFIG["source_dir"])
    p.add_argument("--buffer-dir",    type=str,   default=CONFIG["buffer_dir"])
    p.add_argument("--supervise-every", type=int, default=CONFIG["supervise_every"])
    p.add_argument(
        "--eval-window-multiplier", type=int,
        default=CONFIG["eval_window_multiplier"],
        help="验证窗口倍率：eval_limit = multiplier * seq_len",
    )
    p.add_argument(
        "--checkpoint-path", type=str, default=CONFIG["checkpoint_path"],
        help="实验 checkpoint 保存路径",
    )
    p.add_argument(
        "--best-model-path", type=str, default=CONFIG["best_model_path"],
        help="实验 best model 保存路径",
    )
    p.add_argument(
        "--telemetry-csv", type=str, default=CONFIG["telemetry_csv"],
        help="实验 telemetry 输出 CSV",
    )
    p.add_argument(
        "--shockwave-threshold", type=float,
        default=CONFIG["shockwave_threshold"],
        help="STA-v2 冲击波截断阈值 λ (默认 π/2)",
    )
    p.add_argument(
        "--sta-variant", type=str, default=CONFIG["sta_variant"],
        choices=["sta_v2", "binary_sta"],
        help="偶数层使用的 STA 变体",
    )
    p.add_argument(
        "--binary-num-planes", type=int, default=CONFIG["binary_num_planes"],
        help="binary_sta addressing planes 数量",
    )
    p.add_argument(
        "--binary-chunk-size", type=int, default=CONFIG["binary_chunk_size"],
        help="binary_sta 分块相似度 chunk 大小",
    )
    p.add_argument(
        "--binary-routing-mode", type=str, default=CONFIG["binary_routing_mode"],
        choices=["normalize", "softmax"],
        help="binary_sta 路由归一方式",
    )
    p.add_argument(
        "--binary-backend", type=str, default=CONFIG["binary_backend"],
        choices=["packbits", "int8", "cuda_ext"],
        help="binary_sta 位运算后端",
    )
    p.add_argument(
        "--binary-fused-chunk-compute", type=int,
        default=1 if CONFIG["binary_fused_chunk_compute"] else 0,
        choices=[0, 1],
        help="binary_sta 是否启用分块即算即聚合",
    )
    p.add_argument(
        "--hash-dim", type=int, default=CONFIG["hash_dim"],
        help="TCRH LSH 哈希维度",
    )
    p.add_argument(
        "--num-buckets", type=int, default=CONFIG["num_buckets"],
        help="TCRH Chern 桶数量",
    )
    p.add_argument(
        "--hamming-thresh", type=int, default=CONFIG["hamming_thresh"],
        help="TCRH Hamming 距离阈值",
    )
    return p


def main():
    args = build_parser().parse_args()
    cfg = dict(CONFIG)
    cfg.update({
        "dim":                  args.dim,
        "depth":                args.depth,
        "seq_len":              args.seq_len,
        "batch_size":           args.batch_size,
        "lr":                   args.lr,
        "total_chunks":         args.total_chunks,
        "chunk_size_mb":        args.chunk_size_mb,
        "source_dir":           args.source_dir,
        "buffer_dir":           args.buffer_dir,
        "supervise_every":      args.supervise_every,
        "eval_window_multiplier": args.eval_window_multiplier,
        "checkpoint_path":      args.checkpoint_path,
        "best_model_path":      args.best_model_path,
        "telemetry_csv":        args.telemetry_csv,
        "shockwave_threshold":  args.shockwave_threshold,
        "sta_variant":          args.sta_variant,
        "binary_num_planes":    args.binary_num_planes,
        "binary_chunk_size":    args.binary_chunk_size,
        "binary_routing_mode":  args.binary_routing_mode,
        "binary_backend":       args.binary_backend,
        "binary_fused_chunk_compute": bool(args.binary_fused_chunk_compute),
        "hash_dim":             args.hash_dim,
        "num_buckets":          args.num_buckets,
        "hamming_thresh":       args.hamming_thresh,
    })
    train_joint(cfg)


if __name__ == "__main__":
    main()
