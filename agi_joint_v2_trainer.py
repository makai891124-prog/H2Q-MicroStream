"""
agi_joint_v2_trainer.py  —  AGI 联合加速训练系统 V2
====================================================
在 V1 (agi_joint_trainer.py) 全部 8 种数学加速方法基础上，
完整整合 h2q_prime_engine.py 的四种新数学结构：

新增数学方法清单 (New Math Inventory):
─────────────────────────────────────────────────────
⑨ Mahler P-进 Pascal 位置编码 (Mahler-Pascal Positional Encoding)
   - Mahler定理: 任意整数序列函数可展开为 f(n) = Σ c_k · C(n,k)
   - 位置编码基向量 = 归一化Pascal行向量 [C(n,0)/C(N-1,0), ..., C(n,K-1)/C(N-1,K-1)]
   - 差分层级性: Δ^k PE(n) 只保留第k阶Mahler系数 (天然多尺度分解)
   - 可学习线性投影: K维Mahler坐标 → dim维特征空间
   - 替换原始 self.pos = nn.Parameter(randn) 随机位置编码

⑩ P-进字节嵌入 (P-adic Byte Decomposition Embedding)
   - 数学基础: n = Σ_{k=0}^{7} d_k · 2^k, d_k ∈ {0,1} (2-进展开)
   - 8个可学习位位置嵌入 (每位只有0/1两个向量)
   - 可学习位权重: 高位权重初始化为大值, 衰减至低位
   - 与标准 token embedding 叠加, 提供显式位结构先验
   - 使自回归模型天然感知字节级二进制结构

⑪ Mahler 差分层 (Mahler Difference Layer)
   - 将隐藏序列在后向差分算子 ∇^k 下展开:
     ∇^k h[n] = Σ_{j=0}^{k} (-1)^j · C(k,j) · h[n-j]  (因果: 只看过去)
   - 等价于在整数函数的Mahler基下对序列做多阶多项式分解
   - k=0: 恒等 (当前词); k=1: 差分 (语义变化); k=2: 加速度; ...
   - 每阶用 Rank-8 投影处理, 可学习混合权重聚合各阶输出
   - 使用 F.conv1d 实现高效因果卷积 (直接映射 cuDNN 核)
   - 作为第三种注意力类型 (层索引 mod 3 == 2)

⑫ 素数轮 LSH 投影 (Primorial-Structured LSH Projection)
   - 前 2×|P| 列使用素数谐波基: v[d,2i] = cos(2π·d/p_i)/√D
   - 结构化部分捕获特征空间的模运算对称性
   - 与随机Gaussian部分线性混合 (prime_blend 参数控制比例)
   - 保留TCRH全部三级过滤机制 (Chern + Hamming + 因果掩码)

扩展规格 (Scale-Up):
  dim:       768  → 1024   (+33%)
  depth:     12   → 18     (+50%)   (6 STA + 6 TCRH + 6 Mahler 三路交替)
  seq_len:   128  → 256    (×2)
  batch:     24   → 12     (为更长序列保持VRAM)
  total_chunks: 200000 → 300000 (更长训练)

数据: FineWeb-Edu_Full (E:\\Datasets)
"""

from __future__ import annotations

import math
import os
import sys
# Force UTF-8 output on Windows to avoid GBK/CP936 encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
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
# §2  导入路径
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "H2Q-Single"))

from sta_core_v2 import (
    Rank8_Projection,
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

from core_compute_codec import CoreTelemetryCSV, compute_core_metrics, to_core_telemetry_path
from ungs_core import UNGSCore, ungs_total_loss

# ══════════════════════════════════════════════════════════════════════════════
# §3  设备锁定 (cuda:0 hard-lock)
# ══════════════════════════════════════════════════════════════════════════════
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required — this trainer is hard-locked to cuda:0.")

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

# ══════════════════════════════════════════════════════════════════════════════
# §4  V2 配置
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── 模型架构 (扩大规模) ──────────────────────────────────────────────────
    "dim":          1024,    # 768 → 1024  (+33%)
    "factor_size":  32,      # Hamilton 分块尺寸  (dim/factor_size = 32, 32%4==0 ✓)
    "fixed_rank":   8,       # Rank-8 哲学统一贯穿所有组件
    "depth":        18,      # 12 → 18  (6 STA-v2 + 6 TCRH + 6 Mahler 三路交替)
    "seq_len":      256,     # 128 → 256  (×2)
    "batch_size":   12,      # 24 → 12   (配合更长序列保持 VRAM)
    "dropout_rate": 0.1,
    "axiom_lambda": 0.1,

    # ── 新增: UNGS 核心约束 (Phase A 起步实现) ───────────────────────────────
    "ungs_enabled": True,
    "ungs_closure_lambda": 0.05,
    "ungs_encapsulation_lambda": 0.03,
    "ungs_self_ref_lambda": 0.02,
    "ungs_relation_threshold": 0.60,
    "adaptive_control_enabled": True,
    "adaptive_control_every": 1,
    "control_warmup_chunks": 5,
    "control_curriculum_chunks": 20,
    "control_warmup_scale": 0.20,
    "control_val_worse_tolerance": 0.01,
    "control_val_worse_patience": 2,
    "control_protection_cooldown": 3,
    "control_protection_pressure_scale": 0.50,
    "control_protection_lambda_decay": 0.90,
    "target_relation_density": 0.08,
    "target_hierarchy_ratio": 0.03,
    "target_self_ref_consistency": 0.60,
    "target_ungs_loss": 0.80,
    "target_generalization_gap": 0.20,
    "control_lambda_step": 0.01,
    "control_axiom_step": 0.01,
    "control_lr_down": 0.92,
    "control_lr_up": 1.02,
    "control_lr_min": 1e-5,
    "control_lr_max": 1e-3,
    "ungs_lambda_min": 0.0,
    "ungs_lambda_max": 0.5,
    "axiom_lambda_min": 0.01,
    "axiom_lambda_max": 0.5,

    # ── STA-v2 配置 (加速①) ──────────────────────────────────────────────────
    "shockwave_threshold": math.pi / 2,

    # ── TCRH 配置 (加速②) ────────────────────────────────────────────────────
    "hash_dim":      64,
    "num_buckets":   8,
    "hamming_thresh": 8,

    # ── 新增: 素数轮 LSH 配置 (加速⑫) ────────────────────────────────────────
    "prime_blend":   0.4,    # 40% 素数谐波 + 60% 随机 Gaussian 混合

    # ── 新增: Mahler Pascal 位置编码 (加速⑨) ──────────────────────────────────
    "mahler_basis_order": 32, # Pascal矩阵截断阶数 (K), 投影到 dim

    # ── 新增: P-进字节嵌入 (加速⑩) ───────────────────────────────────────────
    "padic_precision": 8,    # 2-进展开精度 (8位 = 1字节, token范围 0-255)

    # ── 新增: Mahler 差分层 (加速⑪) ──────────────────────────────────────────
    "mahler_diff_order": 8,  # 最大差分阶数 (0..7阶后向差分)

    # ── 优化器 ──────────────────────────────────────────────────────────────
    "lr":           3e-4,
    "weight_decay": 0.02,
    "grad_clip":    1.0,

    # ── 训练 ─────────────────────────────────────────────────────────────────
    "total_chunks":          300_000,  # 200000 → 300000
    "chunk_size_mb":         10,

    # ── 路径 ─────────────────────────────────────────────────────────────────
    "source_dir":        r"E:\Datasets\FineWeb-Edu_Full",
    "buffer_dir":        r"D:\H2Q_Cache_Zone",
    "checkpoint_path":   "agi_joint_v2.pt",
    "best_model_path":   "agi_joint_v2_best.pt",
    "telemetry_csv":     "agi_joint_v2_telemetry.csv",

    # ── DeepSeek 督导 (加速⑧) ────────────────────────────────────────────────
    "supervise_every":       10,
    "supervise_gen_tokens":  256,

    # ── 评估 ─────────────────────────────────────────────────────────────────
    "eval_window_multiplier": 1000,
    "seed": 42,
}

# ══════════════════════════════════════════════════════════════════════════════
# §5  Hamilton 四元数组件  (加速③)  — 与 V1 相同
# ══════════════════════════════════════════════════════════════════════════════

class WaveStructureBank(nn.Module):
    """正交秩-8 四元数因子库 (全部 BalancedHamiltonLayer 共享)。"""

    def __init__(self, num_blocks: int, rank: int):
        super().__init__()
        assert num_blocks % 4 == 0, f"num_blocks={num_blocks} 须被4整除"
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
    """Hamilton 四元数积替代标准线性层 (加速③)。"""

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
        r, i, j, k = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
        row0 = torch.cat([ r, -i, -j, -k], dim=2)
        row1 = torch.cat([ i,  r, -k,  j], dim=2)
        row2 = torch.cat([ j,  k,  r, -i], dim=2)
        row3 = torch.cat([ k, -j,  i,  r], dim=2)
        return torch.cat([row0, row1, row2, row3], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        sub = self.bank.sub_blocks
        x_flat = x.reshape(B * T, 4 * sub, self.factor_size)
        A = self.bank.get_factors().to(dtype=x.dtype)
        B_f = self.factors_B.to(dtype=x.dtype)
        wav = torch.einsum("nsi,rji->rnsj", x_flat, B_f)
        ham = self._construct_hamilton(A)
        out = torch.einsum("rnsj,rks->nkj", wav, ham)
        return out.reshape(B, T, D) + self.bias

    def ortho_loss(self) -> torch.Tensor:
        dev = self.factors_B.device
        loss = torch.tensor(0.0, device=dev)
        for p in self.factors_B:
            pf = p.float()
            loss = loss + torch.norm(pf.t() @ pf - torch.eye(pf.shape[1], device=dev))
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# §6  P-进字节嵌入  (加速⑩)
#     h2q_prime_engine.py: p_adic_encode(n, p=2, precision=8)
#     将 token ID (0-255) 展开为 8 个二进制位，各位独立嵌入后混合叠加
# ══════════════════════════════════════════════════════════════════════════════

class PAdicByteEmbedding(nn.Module):
    """
    2-进 P-adic 字节嵌入。

    数学基础 (来自 h2q_prime_engine.p_adic_encode):
      n = Σ_{k=0}^{7} d_k · 2^k,  d_k = (n >> k) & 1  ∈ {0,1}

    架构:
      - 8个独立嵌入表 E_k ∈ R^{2×dim}, 各对应第k个二进制位
      - 可学习位权重 w_k (初始化为递减: 高位 > 低位)
      - 输出 = Σ_k softmax(w)[k] · E_k[d_k]

    直觉: 模型可以直接"看到"字节的位结构 (奇偶性、高位分类等),
          无需从 token embedding 中隐式学习这些结构。
    """

    def __init__(self, dim: int, precision: int = 8):
        super().__init__()
        self.precision = precision
        self.bit_embeddings = nn.ModuleList(
            [nn.Embedding(2, dim) for _ in range(precision)]
        )
        # 初始化: 高位 (k=7) 权重大, 低位 (k=0) 权重小
        init_w = torch.tensor(
            [1.0 - k * 0.08 for k in range(precision)], dtype=torch.float32
        )
        self.bit_weights = nn.Parameter(init_w)

        # 初始化嵌入为小值
        for emb in self.bit_embeddings:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T]  token id (整数, 0-255)
        Returns:
            out: [B, T, dim]
        """
        w = torch.softmax(self.bit_weights, dim=0)   # [precision]
        out = None
        for k, emb in enumerate(self.bit_embeddings):
            bit_k = (x >> k) & 1                     # [B, T]  第k个二进制位
            emb_k = emb(bit_k)                        # [B, T, dim]
            out = emb_k * w[k] if out is None else out + emb_k * w[k]
        return out                                    # [B, T, dim]


# ══════════════════════════════════════════════════════════════════════════════
# §7  Mahler-Pascal 位置编码  (加速⑨)
#     h2q_prime_engine.py: build_pascal_matrix, compute_mahler_coefficients
#     将位置 n 映射为归一化 Pascal 行向量 [C(n,0)/C(N-1,0), ..., C(n,K-1)/C(N-1,K-1)]
#     再通过可学习线性层投影到 dim 维特征空间
# ══════════════════════════════════════════════════════════════════════════════

class MahlerPascalPositionalEncoding(nn.Module):
    """
    Mahler 二项式基位置编码 (替换 V1 的随机可学习 pos 参数)。

    数学基础 (来自 h2q_prime_engine.build_pascal_matrix):
      Mahler定理: f(n) = Σ_{k≥0} c_k · C(n,k)  (对所有整数函数成立)
      Pascal矩阵: B[n,k] = C(n,k)  (将Mahler系数向量映射为函数值向量)

    编码构造:
      1. 预计算归一化Pascal行向量: B_norm[n,k] = C(n,k) / C(N-1,k)  ∈ [0,1]
         - k=0 列: 全为 1.0 (常数基)
         - k=1 列: n/(N-1) (线性基)
         - k=2 列: n(n-1)/((N-1)(N-2)) (二次基)
         - ...  (k阶多项式基)
      2. 可学习投影: Linear(K, dim) 将K维Mahler坐标映射到dim维空间

    差分性质 (关键优势):
      Δ^k B_norm[n] ≈ e_k (第k个标准基向量), 即第k阶有限差分只保留第k阶Mahler模式
      → 模型的注意力机制天然感知序列的多阶多项式结构
    """

    def __init__(self, seq_len: int, dim: int, mahler_order: int = 32):
        super().__init__()
        self.mahler_order = mahler_order

        # 预计算归一化Pascal矩阵 [seq_len, mahler_order]
        # B_norm[n, k] = C(n,k) / C(seq_len-1, k),  其中 C(0,k>0) = 0
        pascal = torch.zeros(seq_len, mahler_order, dtype=torch.float64)
        for n in range(seq_len):
            for k in range(mahler_order):
                if k > n:
                    pascal[n, k] = 0.0
                elif k == 0:
                    pascal[n, k] = 1.0
                else:
                    denom = math.comb(seq_len - 1, k)
                    pascal[n, k] = math.comb(n, k) / denom
        # 降精度为 float32 并注册为固定 buffer
        self.register_buffer("pascal_basis", pascal.float())  # [seq_len, mahler_order]

        # 可学习投影: K维Mahler坐标 → dim维特征
        self.proj = nn.Linear(mahler_order, dim, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, T: int) -> torch.Tensor:
        """
        Returns:
            pe: [1, T, dim]
        """
        return self.proj(self.pascal_basis[:T]).unsqueeze(0)  # [1, T, dim]


# ══════════════════════════════════════════════════════════════════════════════
# §8  素数轮 LSH 投影  (加速⑫)
#     h2q_prime_engine.py: rank8_sieve_analysis, segmented_sieve
#     用素数谐波基向量 (cos/sin 模运算) 替换纯随机 LSH 投影的40%份额
#     捕获特征空间中与素数余数类对应的模运算对称性
# ══════════════════════════════════════════════════════════════════════════════

class PrimorialHashEncoder(nn.Module):
    """
    素数轮结构化 LSH 哈希编码器 (替换 V1 的纯随机 Topological_Hash_Encoder)。

    数学基础 (来自 h2q_prime_engine.rank8_sieve_analysis 的哲学):
      素数筛矩阵的 SVD 揭示: 数论结构可由少数本征向量近似捕获。
      在特征空间中, cos(2π·d/p_i) 和 sin(2π·d/p_i) 对应
      "第d个特征维度在素数p_i的余数类中的谐波分量"。
      与随机投影混合后, 得到兼具理论结构和通用覆盖的LSH基。

    投影矩阵 proj ∈ R^{dim × hash_dim}:
      前 2×|P| 列 = 素数谐波基 (p=2,3,5,7,11,...的cos/sin)
      其余列    = 随机 Gaussian (标准LSH)
      最终 = prime_blend × 素数部分 + (1-prime_blend) × 随机部分
    """

    # 前32个素数 (足以覆盖所有实际的 hash_dim 场景)
    _PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131,
    ]

    def __init__(
        self,
        hidden_dim: int,
        hash_dim: int = 64,
        num_buckets: int = 8,
        prime_blend: float = 0.4,
    ):
        super().__init__()
        self.hash_dim = hash_dim
        self.num_buckets = num_buckets
        self.tag_bits = max(1, math.ceil(math.log2(max(num_buckets, 2))))

        # ── 随机 Gaussian 投影 (标准 LSH 基线) ──────────────────────────────
        rand_proj = torch.randn(hidden_dim, hash_dim) / (hidden_dim ** 0.5)

        # ── 素数谐波投影 ────────────────────────────────────────────────────
        structured = torch.zeros(hidden_dim, hash_dim)
        d_range = torch.arange(hidden_dim, dtype=torch.float32)
        num_primes_available = min(len(self._PRIMES), hash_dim // 2)
        for idx in range(num_primes_available):
            p = self._PRIMES[idx]
            # cos 分量: 捕获余数类 d mod p 的偶对称性
            structured[:, 2 * idx]     = torch.cos(2.0 * math.pi * d_range / p)
            # sin 分量: 捕获余数类 d mod p 的奇反对称性
            structured[:, 2 * idx + 1] = torch.sin(2.0 * math.pi * d_range / p)
        # 归一化 (与随机投影量纲一致)
        structured = structured / (hidden_dim ** 0.5)

        # ── 混合 ────────────────────────────────────────────────────────────
        proj = prime_blend * structured + (1.0 - prime_blend) * rand_proj

        # 固定不训练 (与原始 TCRH 一致: LSH 投影为固定随机矩阵)
        self.register_buffer("proj", proj)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, D]
        Returns:
            hash_signatures: [B, T, hash_dim]  int8 二进制码
            chern_tags:      [B, T]            int32 桶索引
        """
        projected = x @ self.proj.to(x.dtype)                        # [B, T, hash_dim]
        hash_signatures = (projected > 0).to(torch.int8)             # 二值化 LSH

        # Chern 整数桶: 前 tag_bits 位解释为二进制整数
        powers = torch.arange(
            self.tag_bits, device=x.device, dtype=torch.int32
        )
        chern_tags = (
            hash_signatures[:, :, :self.tag_bits].to(torch.int32) * (2 ** powers)
        ).sum(dim=-1)                                                 # [B, T]

        return hash_signatures, chern_tags


# ══════════════════════════════════════════════════════════════════════════════
# §9  因果 TCRH 注意力 V2  (加速②⑫)
#     使用 PrimorialHashEncoder 替换原始随机 Topological_Hash_Encoder
# ══════════════════════════════════════════════════════════════════════════════

class CausalTCRH_Attention_V2(nn.Module):
    """
    带因果掩码的素数轮拓扑类路由哈希注意力。

    三级整数/位运算过滤 (同 V1) + 素数谐波 LSH (新):
      Level 1 — Chern 整数桶过滤  (素数结构桶分配)
      Level 2 — Homotopy Hamming 过滤
      Level 3 — 因果时间箭头约束
    """

    def __init__(
        self,
        dim: int,
        hash_dim: int = 64,
        num_buckets: int = 8,
        hamming_thresh: int = 8,
        prime_blend: float = 0.4,
    ):
        super().__init__()
        self.hamming_thresh = hamming_thresh
        self.encoder = PrimorialHashEncoder(dim, hash_dim, num_buckets, prime_blend)
        self.v_proj = Rank8_Projection(dim, 8)
        self.o_proj = Rank8_Projection(dim, 8)
        self._last_connectivity: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        hash_sigs, chern_tags = self.encoder(x)
        V = self.v_proj(x)                                           # [B, T, D]

        c_q = chern_tags.unsqueeze(2)
        c_k = chern_tags.unsqueeze(1)
        chern_match = (c_q == c_k)

        h_q = hash_sigs.unsqueeze(2).to(torch.int32)
        h_k = hash_sigs.unsqueeze(1).to(torch.int32)
        hamming = (h_q != h_k).sum(dim=-1)
        connected = chern_match & (hamming <= self.hamming_thresh)

        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        connected = connected & ~causal.unsqueeze(0)

        self._last_connectivity = connected.float().mean().item()

        w = connected.float()
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = w @ V
        return self.o_proj(out)

    def get_connectivity(self) -> float:
        return self._last_connectivity


# ══════════════════════════════════════════════════════════════════════════════
# §10  Mahler 差分层  (加速⑪)
#      h2q_prime_engine.py: forward_difference_k, compute_mahler_coefficients
#      在隐藏序列上施加因果后向差分算子 ∇^k (k=0..max_order-1)
#      多阶差分输出经 Rank-8 投影后加权聚合
# ══════════════════════════════════════════════════════════════════════════════

class MahlerDifferenceLayer(nn.Module):
    """
    Mahler 因果后向差分注意力层 (第三种注意力类型)。

    数学基础 (来自 h2q_prime_engine.forward_difference_k):
      ∇^k h[n] = Σ_{j=0}^{k} (-1)^j · C(k,j) · h[n-j]    (后向差分, 因果)
      k=0: ∇^0 h[n] = h[n]                    (当前位置, 恒等)
      k=1: ∇^1 h[n] = h[n] - h[n-1]           (一阶差分, 语义变化速度)
      k=2: ∇^2 h[n] = h[n] - 2h[n-1] + h[n-2] (二阶差分, 语义变化加速度)
      ...  (高阶差分捕获高阶多项式趋势)

    Mahler定理含义:
      若序列 h 可由 k 阶多项式描述, 则 ∇^{k+1} h ≡ 0
      → 差分阶数分布反映序列的多项式复杂度

    实现:
      - 使用 F.conv1d 实现 ∇^k (因果卷积, 直接映射 cuDNN 优化核)
      - 每阶用独立 Rank-8 投影捕获该阶的语义
      - 可学习混合权重 (softmax归一化) 决定各阶贡献

    遥测指标 mahler_dominant_order:
      argmax(softmax(order_weights)) 的平均值 — 揭示序列主要多项式阶数
    """

    def __init__(self, dim: int, max_order: int = 8, rank: int = 8):
        super().__init__()
        self.max_order = max_order
        self.dim = dim

        # 每阶一个 Rank-8 投影 (捕获该阶差分的语义)
        self.order_projs = nn.ModuleList(
            [Rank8_Projection(dim, rank) for _ in range(max_order)]
        )
        # 可学习混合权重 (初始均匀分布)
        self.order_weights = nn.Parameter(torch.ones(max_order) / max_order)
        # 输出汇聚投影
        self.out_proj = Rank8_Projection(dim, rank)
        self._last_dominant_order: float = 0.0

    @staticmethod
    def _backward_diff_k(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        因果后向 k 阶差分: ∇^k h[n] = Σ_{j=0}^{k} (-1)^j C(k,j) h[n-j]
        使用 F.conv1d 实现, 超出序列左边界的项视为 0 (合法因果零填充)。
        """
        if k == 0:
            return x
        B, T, D = x.shape
        # 差分核: [C(k,0)*(-1)^0, C(k,1)*(-1)^1, ..., C(k,k)*(-1)^k]
        # 注意 conv1d 的 'valid' 模式: kernel[0] 与序列最左端对齐
        # 为获得因果 ∇^k h[n], 需要反转核使 kernel[0] 对应 j=k (最远过去)
        # 然后左侧补 k 个零: x_padded[n+k] = h[n] → out[n+k] = ∇^k h[n+k]
        kernel = torch.tensor(
            [(-1) ** j * math.comb(k, j) for j in range(k + 1)],
            dtype=x.dtype, device=x.device,
        )
        # 翻转: kernel_flipped[0] 对应 j=k (最远过去), kernel_flipped[k] 对应 j=0 (当前)
        kernel_flipped = kernel.flip(0).reshape(1, 1, k + 1)  # [out_ch, in_ch, width]

        # x: [B, T, D] → [B*D, 1, T], 左补 k 个零保证因果性
        x_t = x.permute(0, 2, 1).reshape(B * D, 1, T)         # [B*D, 1, T]
        x_padded = F.pad(x_t, (k, 0))                          # [B*D, 1, T+k]

        out = F.conv1d(x_padded, kernel_flipped)                # [B*D, 1, T]
        return out.reshape(B, D, T).permute(0, 2, 1)            # [B, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.order_weights, dim=0)             # [max_order]
        self._last_dominant_order = float(w.argmax().item())

        result = torch.zeros_like(x)
        for k in range(self.max_order):
            diff_k = self._backward_diff_k(x, k)                # [B, T, D]
            result = result + w[k] * self.order_projs[k](diff_k)

        return self.out_proj(result)

    def get_dominant_order(self) -> float:
        return self._last_dominant_order


# ══════════════════════════════════════════════════════════════════════════════
# §11  混合加速块 V2  (HybridAcceleratedBlock_V2)
#      层 mod 3 == 0 → STA-v2  (球面拓扑 + 冲击波截断)
#      层 mod 3 == 1 → CausalTCRH V2 (素数轮 LSH + 三级整数过滤)
#      层 mod 3 == 2 → MahlerDiff    (后向差分多项式分解)
#      全部层        → Hamilton FF + 梯度检查点
# ══════════════════════════════════════════════════════════════════════════════

class HybridAcceleratedBlock_V2(nn.Module):
    """V2 混合加速变换器块: 三路注意力交替 + Hamilton FF + GradCheckpoint。"""

    LAYER_STA   = 0  # STA-v2
    LAYER_TCRH  = 1  # CausalTCRH V2 (素数轮)
    LAYER_MAHLER = 2  # Mahler 差分层

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
        prime_blend: float,
        mahler_diff_order: int,
    ):
        super().__init__()
        self.attn_type = layer_idx % 3
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ff   = nn.LayerNorm(dim)

        if self.attn_type == self.LAYER_STA:
            self.attn = Stereographic_Attention_Layer_V2(
                hidden_dim=dim,
                shockwave_threshold=shockwave_threshold,
                rank=rank,
                max_seq_len=seq_len,
                causal=True,
            )
        elif self.attn_type == self.LAYER_TCRH:
            self.attn = CausalTCRH_Attention_V2(
                dim=dim,
                hash_dim=hash_dim,
                num_buckets=num_buckets,
                hamming_thresh=hamming_thresh,
                prime_blend=prime_blend,
            )
        else:  # LAYER_MAHLER
            self.attn = MahlerDifferenceLayer(
                dim=dim,
                max_order=mahler_diff_order,
                rank=rank,
            )

        self.ff1 = BalancedHamiltonLayer(dim, factor_size, bank, rank)
        self.ff2 = BalancedHamiltonLayer(dim, factor_size, bank, rank)
        self.act = nn.GELU()

    def _attn_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm_attn(x))

    def _ff_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(self.act(self.ff1(self.norm_ff(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + cp.checkpoint(self._attn_fn, x, use_reentrant=False)
        x = x + cp.checkpoint(self._ff_fn,   x, use_reentrant=False)
        return x

    def ortho_loss(self) -> torch.Tensor:
        return self.ff1.ortho_loss() + self.ff2.ortho_loss()

    def get_attn_stat(self) -> float:
        """统一接口: STA稀疏率 / TCRH连通率 / Mahler主导阶数。"""
        if self.attn_type == self.LAYER_STA:
            return float(self.attn.last_sparsity)
        elif self.attn_type == self.LAYER_TCRH:
            return float(self.attn.get_connectivity())
        else:
            return float(self.attn.get_dominant_order())


# ══════════════════════════════════════════════════════════════════════════════
# §12  AGI V2 加速变换器
#      整合全部 12 种加速/数学方法的 AGI 字节级自回归语言模型
# ══════════════════════════════════════════════════════════════════════════════

class AGI_V2_Transformer(nn.Module):
    """
    V2 架构: 12种数学加速方法的完整整合。

    方法总览:
      ①  STA-v2          球面冲击波截断注意力
      ②  CausalTCRH V2   素数轮LSH + 三级整数过滤
      ③  Hamilton FF      四元数积前馈
      ④  TF32             张量核心加速
      ⑤  GradCheckpoint   梯度检查点
      ⑥  AsyncLoader      FineWeb异步双流
      ⑦  WeightTying      嵌入权重绑定
      ⑧  DeepSeek         外部督导注入
      ⑨  Mahler-Pascal PE 二项式基位置编码
      ⑩  P-adic Byte Emb  P-进字节嵌入
      ⑪  MahlerDiff       因果多阶差分层
      ⑫  Primorial LSH    素数谐波哈希投影

    规格:
      VOCAB=256 (字节级, 无tokenizer)
      depth=18: 6×STA-v2 + 6×CausalTCRH_V2 + 6×MahlerDiff
      dim=1024, seq_len=256, batch=12
    """

    VOCAB: int = 256

    def __init__(self, config: dict):
        super().__init__()
        dim   = config["dim"]
        fs    = config["factor_size"]
        rank  = config["fixed_rank"]
        depth = config["depth"]
        seq_len = config["seq_len"]

        num_blocks = dim // fs
        assert num_blocks % 4 == 0, (
            f"dim/factor_size={num_blocks} 须整除4 (dim={dim}, factor_size={fs})"
        )

        self.bank = WaveStructureBank(num_blocks, rank)

        # ── 加速⑦: 主词嵌入 + 权重绑定 ─────────────────────────────────────
        self.emb = nn.Embedding(self.VOCAB, dim)

        # ── 加速⑩: P-进字节嵌入 (辅助字节结构先验) ─────────────────────────
        self.padic_emb = PAdicByteEmbedding(dim, config["padic_precision"])

        # ── 加速⑨: Mahler-Pascal 位置编码 (替换随机 self.pos) ───────────────
        self.pos_enc = MahlerPascalPositionalEncoding(
            seq_len=seq_len,
            dim=dim,
            mahler_order=config["mahler_basis_order"],
        )

        self.drop = nn.Dropout(config["dropout_rate"])

        # ── UNGS 核心算子 (单一否定+封装+自指记忆) ───────────────────────────
        self.ungs_enabled = bool(config.get("ungs_enabled", True))
        self.ungs_core = (
            UNGSCore(
                dim=dim,
                rank=rank,
                rel_threshold=float(config.get("ungs_relation_threshold", 0.60)),
            )
            if self.ungs_enabled
            else None
        )
        self.ungs_closure_lambda = float(config.get("ungs_closure_lambda", 0.05))
        self.ungs_encapsulation_lambda = float(config.get("ungs_encapsulation_lambda", 0.03))
        self.ungs_self_ref_lambda = float(config.get("ungs_self_ref_lambda", 0.02))

        # ── 18层交替注意力 + Hamilton FF ─────────────────────────────────────
        self.layers = nn.ModuleList([
            HybridAcceleratedBlock_V2(
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
                prime_blend=config["prime_blend"],
                mahler_diff_order=config["mahler_diff_order"],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.VOCAB, bias=False)
        self.head.weight = self.emb.weight          # 加速⑦: 权重绑定

        self.axiom_lambda = config["axiom_lambda"]
        self._seq_len = seq_len
        self.ortho_stats_every = max(1, int(os.environ.get("AGI_ORTHO_STATS_EVERY", "1")))
        self._ortho_stats_calls = 0
        self._ortho_cache: float = 0.0
        self._has_ortho_cache = False
        self._last_ungs_loss: float = 0.0
        self._last_relation_density: float = 0.0
        self._last_hierarchy_ratio: float = 0.0
        self._last_self_ref_consistency: float = 0.0

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = x.shape

        # ── 嵌入: 词嵌入 + P-进字节嵌入 + Mahler-Pascal位置编码 ─────────────
        tok_emb   = self.emb(x)                         # [B, T, dim]  ⑦
        padic_emb = self.padic_emb(x)                   # [B, T, dim]  ⑩
        pos_emb   = self.pos_enc(T)                     # [1, T, dim]  ⑨
        h = self.drop(tok_emb + padic_emb + pos_emb)    # 三路嵌入叠加

        ungs_losses = {}
        if self.ungs_enabled and self.ungs_core is not None:
            h, ungs_losses, ungs_metrics = self.ungs_core(h, compute_losses=(targets is not None))
            self._last_relation_density = float(ungs_metrics.get("relation_density", 0.0))
            self._last_hierarchy_ratio = float(ungs_metrics.get("hierarchy_ratio", 0.0))
            self._last_self_ref_consistency = float(ungs_metrics.get("self_ref_consistency", 0.0))

        ortho = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            h = layer(h)
            ortho = ortho + layer.ortho_loss()

        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, self.VOCAB), targets.reshape(-1))
            ungs_loss = ungs_total_loss(
                ungs_losses,
                closure_lambda=self.ungs_closure_lambda,
                encapsulation_lambda=self.ungs_encapsulation_lambda,
                self_ref_lambda=self.ungs_self_ref_lambda,
            ).to(x.device)
            self._last_ungs_loss = float(ungs_loss.detach().item())
            loss = ce + self.axiom_lambda * ortho * 0.01 + ungs_loss

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
        """收集所有加速层的实时遥测数据 (含新增 Mahler 差分统计)。"""
        sta_sp, tcrh_conn, mahler_orders = [], [], []
        for l in self.layers:
            s = l.get_attn_stat()
            if l.attn_type == HybridAcceleratedBlock_V2.LAYER_STA:
                sta_sp.append(s)
            elif l.attn_type == HybridAcceleratedBlock_V2.LAYER_TCRH:
                tcrh_conn.append(s)
            else:
                mahler_orders.append(s)

        self._ortho_stats_calls += 1
        need_refresh = (
            (not self._has_ortho_cache)
            or (self._ortho_stats_calls % self.ortho_stats_every) == 0
        )
        if need_refresh:
            # Expensive path: evaluate all layer orthogonality penalties.
            with torch.no_grad():
                self._ortho_cache = float(sum(l.ortho_loss().item() for l in self.layers))
            self._has_ortho_cache = True
        ol = self._ortho_cache

        return {
            "sta_sparsity_mean":      sum(sta_sp) / max(len(sta_sp), 1),
            "tcrh_conn_mean":         sum(tcrh_conn) / max(len(tcrh_conn), 1),
            "mahler_dominant_order":  sum(mahler_orders) / max(len(mahler_orders), 1),
            "ortho_loss":             ol,
            "ungs_loss":              self._last_ungs_loss,
            "relation_density":       self._last_relation_density,
            "hierarchy_ratio":        self._last_hierarchy_ratio,
            "self_ref_consistency":   self._last_self_ref_consistency,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §13  V2 遥测 CSV (新增 mahler_dominant_order 字段)
# ══════════════════════════════════════════════════════════════════════════════

class AccelTelemetry_V2:
    FIELDS = [
        "timestamp", "chunk", "train_loss", "val_loss",
        "sta_sparsity", "tcrh_connectivity", "mahler_dominant_order",
        "ortho_loss", "ungs_loss", "relation_density", "hierarchy_ratio", "self_ref_consistency",
        "axiom_residual", "structural_pressure", "lr_dynamic", "axiom_lambda_dynamic",
        "ungs_closure_lambda_dynamic", "ungs_encapsulation_lambda_dynamic", "ungs_self_ref_lambda_dynamic",
        "control_phase", "control_phase_scale", "val_worse_streak", "val_protection_active", "val_protection_triggered",
        "controller_applied",
        "tokens_per_sec", "vram_alloc_gb",
    ]

    def __init__(self, path: str):
        self.path = path
        self._rows_since_flush = 0
        self._flush_every = max(1, int(os.environ.get("AGI_TELEMETRY_FLUSH_EVERY", "8")))
        exists = os.path.exists(path)
        self.fp = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDS)
        if not exists or os.path.getsize(path) == 0:
            self.writer.writeheader()
            self.fp.flush()

    def write(self, **kwargs):
        row = {f: kwargs.get(f, "") for f in self.FIELDS}
        self.writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every:
            self.fp.flush()
            self._rows_since_flush = 0

    def close(self):
        try:
            if self._rows_since_flush > 0:
                self.fp.flush()
            self.fp.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# §14  异步双流数据加载器  (加速⑥ — 与 V1 相同)
# ══════════════════════════════════════════════════════════════════════════════

class AsyncBufferedLoader:
    """FineWeb 异步双流加载器 (加速⑥ + DeepSeek Injection 热交换)。"""

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
        self.pin_memory = torch.cuda.is_available() and os.environ.get("AGI_PIN_MEMORY", "1").strip() == "1"

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
                    if self.pin_memory:
                        t = t.pin_memory()
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
        return data[:valid_len].view(self.batch_size, num_batches).contiguous().to(
            DEVICE,
            non_blocking=self.pin_memory,
        )

    def decode(self, token_ids: list) -> str:
        valid = bytes([i for i in token_ids if 0 < i < 256])
        return valid.decode("utf-8", errors="ignore")

    def get_bookmark(self) -> int:
        return self.current_file_index

    def stop(self):
        self.stop_event.set()


# ══════════════════════════════════════════════════════════════════════════════
# §15  工具函数
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AxiomResidualController:
    """Online controller that maps emergence residuals back to training hyperparameters."""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("adaptive_control_enabled", True))
        self.every = max(1, int(cfg.get("adaptive_control_every", 1)))
        self.warmup_chunks = max(0, int(cfg.get("control_warmup_chunks", 5)))
        self.curriculum_chunks = max(1, int(cfg.get("control_curriculum_chunks", 20)))
        self.warmup_scale = float(cfg.get("control_warmup_scale", 0.20))
        self.val_worse_tolerance = float(cfg.get("control_val_worse_tolerance", 0.01))
        self.val_worse_patience = max(1, int(cfg.get("control_val_worse_patience", 2)))
        self.protection_cooldown = max(1, int(cfg.get("control_protection_cooldown", 3)))
        self.protection_pressure_scale = float(cfg.get("control_protection_pressure_scale", 0.50))
        self.protection_lambda_decay = float(cfg.get("control_protection_lambda_decay", 0.90))
        self.target_relation_density = float(cfg.get("target_relation_density", 0.08))
        self.target_hierarchy_ratio = float(cfg.get("target_hierarchy_ratio", 0.03))
        self.target_self_ref_consistency = float(cfg.get("target_self_ref_consistency", 0.60))
        self.target_ungs_loss = float(cfg.get("target_ungs_loss", 0.80))
        self.target_generalization_gap = float(cfg.get("target_generalization_gap", 0.20))
        self.lambda_step = float(cfg.get("control_lambda_step", 0.01))
        self.axiom_step = float(cfg.get("control_axiom_step", 0.01))
        self.lr_down = float(cfg.get("control_lr_down", 0.92))
        self.lr_up = float(cfg.get("control_lr_up", 1.02))
        self.lr_min = float(cfg.get("control_lr_min", 1e-5))
        self.lr_max = float(cfg.get("control_lr_max", 1e-3))
        self.ungs_lambda_min = float(cfg.get("ungs_lambda_min", 0.0))
        self.ungs_lambda_max = float(cfg.get("ungs_lambda_max", 0.5))
        self.axiom_lambda_min = float(cfg.get("axiom_lambda_min", 0.01))
        self.axiom_lambda_max = float(cfg.get("axiom_lambda_max", 0.5))

        self.prev_val_loss: float | None = None
        self.val_worse_streak = 0
        self.protection_left = 0

    def _phase(self, chunk_counter: int) -> tuple[int, float]:
        if chunk_counter <= self.warmup_chunks:
            return 0, self.warmup_scale
        after_warmup = chunk_counter - self.warmup_chunks
        if after_warmup <= self.curriculum_chunks:
            p = after_warmup / max(1, self.curriculum_chunks)
            return 1, self.warmup_scale + (1.0 - self.warmup_scale) * p
        return 2, 1.0

    def _update_val_worse(self, avg_val: float) -> bool:
        triggered = False
        if self.prev_val_loss is not None:
            if avg_val > self.prev_val_loss * (1.0 + self.val_worse_tolerance):
                self.val_worse_streak += 1
            else:
                self.val_worse_streak = 0
            if self.val_worse_streak >= self.val_worse_patience:
                self.protection_left = self.protection_cooldown
                self.val_worse_streak = 0
                triggered = True
        self.prev_val_loss = avg_val
        return triggered

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def compute_residual(self, stats: dict, avg_train: float, avg_val: float) -> float:
        rel_gap = max(0.0, self.target_relation_density - float(stats.get("relation_density", 0.0)))
        hier_gap = max(0.0, self.target_hierarchy_ratio - float(stats.get("hierarchy_ratio", 0.0)))
        self_ref_gap = max(
            0.0,
            self.target_self_ref_consistency - float(stats.get("self_ref_consistency", 0.0)),
        )
        ungs_gap = max(0.0, float(stats.get("ungs_loss", 0.0)) - self.target_ungs_loss)
        gen_gap_raw = max(0.0, (avg_val - avg_train) - self.target_generalization_gap)
        gen_gap = gen_gap_raw / 10.0
        parts = [rel_gap, hier_gap, self_ref_gap, ungs_gap, gen_gap]
        return float(sum(parts) / len(parts))

    def apply(
        self,
        *,
        model: AGI_V2_Transformer,
        optimizer: torch.optim.Optimizer,
        residual: float,
        chunk_counter: int,
        avg_val: float,
    ) -> dict:
        curr_lr = float(optimizer.param_groups[0]["lr"])
        phase_id, phase_scale = self._phase(chunk_counter)
        protection_triggered = self._update_val_worse(avg_val)
        protection_active = self.protection_left > 0
        if (not self.enabled) or (chunk_counter % self.every != 0):
            return {
                "axiom_residual": residual,
                "structural_pressure": 1.0 * phase_scale,
                "lr_dynamic": curr_lr,
                "axiom_lambda_dynamic": float(model.axiom_lambda),
                "ungs_closure_lambda_dynamic": float(model.ungs_closure_lambda),
                "ungs_encapsulation_lambda_dynamic": float(model.ungs_encapsulation_lambda),
                "ungs_self_ref_lambda_dynamic": float(model.ungs_self_ref_lambda),
                "controller_applied": 0.0,
                "control_phase": float(phase_id),
                "control_phase_scale": phase_scale,
                "val_worse_streak": float(self.val_worse_streak),
                "val_protection_active": float(1 if protection_active else 0),
                "val_protection_triggered": float(1 if protection_triggered else 0),
            }

        high = residual > 0.10
        structural_pressure = (1.0 + min(1.0, residual * 3.0)) * phase_scale

        if protection_active:
            structural_pressure *= self.protection_pressure_scale

        if high:
            model.ungs_closure_lambda = self._clip(
                model.ungs_closure_lambda + self.lambda_step * structural_pressure,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.ungs_encapsulation_lambda = self._clip(
                model.ungs_encapsulation_lambda + self.lambda_step * 0.8 * structural_pressure,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.ungs_self_ref_lambda = self._clip(
                model.ungs_self_ref_lambda + self.lambda_step * 0.6 * structural_pressure,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.axiom_lambda = self._clip(
                model.axiom_lambda + self.axiom_step * structural_pressure,
                self.axiom_lambda_min,
                self.axiom_lambda_max,
            )
            new_lr = self._clip(curr_lr * self.lr_down, self.lr_min, self.lr_max)
        else:
            model.ungs_closure_lambda = self._clip(
                model.ungs_closure_lambda - self.lambda_step * 0.5,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.ungs_encapsulation_lambda = self._clip(
                model.ungs_encapsulation_lambda - self.lambda_step * 0.4,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.ungs_self_ref_lambda = self._clip(
                model.ungs_self_ref_lambda - self.lambda_step * 0.3,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.axiom_lambda = self._clip(
                model.axiom_lambda - self.axiom_step * 0.3,
                self.axiom_lambda_min,
                self.axiom_lambda_max,
            )
            new_lr = self._clip(curr_lr * self.lr_up, self.lr_min, self.lr_max)

        if protection_active:
            model.ungs_closure_lambda = self._clip(
                model.ungs_closure_lambda * self.protection_lambda_decay,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.ungs_encapsulation_lambda = self._clip(
                model.ungs_encapsulation_lambda * self.protection_lambda_decay,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.ungs_self_ref_lambda = self._clip(
                model.ungs_self_ref_lambda * self.protection_lambda_decay,
                self.ungs_lambda_min,
                self.ungs_lambda_max,
            )
            model.axiom_lambda = self._clip(
                model.axiom_lambda * self.protection_lambda_decay,
                self.axiom_lambda_min,
                self.axiom_lambda_max,
            )
            # During protection we avoid over-constraining; ease lr decay.
            new_lr = self._clip(max(new_lr, curr_lr), self.lr_min, self.lr_max)

        if self.protection_left > 0:
            self.protection_left -= 1

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        return {
            "axiom_residual": residual,
            "structural_pressure": structural_pressure,
            "lr_dynamic": new_lr,
            "axiom_lambda_dynamic": float(model.axiom_lambda),
            "ungs_closure_lambda_dynamic": float(model.ungs_closure_lambda),
            "ungs_encapsulation_lambda_dynamic": float(model.ungs_encapsulation_lambda),
            "ungs_self_ref_lambda_dynamic": float(model.ungs_self_ref_lambda),
            "controller_applied": 1.0,
            "control_phase": float(phase_id),
            "control_phase_scale": phase_scale,
            "val_worse_streak": float(self.val_worse_streak),
            "val_protection_active": float(1 if protection_active else 0),
            "val_protection_triggered": float(1 if protection_triggered else 0),
        }


def print_boot_banner(model: AGI_V2_Transformer, cfg: dict):
    props = torch.cuda.get_device_properties(DEVICE)
    total_vram = props.total_memory / 1024 ** 3
    param_count = count_params(model)

    n_sta    = sum(1 for l in model.layers if l.attn_type == HybridAcceleratedBlock_V2.LAYER_STA)
    n_tcrh   = sum(1 for l in model.layers if l.attn_type == HybridAcceleratedBlock_V2.LAYER_TCRH)
    n_mahler = sum(1 for l in model.layers if l.attn_type == HybridAcceleratedBlock_V2.LAYER_MAHLER)

    print("=" * 90)
    print("AGI 联合加速训练系统  V2  /  AGI Joint Accelerated Trainer V2")
    print("整合 12 种数学加速方法 (含全部 h2q_prime_engine 新数学)")
    print("=" * 90)
    print(f"  GPU          : {torch.cuda.get_device_name(DEVICE)}")
    print(f"  VRAM 总量    : {total_vram:.1f} GB")
    print(f"  参数量       : {param_count:,}")
    print(f"  架构 (扩展)  : {cfg['depth']} 层 三路交替")
    print(f"                  {n_sta} STA-v2  |  {n_tcrh} CausalTCRH-V2  |  {n_mahler} MahlerDiff")
    print(f"  dim/rank     : {cfg['dim']} / {cfg['fixed_rank']}")
    print(f"  seq_len      : {cfg['seq_len']}  (V1: 128 → V2: {cfg['seq_len']})")
    print(f"  batch_size   : {cfg['batch_size']}")
    print()
    print("  已启用加速方法 (12种):")
    print(f"    [1]  STA-v2       球面冲击波截断 lam={cfg['shockwave_threshold']:.4f}  ({n_sta} 层)")
    print(f"    [2]  CausalTCRH   素数轮LSH+Hamming  ({n_tcrh} 层, blend={cfg['prime_blend']})")
    print(f"    [3]  Hamilton FF  Rank-8四元数积 factor_size={cfg['factor_size']}")
    print(f"    [4]  TF32         matmul+cudnn allow_tf32=True")
    print(f"    [5]  GradCkpt     torch.utils.checkpoint (全部层)")
    print(f"    [6]  AsyncLoader  FineWeb双流 chunk={cfg['chunk_size_mb']}MB + Injection热交换")
    print(f"    [7]  WeightTie    embedding == head.weight")
    print(f"    [8]  DeepSeek     督导注入 every={cfg['supervise_every']} chunks")
    print(f"    [9]  Mahler-Pascal PE  Pascal二项式基位置编码  order={cfg['mahler_basis_order']}")
    print(f"    [10] P-adic Emb   2-进字节结构嵌入  precision={cfg['padic_precision']}位")
    print(f"    [11] MahlerDiff   因果后向差分层  max_order={cfg['mahler_diff_order']}  ({n_mahler} 层)")
    print(f"    [12] Primorial LSH 素数谐波投影  prime_blend={cfg['prime_blend']}")
    print(
        "    [UNGS] 单一否定生成 core="
        f"{cfg['ungs_enabled']} ("
        f"closure={cfg['ungs_closure_lambda']}, "
        f"encap={cfg['ungs_encapsulation_lambda']}, "
        f"self_ref={cfg['ungs_self_ref_lambda']})"
    )
    print("=" * 90)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §16  V2 联合训练主循环
# ══════════════════════════════════════════════════════════════════════════════

def train_joint_v2(cfg: dict):
    set_global_seed(int(cfg.get("seed", 42)))
    resume_file_index = 0
    chunk_counter     = 0
    best_loss         = float("inf")

    # ── 模型初始化 ──────────────────────────────────────────────────────────
    model = AGI_V2_Transformer(cfg).to(DEVICE)
    controller = AxiomResidualController(cfg)

    adamw_kwargs = {
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
    }
    if DEVICE.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    # ── 恢复检查点 ──────────────────────────────────────────────────────────
    ckpt_path = cfg["checkpoint_path"]
    if os.path.exists(ckpt_path):
        print(f"[Train] 恢复存档: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            load_result = model.load_state_dict(sanitize_state(ckpt["model"]), strict=False)
            if load_result.missing_keys:
                print(f"[Train] state_dict missing_keys={len(load_result.missing_keys)} (UNGS扩展后可预期)")
            if load_result.unexpected_keys:
                print(f"[Train] state_dict unexpected_keys={len(load_result.unexpected_keys)}")
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
        print("[Train] 未找到存档，从头开始 (V2 全新实验)")

    for pg in opt.param_groups:
        pg["lr"] = cfg["lr"]

    print_boot_banner(model, cfg)

    # ── 数据加载器 (加速⑥) ─────────────────────────────────────────────────
    loader = AsyncBufferedLoader(cfg, resume_file_index)

    # ── DeepSeek 督导器 (加速⑧) ────────────────────────────────────────────
    supervisor = None
    if _SUPERVISOR_OK and cfg.get("supervise_every", 0) > 0:
        inj_dir = os.path.join(cfg["buffer_dir"], "Injection")
        try:
            supervisor = DeepSeekSupervisor(
                injection_dir=inj_dir,
                every_n_chunks=cfg["supervise_every"],
                gen_tokens=cfg.get("supervise_gen_tokens", 256),
            )
            print(f"[Train] DeepSeek supervisor enabled (every {cfg['supervise_every']} chunks)")
        except Exception as _sup_err:
            print(f"[Train] DeepSeek supervisor init failed (skipped): {_sup_err}")
            supervisor = None
    else:
        print("[Train] DeepSeek supervisor disabled")

    # ── 遥测 ────────────────────────────────────────────────────────────────
    telemetry = AccelTelemetry_V2(cfg["telemetry_csv"])
    core_telemetry = CoreTelemetryCSV(to_core_telemetry_path(cfg["telemetry_csv"]))

    print("[Train] 等待初始数据 ...")
    current_data = loader.load_next_chunk()
    if current_data is None:
        print("[Train] 数据加载失败，退出")
        return

    print(f"[Train] 🚀 AGI V2 开始训练 chunk={chunk_counter}/{cfg['total_chunks']}")
    print(f"[Train]    12种数学加速方法全部激活")

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

            # ── 验证阶段 (Rolling Horizon) ─────────────────────────────────
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
                cfg["batch_size"] * seq_len * train_steps
                / max(time.time() - chunk_t0, 1e-6)
            )
            current_data = future_data
            chunk_counter += 1

            # ── 加速统计 + 闭环控制 ───────────────────────────────────────
            stats = model.accel_stats()
            residual = controller.compute_residual(stats, avg_train=avg_train, avg_val=avg_val)
            ctrl = controller.apply(
                model=model,
                optimizer=opt,
                residual=residual,
                chunk_counter=chunk_counter,
                avg_val=avg_val,
            )
            stats.update(ctrl)
            total_time = time.time() - t0

            print(
                f"\n{'='*75}\n"
                f"  Chunk {chunk_counter:5d} | train={avg_train:.4f} val={avg_val:.4f} "
                f"diff={avg_val-avg_train:+.4f}\n"
                f"  STA稀疏率={stats['sta_sparsity_mean']*100:.1f}%  "
                f"TCRH连通率={stats['tcrh_conn_mean']*100:.1f}%  "
                f"Mahler主导阶={stats['mahler_dominant_order']:.1f}  "
                f"ortho={stats['ortho_loss']:.3f}  UNGS={stats['ungs_loss']:.4f}\n"
                f"  关系密度={stats['relation_density']:.4f}  "
                f"层级压缩比={stats['hierarchy_ratio']:.4f}  "
                f"自指一致性={stats['self_ref_consistency']:.4f}\n"
                f"  残差={stats['axiom_residual']:.4f}  "
                f"压力={stats['structural_pressure']:.3f}  "
                f"lr={stats['lr_dynamic']:.6f}  "
                f"axiomλ={stats['axiom_lambda_dynamic']:.4f}  "
                f"phase={int(stats['control_phase'])}({stats['control_phase_scale']:.2f})  "
                f"protect={int(stats['val_protection_active'])}\n"
                f"  速度={int(tokens_per_sec)} tok/s  "
                f"VRAM={get_vram_gb():.2f}GB  "
                f"时间={total_time:.1f}s  "
                f"文件索引={loader.get_bookmark()}"
            )

            core_metrics = compute_core_metrics(
                train_loss=avg_train,
                val_loss=avg_val,
                ortho_loss=float(stats["ortho_loss"]),
                tokens_per_sec=tokens_per_sec,
                axiom_lambda=float(model.axiom_lambda),
                stats=stats,
            )
            ts = datetime.utcnow().isoformat() + "Z"

            # ── 遥测写入 ───────────────────────────────────────────────────
            telemetry.write(
                timestamp=ts,
                chunk=chunk_counter,
                train_loss=f"{avg_train:.6f}",
                val_loss=f"{avg_val:.6f}",
                sta_sparsity=f"{stats['sta_sparsity_mean']:.4f}",
                tcrh_connectivity=f"{stats['tcrh_conn_mean']:.4f}",
                mahler_dominant_order=f"{stats['mahler_dominant_order']:.2f}",
                ortho_loss=f"{stats['ortho_loss']:.4f}",
                ungs_loss=f"{stats['ungs_loss']:.6f}",
                relation_density=f"{stats['relation_density']:.6f}",
                hierarchy_ratio=f"{stats['hierarchy_ratio']:.6f}",
                self_ref_consistency=f"{stats['self_ref_consistency']:.6f}",
                axiom_residual=f"{stats['axiom_residual']:.6f}",
                structural_pressure=f"{stats['structural_pressure']:.6f}",
                lr_dynamic=f"{stats['lr_dynamic']:.8f}",
                axiom_lambda_dynamic=f"{stats['axiom_lambda_dynamic']:.6f}",
                ungs_closure_lambda_dynamic=f"{stats['ungs_closure_lambda_dynamic']:.6f}",
                ungs_encapsulation_lambda_dynamic=f"{stats['ungs_encapsulation_lambda_dynamic']:.6f}",
                ungs_self_ref_lambda_dynamic=f"{stats['ungs_self_ref_lambda_dynamic']:.6f}",
                control_phase=f"{stats['control_phase']:.0f}",
                control_phase_scale=f"{stats['control_phase_scale']:.6f}",
                val_worse_streak=f"{stats['val_worse_streak']:.0f}",
                val_protection_active=f"{stats['val_protection_active']:.0f}",
                val_protection_triggered=f"{stats['val_protection_triggered']:.0f}",
                controller_applied=f"{stats['controller_applied']:.0f}",
                tokens_per_sec=f"{tokens_per_sec:.0f}",
                vram_alloc_gb=f"{get_vram_gb():.3f}",
            )
            core_telemetry.write(timestamp=ts, chunk=chunk_counter, metrics=core_metrics)

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
                ckpt_path,
            )

            # ── DeepSeek 督导注入 (加速⑧) ──────────────────────────────────
            if supervisor is not None:
                supervisor.maybe_supervise(chunk_counter, model, loader, DEVICE)

            # ── 思维流生成 (每 5 chunk) ─────────────────────────────────────
            if chunk_counter % 5 == 0:
                print("\n[思维流-V2] 自由联想生成 (12种加速协同):")
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
        try:
            torch.save(
                {
                    "chunk_counter":  chunk_counter,
                    "model":          model.state_dict(),
                    "optimizer":      opt.state_dict(),
                    "best_loss":      best_loss,
                    "dataset_offset": loader.get_bookmark(),
                },
                ckpt_path,
            )
            print(f"[Train] 状态已保存: {ckpt_path}")
        except Exception:
            print("[Train] 紧急保存失败")
        loader.stop()
        telemetry.close()
        core_telemetry.close()
        print(
            f"\n[Train] V2 完成 — chunk={chunk_counter}, best_loss={best_loss:.4f}, "
            f"VRAM={get_vram_gb():.2f}GB"
        )


# ══════════════════════════════════════════════════════════════════════════════
# §17  命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "AGI 联合加速训练系统 V2 — 12种数学加速方法\n"
            "新增: ⑨Mahler-Pascal PE  ⑩P-adic字节嵌入  ⑪Mahler差分层  ⑫素数轮LSH"
        )
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
    p.add_argument("--eval-window-multiplier", type=int, default=CONFIG["eval_window_multiplier"])
    p.add_argument("--checkpoint-path", type=str, default=CONFIG["checkpoint_path"])
    p.add_argument("--best-model-path", type=str, default=CONFIG["best_model_path"])
    p.add_argument("--telemetry-csv",   type=str, default=CONFIG["telemetry_csv"])
    p.add_argument("--shockwave-threshold", type=float, default=CONFIG["shockwave_threshold"])
    p.add_argument("--hash-dim",     type=int,   default=CONFIG["hash_dim"])
    p.add_argument("--ungs-enabled", type=int, default=int(CONFIG["ungs_enabled"]))
    p.add_argument("--ungs-closure-lambda", type=float, default=CONFIG["ungs_closure_lambda"])
    p.add_argument("--ungs-encapsulation-lambda", type=float, default=CONFIG["ungs_encapsulation_lambda"])
    p.add_argument("--ungs-self-ref-lambda", type=float, default=CONFIG["ungs_self_ref_lambda"])
    p.add_argument("--ungs-relation-threshold", type=float, default=CONFIG["ungs_relation_threshold"])
    p.add_argument("--adaptive-control-enabled", type=int, default=int(CONFIG["adaptive_control_enabled"]))
    p.add_argument("--adaptive-control-every", type=int, default=CONFIG["adaptive_control_every"])
    p.add_argument("--control-warmup-chunks", type=int, default=CONFIG["control_warmup_chunks"])
    p.add_argument("--control-curriculum-chunks", type=int, default=CONFIG["control_curriculum_chunks"])
    p.add_argument("--control-warmup-scale", type=float, default=CONFIG["control_warmup_scale"])
    p.add_argument("--control-val-worse-tolerance", type=float, default=CONFIG["control_val_worse_tolerance"])
    p.add_argument("--control-val-worse-patience", type=int, default=CONFIG["control_val_worse_patience"])
    p.add_argument("--control-protection-cooldown", type=int, default=CONFIG["control_protection_cooldown"])
    p.add_argument("--control-protection-pressure-scale", type=float, default=CONFIG["control_protection_pressure_scale"])
    p.add_argument("--control-protection-lambda-decay", type=float, default=CONFIG["control_protection_lambda_decay"])
    p.add_argument("--target-relation-density", type=float, default=CONFIG["target_relation_density"])
    p.add_argument("--target-hierarchy-ratio", type=float, default=CONFIG["target_hierarchy_ratio"])
    p.add_argument("--target-self-ref-consistency", type=float, default=CONFIG["target_self_ref_consistency"])
    p.add_argument("--target-ungs-loss", type=float, default=CONFIG["target_ungs_loss"])
    p.add_argument("--target-generalization-gap", type=float, default=CONFIG["target_generalization_gap"])
    p.add_argument("--control-lambda-step", type=float, default=CONFIG["control_lambda_step"])
    p.add_argument("--control-axiom-step", type=float, default=CONFIG["control_axiom_step"])
    p.add_argument("--control-lr-down", type=float, default=CONFIG["control_lr_down"])
    p.add_argument("--control-lr-up", type=float, default=CONFIG["control_lr_up"])
    p.add_argument("--control-lr-min", type=float, default=CONFIG["control_lr_min"])
    p.add_argument("--control-lr-max", type=float, default=CONFIG["control_lr_max"])
    p.add_argument("--ungs-lambda-min", type=float, default=CONFIG["ungs_lambda_min"])
    p.add_argument("--ungs-lambda-max", type=float, default=CONFIG["ungs_lambda_max"])
    p.add_argument("--axiom-lambda-min", type=float, default=CONFIG["axiom_lambda_min"])
    p.add_argument("--axiom-lambda-max", type=float, default=CONFIG["axiom_lambda_max"])
    p.add_argument("--num-buckets",  type=int,   default=CONFIG["num_buckets"])
    p.add_argument("--hamming-thresh", type=int, default=CONFIG["hamming_thresh"])
    # V2 新增参数
    p.add_argument("--prime-blend",     type=float, default=CONFIG["prime_blend"],
                   help="素数谐波LSH占比 (0.0=全随机, 1.0=全素数谐波)")
    p.add_argument("--mahler-basis-order", type=int, default=CONFIG["mahler_basis_order"],
                   help="Mahler-Pascal位置编码阶数K")
    p.add_argument("--padic-precision",  type=int, default=CONFIG["padic_precision"],
                   help="P-进字节嵌入位数 (建议8)")
    p.add_argument("--mahler-diff-order", type=int, default=CONFIG["mahler_diff_order"],
                   help="Mahler差分层最大阶数")
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    return p


def main():
    args = build_parser().parse_args()
    cfg = dict(CONFIG)
    cfg.update({
        "dim":                    args.dim,
        "depth":                  args.depth,
        "seq_len":                args.seq_len,
        "batch_size":             args.batch_size,
        "lr":                     args.lr,
        "total_chunks":           args.total_chunks,
        "chunk_size_mb":          args.chunk_size_mb,
        "source_dir":             args.source_dir,
        "buffer_dir":             args.buffer_dir,
        "supervise_every":        args.supervise_every,
        "eval_window_multiplier": args.eval_window_multiplier,
        "checkpoint_path":        args.checkpoint_path,
        "best_model_path":        args.best_model_path,
        "telemetry_csv":          args.telemetry_csv,
        "shockwave_threshold":    args.shockwave_threshold,
        "hash_dim":               args.hash_dim,
        "ungs_enabled":           bool(args.ungs_enabled),
        "ungs_closure_lambda":    args.ungs_closure_lambda,
        "ungs_encapsulation_lambda": args.ungs_encapsulation_lambda,
        "ungs_self_ref_lambda":   args.ungs_self_ref_lambda,
        "ungs_relation_threshold": args.ungs_relation_threshold,
        "adaptive_control_enabled": bool(args.adaptive_control_enabled),
        "adaptive_control_every": args.adaptive_control_every,
        "control_warmup_chunks": args.control_warmup_chunks,
        "control_curriculum_chunks": args.control_curriculum_chunks,
        "control_warmup_scale": args.control_warmup_scale,
        "control_val_worse_tolerance": args.control_val_worse_tolerance,
        "control_val_worse_patience": args.control_val_worse_patience,
        "control_protection_cooldown": args.control_protection_cooldown,
        "control_protection_pressure_scale": args.control_protection_pressure_scale,
        "control_protection_lambda_decay": args.control_protection_lambda_decay,
        "target_relation_density": args.target_relation_density,
        "target_hierarchy_ratio": args.target_hierarchy_ratio,
        "target_self_ref_consistency": args.target_self_ref_consistency,
        "target_ungs_loss": args.target_ungs_loss,
        "target_generalization_gap": args.target_generalization_gap,
        "control_lambda_step": args.control_lambda_step,
        "control_axiom_step": args.control_axiom_step,
        "control_lr_down": args.control_lr_down,
        "control_lr_up": args.control_lr_up,
        "control_lr_min": args.control_lr_min,
        "control_lr_max": args.control_lr_max,
        "ungs_lambda_min": args.ungs_lambda_min,
        "ungs_lambda_max": args.ungs_lambda_max,
        "axiom_lambda_min": args.axiom_lambda_min,
        "axiom_lambda_max": args.axiom_lambda_max,
        "num_buckets":            args.num_buckets,
        "hamming_thresh":         args.hamming_thresh,
        # V2 新增
        "prime_blend":            args.prime_blend,
        "mahler_basis_order":     args.mahler_basis_order,
        "padic_precision":        args.padic_precision,
        "mahler_diff_order":      args.mahler_diff_order,
        "seed":                   args.seed,
    })
    train_joint_v2(cfg)


if __name__ == "__main__":
    main()
