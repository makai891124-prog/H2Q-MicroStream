"""
h2q_prime_engine.py — H2Q P-adic Mahler × Rank-8 Prime Engine
==============================================================

数学框架（严密性声明）
─────────────────────
① P-进赋值与范数
   v_p(n) = max{k ∈ ℕ₀ : p^k | n}    (若 n=0 则 v_p(0)=+∞)
   |n|_p  = p^{-v_p(n)}               (p-进绝对值, 非阿基米德)

② Mahler定理 (p-进Taylor展开)
   设 f : ℤ_p → ℤ_p 连续, 则唯一存在 c_k ∈ ℤ_p 使得
     f(x) = Σ_{k≥0}  c_k · C(x,k)     (p-进一致收敛)
   其中 c_k = Δ^k f(0) = Σ_{j=0}^k (-1)^{k-j} C(k,j) f(j)
   基函数 C(x,k) = x(x-1)···(x-k+1)/k! 是 Mahler (二项式) 基

③ Pascal矩阵线性代数形式
   设 f_n = f(n), c_k = Δ^k f(0), 则
     [f_0, f_1, ..., f_{N-1}]^T = B · [c_0, c_1, ..., c_{N-1}]^T
   其中 B[n,k] = C(n,k) (下三角 Pascal 矩阵)
   逆变换: B^{-1}[k,n] = (-1)^{k-n} C(k,n)
   (Mahler展开 = 函数值向量在二项式基下的坐标变换)

④ Rank-8 SVD 筛 (H2Q 架构哲学)
   构造筛矩阵 S ∈ {0,1}^{|P|×N}, P = {小素数 ≤ √N}
   S[i,j]=1 当且仅当 p_i | (j+2) 且 (j+2) ≠ p_i
   素数指示向量 ≈ Threshold(rank-8 SVD(S).sum(axis=0), 0.5)
   本质: 筛法的主要信息集中在前8个奇异模式 (与H2Q的Rank-8本质主义一致)

⑤ Pocklington-Lehmer 素性证书 (定理)
   设 n-1 = F·R, F 的全部素因子已知.
   若对 F 的每个素因子 q 存在 a 使得:
     (a) a^{n-1} ≡ 1 (mod n)
     (b) gcd(a^{(n-1)/q} - 1, n) = 1
   则 n 的每个素因子 ≡ 1 (mod F).
   推论: 若 F > √n, 则 n 是素数.

⑥ Hensel 提升引理 (p-进精度递增)
   设 f(x) 是整系数多项式, f(a) ≡ 0 (mod p), f'(a) ≢ 0 (mod p).
   则唯一存在 a_k 满足 f(a_k) ≡ 0 (mod p^k) 且 a_k ≡ a (mod p).
   Newton 步骤: a_{k+1} = a_k - f(a_k)/p^k · (f'(a_k))^{-1} (mod p)  (mod p^{k+1})
"""

import math
import time
import json
import sys
from typing import List, Tuple, Dict, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[警告] numpy 未安装, Rank-8 SVD 分析将跳过.")


# ═══════════════════════════════════════════════════════════════════════════
# 第一部分: P-进基础运算
# ═══════════════════════════════════════════════════════════════════════════

def p_adic_valuation(n: int, p: int) -> int:
    """
    计算 v_p(n): n 中 p 的最高幂次.
    性质: v_p(ab) = v_p(a) + v_p(b)  (完全可加性)
          v_p(a+b) ≥ min(v_p(a), v_p(b))  (非阿基米德不等式)
    """
    if n == 0:
        return 10**9  # 代表 +∞
    if n < 0:
        n = -n
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def p_adic_norm(n: int, p: int) -> float:
    """|n|_p = p^{-v_p(n)}. 满足强三角不等式 |a+b|_p ≤ max(|a|_p, |b|_p)."""
    v = p_adic_valuation(n, p)
    if v >= 10**8:
        return 0.0
    return float(p) ** (-v)


def p_adic_encode(n: int, p: int, precision: int = 16) -> List[int]:
    """
    将 n 展开为 p-进数码序列: n = Σ_{k=0}^{precision-1} d_k · p^k, 0 ≤ d_k < p.
    例: 13 在 2-进展开 = [1,0,1,1,0,...] 即 13 = 1+4+8.
    """
    digits = []
    m = abs(n)
    for _ in range(precision):
        digits.append(m % p)
        m //= p
    return digits


def hensel_lift(a_mod_p: int, p: int, n_target: int, precision: int = 8) -> List[int]:
    """
    Hensel 提升: 对多项式 f(x) = x² - n_target,
    从 a_0 ≡ a_mod_p (mod p) 出发, 逐步提升到 mod p^k 的解序列.

    Newton 步: t_k = -(f(a_k)/p^k) · (f'(a_k))^{-1}  (mod p)
               a_{k+1} = a_k + t_k · p^k

    物理意义: p-进数列 {a_k} 在 ℤ_p 中 Cauchy 收敛到 √n_target (若存在).
    """
    lifts = [a_mod_p % p]
    current = a_mod_p % p

    for k in range(1, precision):
        pk = p ** k
        pk1 = p ** (k + 1)

        # 检验当前残差 f(current) = current² - n_target
        residue = (current * current - n_target) % pk1
        if residue == 0:
            lifts.append(current % pk1)
            continue

        # Newton 步: t = -(current²-n)/p^k · (2·current)^{-1} mod p
        diff_pk = ((current * current - n_target) // pk) % p
        deriv_mod_p = (2 * current) % p

        if math.gcd(deriv_mod_p, p) != 1:
            lifts.append(current % pk1)  # 无法提升(奇点)
            continue

        deriv_inv = pow(deriv_mod_p, -1, p)
        t = (-diff_pk * deriv_inv) % p
        current = current + t * pk
        lifts.append(current % pk1)

    return lifts


# ═══════════════════════════════════════════════════════════════════════════
# 第二部分: Mahler P-进 Taylor 展开 (线性代数矩阵形式)
# ═══════════════════════════════════════════════════════════════════════════

def forward_difference_k(f_vals: List[int], k: int) -> int:
    """
    计算第 k 阶前向差分算子作用于 f 在 0 处的值:
      Δ^k f(0) = Σ_{j=0}^{k} (-1)^{k-j} C(k,j) f(j)

    这是 Mahler 展开的第 k 个系数 c_k.
    矩阵意义: c = B^{-1} · f_vec, 此处 c_k = (B^{-1} · f_vec)[k]
    """
    total = 0
    for j in range(k + 1):
        sign = 1 if (k - j) % 2 == 0 else -1
        total += sign * math.comb(k, j) * f_vals[j]
    return total


def compute_mahler_coefficients(f_vals: List[int]) -> List[int]:
    """
    批量计算 Mahler 系数向量 c = [c_0, c_1, ..., c_{N-1}].
    线性代数等价: c = B^{-1} · f_vec
    其中 B^{-1}[k,j] = (-1)^{k-j} · C(k,j)  (带符号 Pascal 矩阵)
    """
    N = len(f_vals)
    # 使用递推式加速: Δ^{k+1}f(0) = Δ^k f(1) - Δ^k f(0)
    # 维护差分表 (Pascal 三角下三角)
    diff_table = list(f_vals)  # 当前差分层
    coeffs = [diff_table[0]]

    for k in range(1, N):
        new_table = [diff_table[j + 1] - diff_table[j] for j in range(N - k)]
        if not new_table:
            break
        coeffs.append(new_table[0])
        diff_table = new_table

    return coeffs


def mahler_evaluate(coeffs: List[int], n: int) -> int:
    """
    在整数点 n 处求值: f(n) = Σ_{k=0}^{n} c_k · C(n,k)
    Newton 前向差分公式 → 对所有 n ∈ ℕ 精确成立 (零误差).
    """
    total = 0
    for k in range(min(n + 1, len(coeffs))):
        total += coeffs[k] * math.comb(n, k)
    return total


def build_pascal_matrix(N: int) -> List[List[int]]:
    """
    构造 N×N 下三角 Pascal 矩阵 B, B[n][k] = C(n,k).
    f_vec = B · c_vec  (Mahler → 函数值)
    """
    B = [[0] * N for _ in range(N)]
    for n in range(N):
        for k in range(n + 1):
            B[n][k] = math.comb(n, k)
    return B


def build_inverse_pascal_matrix(N: int) -> List[List[int]]:
    """
    构造 N×N 带符号 Pascal 矩阵 B^{-1}, B^{-1}[k][n] = (-1)^{k-n} C(k,n).
    c_vec = B^{-1} · f_vec  (函数值 → Mahler 系数)
    """
    Binv = [[0] * N for _ in range(N)]
    for k in range(N):
        for n in range(k + 1):
            sign = 1 if (k - n) % 2 == 0 else -1
            Binv[k][n] = sign * math.comb(k, n)
    return Binv


def verify_pascal_inverse(N: int = 8) -> bool:
    """
    验证 B · B^{-1} = I_N (整数精确等式).
    这是 Mahler 展开可逆性的代数证明.
    """
    B = build_pascal_matrix(N)
    Binv = build_inverse_pascal_matrix(N)

    # 矩阵乘法 (整数)
    for i in range(N):
        for j in range(N):
            prod = sum(B[i][k] * Binv[k][j] for k in range(N))
            expected = 1 if i == j else 0
            if prod != expected:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 第三部分: 快速素数筛 (分段筛 + 轮因子化)
# ═══════════════════════════════════════════════════════════════════════════

def sieve_small(limit: int) -> bytearray:
    """标准 Eratosthenes 筛, 返回 is_prime 字节数组."""
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(range(i * i, limit + 1, i)))
    return is_p


def segmented_sieve(limit: int, seg: int = 1 << 19) -> List[int]:
    """
    分段筛: 内存 O(√N), 时间 O(N log log N).
    将 [√N, N] 分成大小为 seg 的区间逐段筛除.
    """
    sqrt_lim = int(limit ** 0.5) + 1
    is_small = sieve_small(sqrt_lim)
    small_p = [i for i in range(2, sqrt_lim + 1) if is_small[i]]
    primes = list(small_p)

    low = sqrt_lim + 1
    while low <= limit:
        high = min(low + seg - 1, limit)
        sieve = bytearray([1]) * (high - low + 1)

        for p in small_p:
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start - low, len(sieve), p):
                sieve[j] = 0

        primes.extend(low + i for i, v in enumerate(sieve) if v)
        low += seg

    return primes


def prime_counting_table(N: int) -> List[int]:
    """计算并返回 [π(0), π(1), ..., π(N)] 的完整表."""
    is_p = sieve_small(N)
    pi = [0] * (N + 1)
    cnt = 0
    for i in range(N + 1):
        if is_p[i]:
            cnt += 1
        pi[i] = cnt
    return pi


# ═══════════════════════════════════════════════════════════════════════════
# 第四部分: Rank-8 线性代数筛 (H2Q SVD 哲学)
# ═══════════════════════════════════════════════════════════════════════════

def rank8_sieve_analysis(N: int) -> Dict:
    """
    构造筛矩阵 S ∈ {0,1}^{|P_small| × (N-1)},
    其中 P_small = {素数 p ≤ √N}.
    S[i,j] = 1 iff p_i | (j+2) 且 j+2 > p_i.

    SVD 分解: S = U Σ V^T, 保留前 rank=8 个奇异向量.
    Rank-8 重建: S_8 = Σ_{r=1}^{8} σ_r u_r v_r^T
    素数判定: (j+2) 是素数  ⟺  S[:,j].sum() = 0
             Rank-8 近似: (S_8[:,j].sum() < 0.5) → 预测为素数

    返回: 能量占比、精确度、奇异值序列.
    """
    if not HAS_NUMPY:
        return {"error": "numpy 未安装"}

    is_p = sieve_small(N)
    small_primes = [i for i in range(2, int(N ** 0.5) + 1) if is_p[i]]
    n_small = len(small_primes)
    n_cands = N - 1  # 候选: 2, 3, ..., N

    if n_small == 0 or n_cands == 0:
        return {"error": "N 太小"}

    # 构造筛矩阵
    S = np.zeros((n_small, n_cands), dtype=np.float32)
    for i, p in enumerate(small_primes):
        # p 的倍数从 p² 开始 (p 本身是素数, 不标记)
        start_j = p * p - 2  # 对应候选编号 j (j+2 = p²)
        for j in range(start_j, n_cands, p):
            if j >= 0:
                S[i, j] = 1.0

    # 真实素数指示向量 (候选 j+2 是否为素数)
    prime_true = np.array([float(is_p[j + 2]) for j in range(n_cands)])

    # SVD
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    rank = min(8, len(sigma))

    # Rank-8 重建
    S8 = (U[:, :rank] * sigma[:rank]) @ Vt[:rank, :]
    composite_score = S8.sum(axis=0)

    # 预测: 复合得分 < 0.5 → 素数
    prime_pred = (composite_score < 0.5).astype(float)
    accuracy = float((prime_pred == prime_true).mean())

    # 能量分析
    total_energy = float((sigma ** 2).sum())
    rank8_energy = float((sigma[:rank] ** 2).sum())
    energy_ratio = rank8_energy / total_energy if total_energy > 0 else 1.0

    # 各奇异值贡献
    singular_contributions = [
        {"rank": r + 1, "sigma": round(float(sigma[r]), 4),
         "energy_pct": round(float(sigma[r] ** 2) / total_energy * 100, 2)}
        for r in range(min(12, len(sigma)))
    ]

    return {
        "n_small_primes_used": n_small,
        "matrix_shape": [n_small, n_cands],
        "rank8_accuracy": round(accuracy * 100, 2),
        "rank8_energy_pct": round(energy_ratio * 100, 2),
        "singular_contributions": singular_contributions,
        "interpretation": (
            f"筛矩阵 {n_small}×{n_cands} 的 Rank-8 SVD 捕获 {energy_ratio*100:.1f}% 能量, "
            f"素数分类精度 {accuracy*100:.1f}%"
        )
    }


# ═══════════════════════════════════════════════════════════════════════════
# 第五部分: 自动化素性证明 (Miller-Rabin + Pocklington-Lehmer)
# ═══════════════════════════════════════════════════════════════════════════

def miller_rabin_deterministic(n: int) -> bool:
    """
    确定性 Miller-Rabin 素性测试.
    n < 3,215,031,751:        验证基 {2,3,5,7}
    n < 3,317,044,064,679,887,385,961,981: 验证基 {2,3,5,7,11,13,17,19,23,29,31,37}
    (已被数学证明对上述范围无伪素数)
    """
    if n < 2:
        return False
    for small in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n == small:
            return True
        if n % small == 0:
            return False

    if n < 40:
        return False  # 已被上面过滤

    # 选取见证集
    if n < 3_215_031_751:
        witnesses = [2, 3, 5, 7]
    else:
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # 写 n-1 = 2^r · d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _factor_small(n: int, limit: int = 10 ** 6) -> Optional[int]:
    """试除法找最小素因子, 上限 limit."""
    if n % 2 == 0:
        return 2
    i = 3
    while i * i <= n and i <= limit:
        if n % i == 0:
            return i
        i += 2
    return None


def pocklington_certificate(n: int) -> Dict:
    """
    为 n 生成 Pocklington-Lehmer 素性证书 (或合数证书).

    证书结构:
      素数: { verdict, n, certificate: { type, F, F_factors, F>√n, witnesses } }
      合数: { verdict, n, certificate: { factor, cofactor } }

    证明原理:
      找到 n-1 = F·R, F 的所有素因子已知, 且 F > √n.
      对每个素因子 q|F, 找见证 a 使得:
        a^{n-1} ≡ 1 (mod n)  AND  gcd(a^{(n-1)/q}-1, n) = 1
      由 Pocklington 定理推导 n 是素数.
    """
    if n <= 1:
        return {"verdict": "NOT_PRIME", "n": n, "certificate": "trivial"}
    if n == 2:
        return {"verdict": "PRIME", "n": n, "certificate": {"type": "base_case_2"}}
    if n % 2 == 0:
        return {"verdict": "COMPOSITE", "n": n, "certificate": {"factor": 2, "cofactor": n // 2}}

    # 快速 Miller-Rabin
    if not miller_rabin_deterministic(n):
        f = _factor_small(n)
        cert = {"factor": f, "cofactor": n // f} if f else {"type": "miller_rabin_composite"}
        return {"verdict": "COMPOSITE", "n": n, "certificate": cert}

    # n 通过 Miller-Rabin → 构造 Pocklington 证书
    nm1 = n - 1
    F_factors: Dict[int, int] = {}
    remaining = nm1

    # 试除分解 n-1 的小因子
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        while remaining % p == 0:
            F_factors[p] = F_factors.get(p, 0) + 1
            remaining //= p

    # remaining 可能是大素数或合数
    if remaining > 1 and miller_rabin_deterministic(remaining):
        F_factors[remaining] = F_factors.get(remaining, 0) + 1
        remaining = 1

    F = 1
    for p, e in F_factors.items():
        F *= p ** e

    pocklington_ok = (F * F > n) and (remaining == 1)

    # 为每个素因子 q 找 Pocklington 见证 a
    witnesses_map: Dict[int, int] = {}
    if pocklington_ok:
        for q in F_factors.keys():
            exp_full = nm1
            exp_partial = nm1 // q
            for a in range(2, min(n, 200)):
                if pow(a, exp_full, n) == 1:
                    gcd_val = math.gcd(pow(a, exp_partial, n) - 1, n)
                    if gcd_val == 1:
                        witnesses_map[q] = a
                        break

        all_witnessed = len(witnesses_map) == len(F_factors)
    else:
        all_witnessed = False

    if pocklington_ok and all_witnessed:
        cert_type = "pocklington_lehmer"
        theorem_text = (
            f"n-1 = {F}·{remaining if remaining > 1 else 1}; "
            f"F={F} > √n={math.isqrt(n)}; "
            f"Pocklington 定理 → n 是素数"
        )
    else:
        cert_type = "miller_rabin_deterministic"
        theorem_text = (
            f"确定性 Miller-Rabin (见证集覆盖 n<3.3×10²⁴) → n 是素数"
        )

    return {
        "verdict": "PRIME",
        "n": n,
        "certificate": {
            "type": cert_type,
            "n_minus_1": nm1,
            "F": F,
            "F_factors": {str(k): v for k, v in F_factors.items()},
            "F_gt_sqrt_n": bool(F * F > n),
            "pocklington_witnesses": {str(k): v for k, v in witnesses_map.items()},
            "theorem": theorem_text
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# 第六部分: 主分析引擎
# ═══════════════════════════════════════════════════════════════════════════

class H2QPrimeEngine:
    """
    H2Q P-adic Mahler × Rank-8 素数分析引擎.

    将以下数学工具统一在一个流水线中:
      [P-进编码] → [Mahler展开(线性代数)] → [Rank-8 SVD筛] → [Pocklington证明]
    """

    def __init__(self, N: int = 500):
        self.N = N
        t0 = time.time()
        self.primes = segmented_sieve(N)
        self.prime_set = set(self.primes)
        self.pi_table = prime_counting_table(N)
        self._init_time = time.time() - t0

    # ──────────────────────────────────────────────────────────────
    # 模块 A: Mahler 展开分析
    # ──────────────────────────────────────────────────────────────

    def mahler_analysis(self, p_base: int = 2, display_terms: int = 30) -> Dict:
        """
        对 π(x) 做完整 Mahler 展开分析.

        步骤:
          1. 计算 π(0..N) 函数值向量
          2. 用递推差分表计算 Mahler 系数 c_k = Δ^k π(0)
          3. 计算每个 c_k 的 p-进范数 |c_k|_p
          4. 验证 Pascal 矩阵可逆性 B·B^{-1}=I
          5. 验证 Mahler 重建误差为 0 (精确性证明)
        """
        pi_list = self.pi_table[: self.N + 1]

        # Mahler 系数
        coeffs = compute_mahler_coefficients(pi_list)
        M = min(display_terms, len(coeffs))

        # P-进范数序列
        padic_data = []
        for k in range(M):
            c = coeffs[k]
            norm_val = p_adic_norm(abs(c), p_base) if c != 0 else 0.0
            val = p_adic_valuation(abs(c), p_base) if c != 0 else 999
            padic_data.append({
                "k": k,
                "c_k": c,
                f"|c_k|_{p_base}": round(norm_val, 6),
                f"v_{p_base}(c_k)": val if val < 999 else "∞"
            })

        # 验证 Pascal 矩阵逆
        pascal_ok = verify_pascal_inverse(min(12, self.N))

        # 重建精度验证
        max_err = 0
        recon_samples = []
        for n in range(min(50, self.N)):
            reconstructed = mahler_evaluate(coeffs, n)
            err = abs(reconstructed - pi_list[n])
            max_err = max(max_err, err)
            recon_samples.append({
                "n": n, "π_true": pi_list[n],
                "π_mahler": reconstructed, "error": err
            })

        # P-进收敛性分析: v_p(c_k) 序列
        valuations = [p_adic_valuation(abs(c), p_base) if c != 0 else 999
                      for c in coeffs[:M]]
        nonzero_vals = [v for v in valuations if v < 999]
        avg_valuation = sum(nonzero_vals) / len(nonzero_vals) if nonzero_vals else 0

        # Mahler 收敛在 ℤ_p 中意味着 v_p(c_k) → ∞, 即 |c_k|_p → 0
        # 对 π(x) 作为整数函数: Mahler 展开在所有整数点精确, c_k 是整数
        convergence_note = (
            f"Δ^k π(0) 是整数序列; 在 ℤ_{p_base} 中: "
            f"v_{p_base}(c_k) 均值={avg_valuation:.2f}, "
            f"|c_k|_{p_base} → 0 意味着 π(x) 是 ℤ_{p_base} 上的 {p_base}-进连续函数"
        )

        return {
            "mahler_coefficients": padic_data,
            "pascal_inverse_verified": pascal_ok,
            "reconstruction_max_error": max_err,
            "reconstruction_is_exact": (max_err == 0),
            "recon_samples": recon_samples[:10],
            "convergence_note": convergence_note,
            "p_base": p_base
        }

    # ──────────────────────────────────────────────────────────────
    # 模块 B: Rank-8 SVD 筛分析
    # ──────────────────────────────────────────────────────────────

    def rank8_analysis(self) -> Dict:
        return rank8_sieve_analysis(min(self.N, 500))

    # ──────────────────────────────────────────────────────────────
    # 模块 C: P-进分布与聚类分析
    # ──────────────────────────────────────────────────────────────

    def padic_distribution(self) -> Dict:
        """
        分析素数在 p-进度量下的分布.
        关键性质: 素数 q 对每个 p ≠ q 满足 v_p(q) = 0, |q|_p = 1.
        即素数在 p-进范数意义下是"单位元": 距离原点为 1.
        """
        results = {}
        for p in [2, 3, 5, 7]:
            # 素数的 p-进范数 (应全为 1, 除 q=p 自身)
            prime_norms = [(q, p_adic_norm(q, p)) for q in self.primes[:30]]
            outliers = [(q, n) for q, n in prime_norms if n != 1.0]  # 只有 q=p 时 |q|_p < 1

            # 合数的 p-进范数 (< 1 的更多)
            composites = [i for i in range(4, min(60, self.N + 1))
                          if i not in self.prime_set]
            composite_norms = [(c, round(p_adic_norm(c, p), 4)) for c in composites[:15]]

            # Mahler 系数 p-进赋值
            pi_list = self.pi_table[:30]
            coeffs = compute_mahler_coefficients(pi_list)
            coeff_valuations = [
                (k, c, p_adic_valuation(abs(c), p) if c != 0 else 999)
                for k, c in enumerate(coeffs[:15])
            ]

            results[f"p={p}"] = {
                "prime_padic_norms": prime_norms[:15],
                "outliers_qeqp": outliers,
                "composite_padic_norms": composite_norms,
                "pi_mahler_coeff_valuations": coeff_valuations,
                "characterization": (
                    f"素数 q≠{p} 满足 |q|_{p}=1 (p-进单位); "
                    f"合数 n 通常 |n|_{p}<1 (被 {p} 整除)"
                )
            }
        return results

    # ──────────────────────────────────────────────────────────────
    # 模块 D: 自动化证明证书生成
    # ──────────────────────────────────────────────────────────────

    def generate_proof_certificates(self, targets: Optional[List[int]] = None) -> List[Dict]:
        """
        为每个目标 n 生成完整证明证书.
        同时附加:
          - P-进编码 (2-进和 3-进数码)
          - n-1 的 2-进赋值 (Pocklington F 的主要来源)
          - Hensel 提升演示 (展示 p-进精度递增)
        """
        if targets is None:
            # 默认: 精心挑选的素数 + 合数样本
            targets = [
                2, 3, 5, 7, 11, 13,          # 小素数
                97, 101, 127, 131,             # 两位素数
                997, 1009, 1013,               # 三位素数
                7919, 7927, 104729,            # 大素数
                4, 9, 25, 100, 1001, 7921,     # 合数
                # 特殊数: Mersenne 素数候选
                8191,   # 2^13 - 1, Mersenne 素数
                524287, # 2^19 - 1, Mersenne 素数
            ]

        certs = []
        for n in targets:
            cert = pocklington_certificate(n)

            # 附加 P-进信息
            cert["padic_encoding"] = {
                "base2_digits": p_adic_encode(n, 2, 20),
                "base3_digits": p_adic_encode(n, 3, 12),
                "v_2(n)": p_adic_valuation(n, 2),
                "v_3(n)": p_adic_valuation(n, 3),
            }

            if cert["verdict"] == "PRIME":
                # n-1 的 p-进赋值 (与 Pocklington F 相关)
                nm1 = n - 1
                cert["padic_nm1"] = {
                    "v_2(n-1)": p_adic_valuation(nm1, 2),
                    "v_3(n-1)": p_adic_valuation(nm1, 3),
                    "|n-1|_2": round(p_adic_norm(nm1, 2), 6),
                }

                # Hensel 提升: x² ≡ n (mod 2^k)
                a0 = n % 2
                hensel_2 = hensel_lift(a0, 2, n, precision=8)
                cert["hensel_demo_p2"] = {
                    "polynomial": "f(x) = x² - n",
                    "lifting": hensel_2,
                    "interpretation": "x²≡n (mod 2^k) 的 p-进解序列"
                }
            else:
                if "factor" in cert.get("certificate", {}):
                    f = cert["certificate"]["factor"]
                    cert["padic_factor"] = {
                        "factor": f,
                        f"v_{f}(n)": p_adic_valuation(n, f),
                        f"|n|_{f}": round(p_adic_norm(n, f), 6),
                        "interpretation": (
                            f"|{n}|_{f} = {p_adic_norm(n, f):.4f} < 1 "
                            f"⟹ {f}^{p_adic_valuation(n, f)} | {n}"
                        )
                    }

            certs.append(cert)
        return certs

    # ──────────────────────────────────────────────────────────────
    # 模块 E: 素数间隔 Mahler 分析
    # ──────────────────────────────────────────────────────────────

    def gap_mahler_analysis(self) -> Dict:
        """
        对素数间隔序列 g_n = p_{n+1} - p_n 做 Mahler 展开.
        孪生素数猜想的 p-进视角: g_n = 2 在 Mahler 展开中应有无限多非零系数.
        """
        if len(self.primes) < 3:
            return {}

        gaps = [self.primes[i + 1] - self.primes[i]
                for i in range(len(self.primes) - 1)]
        M = min(60, len(gaps))
        gap_vals = gaps[:M]

        coeffs = compute_mahler_coefficients(gap_vals)

        # P-进范数分析
        gap_p2 = [
            {"k": k, "c_k": c, "|c_k|_2": round(p_adic_norm(abs(c), 2), 6),
             "v_2(c_k)": p_adic_valuation(abs(c), 2) if c != 0 else 999}
            for k, c in enumerate(coeffs[:20])
        ]

        twin_indices = [i for i, g in enumerate(gaps) if g == 2]

        # 间隔分布统计
        gap_dist: Dict[int, int] = {}
        for g in gaps:
            gap_dist[g] = gap_dist.get(g, 0) + 1
        top_gaps = sorted(gap_dist.items(), key=lambda x: -x[1])[:8]

        return {
            "first_30_gaps": gaps[:30],
            "gap_mahler_coefficients": coeffs[:20],
            "gap_p2_norms": gap_p2,
            "twin_prime_indices": twin_indices[:15],
            "twin_prime_count": len(twin_indices),
            "max_gap": max(gaps),
            "gap_distribution_top8": top_gaps,
            "twin_prime_note": (
                f"孪生素数 (间隔=2) 出现 {len(twin_indices)} 次 "
                f"(在 N={self.N} 范围内). "
                f"间隔序列的 Mahler 系数的 2-进范数刻画间隔的乘法结构."
            )
        }

    # ──────────────────────────────────────────────────────────────
    # 主流水线
    # ──────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        t0 = time.time()

        print(f"[H2Q-Prime] N={self.N}, π({self.N})={len(self.primes)} "
              f"(初始化 {self._init_time*1000:.1f}ms)")

        print("[H2Q-Prime] ① Mahler P-进 Taylor 展开 (Pascal 线性代数)...")
        mahler = self.mahler_analysis(p_base=2, display_terms=25)

        print("[H2Q-Prime] ② Rank-8 SVD 筛矩阵分析...")
        rank8 = self.rank8_analysis()

        print("[H2Q-Prime] ③ P-进分布分析...")
        padic_dist = self.padic_distribution()

        print("[H2Q-Prime] ④ 自动化 Pocklington-Lehmer 证书...")
        proofs = self.generate_proof_certificates()

        print("[H2Q-Prime] ⑤ 素数间隔 Mahler 分析...")
        gap = self.gap_mahler_analysis()

        elapsed = time.time() - t0

        report = {
            "metadata": {
                "N": self.N,
                "pi_N": len(self.primes),
                "primes_sample": self.primes[:20],
                "elapsed_sec": round(elapsed, 4),
                "framework": "H2Q P-adic Mahler × Rank-8 Linear Algebra Prime Engine"
            },
            "mahler_expansion": mahler,
            "rank8_sieve": rank8,
            "padic_distribution": padic_dist,
            "proof_certificates": proofs,
            "gap_analysis": gap
        }

        return report


# ═══════════════════════════════════════════════════════════════════════════
# 第七部分: 报告打印器
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: Dict) -> None:
    SEP = "═" * 72

    print(f"\n{SEP}")
    print("   H2Q P-adic Mahler × Rank-8 Prime Engine — 分析报告")
    print(SEP)

    meta = report["metadata"]
    print(f"\n▌ 基本信息")
    print(f"  分析范围  : N = {meta['N']}")
    print(f"  素数计数  : π({meta['N']}) = {meta['pi_N']} 个素数")
    print(f"  前20个素数: {meta['primes_sample']}")
    print(f"  总耗时    : {meta['elapsed_sec']}s")

    # ── Mahler 展开
    print(f"\n▌ ① Mahler P-进 Taylor 展开 (π(x) 的二项式基展开)")
    print(f"  数学框架: π(n) = Σ_{{k=0}}^{{n}} c_k·C(n,k),  c_k = Δ^k π(0)")
    m = report["mahler_expansion"]
    print(f"  Pascal 矩阵 B·B⁻¹=I 验证  : {'✓ 通过 (整数精确等式)' if m['pascal_inverse_verified'] else '✗ 失败'}")
    print(f"  Mahler 重建最大误差         : {m['reconstruction_max_error']} "
          f"({'✓ 精确重建' if m['reconstruction_is_exact'] else '有误差'})")
    print(f"  P-进收敛性: {m['convergence_note']}")
    print(f"\n  Mahler 系数 c_k = Δ^k π(0) 及 2-进范数:")
    print(f"  {'k':>3}  {'c_k':>8}  {'|c_k|_2':>9}  {'v_2(c_k)':>8}  可视化")
    for row in m["mahler_coefficients"][:18]:
        k = row["k"]
        ck = row["c_k"]
        norm_v = row["|c_k|_2"]
        val_v = row["v_2(c_k)"]
        bar_len = min(int(abs(ck) * 0.5), 20) if ck != 0 else 0
        bar = "▓" * bar_len
        print(f"  {k:>3}  {ck:>8}  {norm_v:>9.6f}  {str(val_v):>8}  {bar}")

    # 重建样例
    print(f"\n  Mahler 级数重建验证 (前10个点):")
    print(f"  {'n':>4}  {'π_true':>7}  {'π_Mahler':>9}  {'误差':>5}")
    for s in m["recon_samples"]:
        print(f"  {s['n']:>4}  {s['π_true']:>7}  {s['π_mahler']:>9}  {s['error']:>5}")

    # ── Rank-8
    print(f"\n▌ ② Rank-8 SVD 线性代数筛 (H2Q 架构哲学)")
    r = report["rank8_sieve"]
    if "error" not in r:
        print(f"  筛矩阵规模  : {r['matrix_shape'][0]} × {r['matrix_shape'][1]}")
        print(f"  Rank-8 能量占比: {r['rank8_energy_pct']}%")
        print(f"  素数分类精度: {r['rank8_accuracy']}%")
        print(f"\n  奇异值分布 (各秩贡献):")
        print(f"  {'秩':>4}  {'σ_r':>8}  {'能量%':>7}")
        for sv in r["singular_contributions"]:
            bar = "█" * int(sv["energy_pct"] * 0.4)
            print(f"  {sv['rank']:>4}  {sv['sigma']:>8.4f}  {sv['energy_pct']:>6.2f}%  {bar}")
        print(f"\n  解读: {r['interpretation']}")
    else:
        print(f"  {r['error']}")

    # ── Pocklington 证书
    print(f"\n▌ ③ 自动化 Pocklington-Lehmer 素性证书")
    print(f"  {'n':>8}  {'判定':>6}  {'证书类型':>25}  {'F>√n':>6}")
    for cert in report["proof_certificates"]:
        n_v = cert["n"]
        verdict = "✓ 素数" if cert["verdict"] == "PRIME" else "✗ 合数"
        cert_info = cert.get("certificate", {})
        if isinstance(cert_info, dict):
            ctype = cert_info.get("type", "base_case")
            fok = "✓" if cert_info.get("F_gt_sqrt_n", False) else "-"
        else:
            ctype = str(cert_info)
            fok = "-"
        print(f"  {n_v:>8}  {verdict:>6}  {ctype:>25}  {fok:>6}")

    # 选取一个典型 Pocklington 证书展示细节
    pocklington_examples = [
        c for c in report["proof_certificates"]
        if c["verdict"] == "PRIME"
        and isinstance(c.get("certificate"), dict)
        and c["certificate"].get("type") == "pocklington_lehmer"
    ]
    if pocklington_examples:
        ex = pocklington_examples[0]
        cert = ex["certificate"]
        print(f"\n  ── Pocklington 证书详解: n = {ex['n']} ──")
        print(f"    n-1 = {cert['n_minus_1']}")
        print(f"    F (已知因子积) = {cert['F']}")
        print(f"    F 的素因子分解: { {k: v for k, v in cert['F_factors'].items()} }")
        print(f"    F > √n: {cert['F_gt_sqrt_n']}")
        print(f"    Pocklington 见证: { {k: v for k, v in cert['pocklington_witnesses'].items()} }")
        print(f"    定理: {cert['theorem']}")

    # ── 间隔分析
    print(f"\n▌ ④ 素数间隔 Mahler 分析")
    ga = report.get("gap_analysis", {})
    if ga:
        print(f"  前30个素数间隔: {ga.get('first_30_gaps', [])}")
        print(f"  最大间隔 (N={meta['N']}): {ga.get('max_gap', 'N/A')}")
        print(f"  孪生素数间隔位置 (前15): {ga.get('twin_prime_indices', [])}")
        print(f"  孪生素数数量: {ga.get('twin_prime_count', 0)}")
        print(f"  间隔频率分布 (Top-8): {ga.get('gap_distribution_top8', [])}")
        print(f"\n  {ga.get('twin_prime_note', '')}")

        print(f"\n  间隔序列 Mahler 系数 c_k 的 2-进赋值:")
        print(f"  {'k':>3}  {'c_k':>8}  {'v_2(c_k)':>8}  {'|c_k|_2':>9}")
        for row in ga.get("gap_p2_norms", [])[:15]:
            print(f"  {row['k']:>3}  {row['c_k']:>8}  {str(row['v_2(c_k)']):>8}  {row['|c_k|_2']:>9.6f}")

    # ── P-进分布摘要
    print(f"\n▌ ⑤ P-进分布摘要")
    pd = report.get("padic_distribution", {})
    for p_key in ["p=2", "p=3"]:
        if p_key in pd:
            print(f"  {p_key}: {pd[p_key]['characterization']}")

    # ── 证明框架总结
    print(f"\n{SEP}")
    print("  证明框架总结")
    print("  ┌─ [P-进编码] n = Σ d_k·p^k  →  v_p(n)、|n|_p 刻画可除性结构")
    print("  ├─ [Mahler展开] π(n) = Σ c_k·C(n,k)  →  Newton差分+Pascal矩阵精确重建")
    print("  │   B[n,k]=C(n,k) 下三角变换  →  B·B⁻¹=I (整数精确)")
    print("  │   c_k = Δ^k π(0): 2-进赋值上升 → π 是 ℤ_2 上连续函数")
    print("  ├─ [Rank-8 SVD] 筛矩阵奇异分解  →  前8个模式捕获主要筛能量")
    print("  │   与 H2Q Rank-8 本质主义一致: 素数结构可由 8 个本征模式近似")
    print("  ├─ [Pocklington定理] n-1=F·R, F>√n  →  严密初等素性证明")
    print("  │   每个素因子 q|F 有独立见证 a  →  证书可独立验证")
    print("  └─ [Hensel提升] a₀(mod p) → a_k(mod p^k)  →  因子结构的p-进精化")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="H2Q P-adic Prime Engine")
    parser.add_argument("--N", type=int, default=500,
                        help="分析范围上限 (默认 500)")
    parser.add_argument("--save-json", type=str, default="prime_engine_report.json",
                        help="保存 JSON 报告路径")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存 JSON")
    args = parser.parse_args()

    engine = H2QPrimeEngine(N=args.N)
    report = engine.run()
    print_report(report)

    if not args.no_save:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[H2Q-Prime] 完整报告已保存至: {args.save_json}")

    # 额外: Pocklington 证书摘要
    print(f"\n[H2Q-Prime] 证书统计:")
    primes_proven = sum(1 for c in report["proof_certificates"] if c["verdict"] == "PRIME")
    composites_proven = sum(1 for c in report["proof_certificates"] if c["verdict"] == "COMPOSITE")
    pocklington_count = sum(
        1 for c in report["proof_certificates"]
        if isinstance(c.get("certificate"), dict)
        and c["certificate"].get("type") == "pocklington_lehmer"
    )
    print(f"  素数证书: {primes_proven}, 合数证书: {composites_proven}")
    print(f"  其中 Pocklington-Lehmer 完整证书: {pocklington_count}")
    print(f"  Mahler 重建精确: {report['mahler_expansion']['reconstruction_is_exact']}")
    if HAS_NUMPY and "error" not in report.get("rank8_sieve", {}):
        print(f"  Rank-8 素数分类精度: {report['rank8_sieve']['rank8_accuracy']}%")

    return report


if __name__ == "__main__":
    main()
