"""
cuda_ext_protocol.py
====================
一键流程：
  1. 诊断编译环境（CUDA_HOME / nvcc / cl / ninja）
  2. 尝试 JIT 编译 binary_sta_fused_ext
  3. 若编译成功，运行 3×3 协议（cuda_ext vs packbits vs sta_v2）
  4. 给出"是否默认切换到 cuda_ext"的量化门槛结论
  5. 写出 cuda_ext_protocol_result.json 和 CUDA_EXT_DECISION.md

用法:
    python cuda_ext_protocol.py            # 完整流程
    python cuda_ext_protocol.py --diag     # 只输出环境诊断
    python cuda_ext_protocol.py --force    # 跳过编译检查，强制运行（供已编译好的情况）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch

WORKDIR = Path(__file__).resolve().parent
JSON_OUT = WORKDIR / "cuda_ext_protocol_result.json"
MD_OUT   = WORKDIR / "CUDA_EXT_DECISION.md"

# ─── 门槛常量 ─────────────────────────────────────────────────────
# "推荐切换到 cuda_ext 为默认" 的判据（三条全部满足才切换）
THRESHOLD_LOSS_DELTA_MAX   = 0.05   # binary 末步 loss 相比 packbits ≤ +0.05
THRESHOLD_TPS_GAIN_MIN     = 1.10   # cuda_ext 相比 packbits TPS 提升 ≥ 10%
THRESHOLD_P99_REGRESS_MAX  = 0.20   # cuda_ext 相比 packbits p99 步时回退 ≤ +20%
# 额外保险：VRAM 节省不得为负（cuda_ext 不应比 packbits 用更多显存）
THRESHOLD_VRAM_OVERHEAD_MB = 50.0   # 允许 VRAM 额外开销 ≤ 50 MB

# ─── 环境诊断 ─────────────────────────────────────────────────────

def run_diagnosis() -> Dict[str, Any]:
    import fix_cuda_env as envfix

    raw = envfix.check_env()
    return {
        "torch_version": raw.get("torch_version", torch.__version__),
        "torch_cuda_version": raw.get("torch_cuda_version", torch.version.cuda),
        "cuda_available": raw.get("cuda_available", torch.cuda.is_available()),
        "device": raw.get("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "CUDA_HOME": raw.get("cpp_ext_CUDA_HOME"),
        "nvcc": raw.get("nvcc"),
        "nvcc_version": raw.get("nvcc_version"),
        "cl": raw.get("cl"),
        "cl_version": raw.get("cl_version"),
        "ninja": raw.get("ninja"),
        "missing": raw.get("missing", []),
        "can_compile": bool(raw.get("can_compile", False)),
    }


def print_diag(d: Dict[str, Any]) -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print(  "║     Binary STA CUDA 编译环境诊断                ║")
    print(  "╚══════════════════════════════════════════════════╝")
    ok = "✓" if d["cuda_available"] else "✗"
    print(f"  {ok} PyTorch {d['torch_version']}  CUDA {d['torch_cuda_version']}")
    print(f"     Device: {d['device'] or 'N/A'}")
    ok = "✓" if d["CUDA_HOME"] else "✗"
    print(f"  {ok} CUDA_HOME: {d['CUDA_HOME'] or '(未设置)'}")
    ok = "✓" if d["nvcc"] else "✗"
    print(f"  {ok} nvcc: {d['nvcc'] or '(未找到)'}")
    if d["nvcc_version"]:
        print(f"       {d['nvcc_version']}")
    ok = "✓" if d["cl"] else "✗"
    print(f"  {ok} cl.exe: {d['cl'] or '(未找到)'}")
    if d["cl_version"]:
        print(f"       {d['cl_version']}")
    ok = "✓" if d["ninja"] else "✗"
    print(f"  {ok} ninja: {d['ninja'] or '(未找到)'}")
    print()
    if d["can_compile"]:
        print("  ✓ 环境完整，可以编译 cuda_ext")
    else:
        print("  ✗ 以下组件缺失，无法编译:")
        for m in d["missing"]:
            print(f"      • {m}")
        print()
        print("  快速修复:")
        print("      python fix_cuda_env.py")
        print()
    print()


# ─── 编译尝试 ─────────────────────────────────────────────────────

def try_compile(verbose: bool = False) -> bool:
    """尝试编译 binary_sta_fused_ext，返回是否成功。"""
    import binary_sta_cuda_ext as ext_mod
    try:
        ext_mod.load_extension(verbose=verbose)
        return True
    except Exception as e:
        print(f"  编译失败: {e}", file=sys.stderr)
        return False


# ─── 训练工具 ─────────────────────────────────────────────────────

def load_corpus(min_bytes: int = 1 << 20) -> bytes:
    paths = [
        WORKDIR / "data/open_corpus/open_corpus.txt",
        WORKDIR / "corpus_mix_256mb.bin",
    ]
    for p in paths:
        if p.exists():
            data = p.read_bytes()
            if len(data) >= 4096:
                return data
    base = b"CUDA extension benchmark corpus for H2Q-MicroStream. " * 4096
    while len(base) < min_bytes:
        base = base * 2
    return base[:min_bytes]


def sample_batch(data: bytes, batch_size: int, seq_len: int, device: torch.device):
    max_start = len(data) - seq_len - 2
    xs, ys = [], []
    for _ in range(batch_size):
        s = random.randint(0, max_start)
        c = data[s : s + seq_len + 1]
        xs.append(torch.tensor(list(c[:-1]), dtype=torch.long))
        ys.append(torch.tensor(list(c[1:]),  dtype=torch.long))
    return torch.stack(xs, 0).to(device), torch.stack(ys, 0).to(device)


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vs = sorted(vals)
    return float(vs[int(round((len(vs) - 1) * q))])


def run_once(
    *,
    variant: str,
    attention_type: str,
    binary_backend: str,
    seed: int,
    steps: int,
    data: bytes,
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 128,
    dim: int = 128,
    layers: int = 4,
    lr: float = 3e-4,
) -> dict:
    from h2q_evolution import H2Q_Evolution_Engine

    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = H2Q_Evolution_Engine(
        dim=dim,
        num_layers=layers,
        rank=8,
        max_seq_len=seq_len,
        attention_type=attention_type,
        binary_num_planes=128,
        binary_chunk_size=64,
        binary_routing_mode="normalize",
        binary_backend=binary_backend,
        binary_fused_chunk_compute=True,
    ).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses, step_times = [], []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = sample_batch(data, batch_size, seq_len, device)
        s0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append((time.perf_counter() - s0) * 1000.0)
        losses.append(float(loss.item()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    tps = (steps * batch_size * seq_len) / max(elapsed, 1e-6)
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                 if device.type == "cuda" else 0.0)

    attn0 = model.blocks[0].attn
    backend_eff = getattr(attn0, "binary_backend", "n/a")
    ext_enabled = bool(getattr(attn0, "cuda_ext_enabled", False))

    return {
        "variant":     variant,
        "seed":        seed,
        "steps":       steps,
        "avg_loss":    sum(losses) / len(losses),
        "last_loss":   losses[-1],
        "tps":         tps,
        "peak_vram_mb":           peak_vram,
        "step_ms_mean":           sum(step_times) / len(step_times),
        "step_ms_p50":            percentile(step_times, 0.50),
        "step_ms_p90":            percentile(step_times, 0.90),
        "step_ms_p99":            percentile(step_times, 0.99),
        "topology_sparsity":      float(model.get_topology_sparsity()),
        "binary_backend_effective": backend_eff,
        "cuda_ext_enabled":         ext_enabled,
    }


def aggregate(records: list[dict], variant: str, steps: int) -> dict:
    rs = [r for r in records if r["variant"] == variant and r["steps"] == steps]
    if not rs:
        return {}
    def mean(k):
        return float(sum(r[k] for r in rs) / len(rs))
    return {
        "variant":      variant,
        "steps":        steps,
        "runs":         len(rs),
        "avg_loss":     mean("avg_loss"),
        "last_loss":    mean("last_loss"),
        "tps":          mean("tps"),
        "peak_vram_mb": mean("peak_vram_mb"),
        "step_ms_mean": mean("step_ms_mean"),
        "step_ms_p90":  mean("step_ms_p90"),
        "step_ms_p99":  mean("step_ms_p99"),
        "topology_sparsity": mean("topology_sparsity"),
        "backends":     sorted({str(r["binary_backend_effective"]) for r in rs}),
        "cuda_ext_flags": sorted({bool(r["cuda_ext_enabled"]) for r in rs}),
    }


# ─── 门槛判决 ─────────────────────────────────────────────────────

def make_decision(summary: list[dict]) -> dict:
    """
    对每个 steps 时长独立判断，最终取保守交集：
      所有时长均满足门槛 → 建议切换到 cuda_ext 为默认。
    """
    per_budget: list[dict] = []

    for item in summary:
        steps = item["steps"]
        # cuda_ext vs packbits
        cx = item.get("cuda_ext")
        pb = item.get("packbits")
        sv = item.get("sta_v2")
        if not cx or not pb:
            per_budget.append({"steps": steps, "verdict": "SKIP", "reason": "数据不完整"})
            continue

        loss_delta  = cx["last_loss"] - pb["last_loss"]
        tps_ratio   = cx["tps"] / max(pb["tps"], 1e-6)
        p99_ratio   = cx["step_ms_p99"] / max(pb["step_ms_p99"], 1e-6)
        vram_delta  = cx["peak_vram_mb"] - pb["peak_vram_mb"]

        ok_loss   = loss_delta <= THRESHOLD_LOSS_DELTA_MAX
        ok_tps    = tps_ratio  >= THRESHOLD_TPS_GAIN_MIN
        ok_p99    = (p99_ratio - 1.0) <= THRESHOLD_P99_REGRESS_MAX
        ok_vram   = vram_delta <= THRESHOLD_VRAM_OVERHEAD_MB

        verdict = "PASS" if (ok_loss and ok_tps and ok_p99 and ok_vram) else "FAIL"
        per_budget.append({
            "steps":       steps,
            "verdict":     verdict,
            "loss_delta":  round(loss_delta, 6),
            "tps_ratio":   round(tps_ratio, 4),
            "p99_ratio":   round(p99_ratio, 4),
            "vram_delta_mb": round(vram_delta, 2),
            "ok_loss":     ok_loss,
            "ok_tps":      ok_tps,
            "ok_p99":      ok_p99,
            "ok_vram":     ok_vram,
            "details": {
                "cuda_ext_tps":    round(cx["tps"], 1),
                "packbits_tps":    round(pb["tps"], 1),
                "cuda_ext_loss":   round(cx["last_loss"], 6),
                "packbits_loss":   round(pb["last_loss"], 6),
                "cuda_ext_p99_ms": round(cx["step_ms_p99"], 3),
                "packbits_p99_ms": round(pb["step_ms_p99"], 3),
                "cuda_ext_vram":   round(cx["peak_vram_mb"], 2),
                "packbits_vram":   round(pb["peak_vram_mb"], 2),
            },
        })

    all_pass   = all(b["verdict"] == "PASS" for b in per_budget)
    any_skip   = any(b["verdict"] == "SKIP" for b in per_budget)

    if any_skip:
        overall = "INCOMPLETE"
        recommendation = "数据不完整，无法给出结论（可能 cuda_ext 编译失败）"
    elif all_pass:
        overall = "SWITCH"
        recommendation = (
            "✅ 建议将 cuda_ext 设为默认 binary_backend。\n"
            "所有 step 时长均满足:\n"
            f"  • loss delta ≤ {THRESHOLD_LOSS_DELTA_MAX} (损失无显著回退)\n"
            f"  • TPS gain ≥ {THRESHOLD_TPS_GAIN_MIN:.0%} (相比 packbits 明显提速)\n"
            f"  • p99 回退 ≤ {THRESHOLD_P99_REGRESS_MAX:.0%} (尾部延迟可接受)\n"
            f"  • VRAM 额外开销 ≤ {THRESHOLD_VRAM_OVERHEAD_MB} MB\n"
            "实施: sta_core_v2.py 中 binary_backend 默认值改为 'cuda_ext'"
        )
    else:
        fails = [b for b in per_budget if b["verdict"] == "FAIL"]
        fail_reasons = []
        for b in fails:
            r = []
            if not b["ok_loss"]: r.append(f"loss_delta={b['loss_delta']:+.4f}>{THRESHOLD_LOSS_DELTA_MAX}")
            if not b["ok_tps"]:  r.append(f"tps_ratio={b['tps_ratio']:.3f}<{THRESHOLD_TPS_GAIN_MIN}")
            if not b["ok_p99"]:  r.append(f"p99_ratio={b['p99_ratio']:.3f}>{1+THRESHOLD_P99_REGRESS_MAX:.1f}")
            if not b["ok_vram"]: r.append(f"vram_delta={b['vram_delta_mb']:+.1f}>{THRESHOLD_VRAM_OVERHEAD_MB}")
            fail_reasons.append(f"steps={b['steps']}: {', '.join(r)}")
        overall = "KEEP_PACKBITS"
        recommendation = (
            "⚠️  保持 packbits 为默认（cuda_ext 未完全满足门槛）\n"
            "未通过的判据:\n"
            + "\n".join(f"  • {r}" for r in fail_reasons)
            + "\n\n下一步建议:\n"
            "  1. 检查 cuda_ext kernel 是否正确实现（数值精度）\n"
            "  2. 若仅 TPS 不达标，考虑优化 kernel 内存访问\n"
            "  3. 可降低门槛（修改本脚本顶部的 THRESHOLD_* 常量）"
        )

    return {
        "thresholds": {
            "loss_delta_max":    THRESHOLD_LOSS_DELTA_MAX,
            "tps_gain_min":      THRESHOLD_TPS_GAIN_MIN,
            "p99_regress_max":   THRESHOLD_P99_REGRESS_MAX,
            "vram_overhead_max": THRESHOLD_VRAM_OVERHEAD_MB,
        },
        "per_budget":    per_budget,
        "overall":       overall,
        "recommendation": recommendation,
    }


# ─── Markdown 报告 ─────────────────────────────────────────────────

def write_md(diagnosis: dict, summary: list[dict], decision: dict) -> None:
    lines = [
        "# Binary STA cuda_ext 切换决策报告",
        "",
        "## 编译环境",
        f"- PyTorch: {diagnosis['torch_version']}  (CUDA {diagnosis['torch_cuda_version']})",
        f"- Device:  {diagnosis['device']}",
        f"- nvcc:    {diagnosis['nvcc_version'] or diagnosis['nvcc'] or '未找到'}",
        f"- cl.exe:  {diagnosis['cl_version'] or diagnosis['cl'] or '未找到'}",
        f"- ninja:   {diagnosis['ninja'] or '未找到'}",
        f"- CUDA_HOME: {diagnosis['CUDA_HOME'] or '(未设置)'}",
        f"- **can_compile: {diagnosis['can_compile']}**",
        "",
        "## 3×3 协议结果",
        "",
        "| steps | variant | TPS | last_loss | p99_ms | VRAM_MB | backend_eff |",
        "|------:|---------|----:|----------:|-------:|--------:|-------------|",
    ]
    for item in summary:
        for key in ("sta_v2", "packbits", "cuda_ext"):
            v = item.get(key)
            if not v:
                continue
            lines.append(
                f"| {item['steps']} | {key} "
                f"| {v['tps']:.0f} "
                f"| {v['last_loss']:.5f} "
                f"| {v['step_ms_p99']:.2f} "
                f"| {v['peak_vram_mb']:.1f} "
                f"| {','.join(v.get('backends', ['n/a']))} |"
            )
    lines += ["", "## cuda_ext vs packbits 差值", ""]
    lines += [
        "| steps | loss_delta | tps_ratio | p99_ratio | vram_delta_MB | verdict |",
        "|------:|-----------:|----------:|----------:|--------------:|---------|",
    ]
    for b in decision["per_budget"]:
        verdict_icon = "✅" if b["verdict"] == "PASS" else ("⚠️" if b["verdict"] == "SKIP" else "❌")
        lines.append(
            f"| {b['steps']} "
            f"| {b.get('loss_delta', 'N/A')} "
            f"| {b.get('tps_ratio', 'N/A')} "
            f"| {b.get('p99_ratio', 'N/A')} "
            f"| {b.get('vram_delta_mb', 'N/A')} "
            f"| {verdict_icon} {b['verdict']} |"
        )
    lines += [
        "",
        "## 门槛定义",
        f"- loss_delta ≤ {THRESHOLD_LOSS_DELTA_MAX}",
        f"- TPS 提升 ≥ {THRESHOLD_TPS_GAIN_MIN:.0%}（vs packbits）",
        f"- p99 回退 ≤ {THRESHOLD_P99_REGRESS_MAX:.0%}",
        f"- VRAM 额外开销 ≤ {THRESHOLD_VRAM_OVERHEAD_MB} MB",
        "",
        "## 最终结论",
        "",
        f"**整体判决: {decision['overall']}**",
        "",
    ]
    lines += decision["recommendation"].splitlines()
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  报告已写入: {MD_OUT}")


# ─── 主流程 ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag",  action="store_true", help="仅输出环境诊断后退出")
    parser.add_argument("--force", action="store_true", help="跳过编译检查，直接运行协议")
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--steps",    type=int, nargs="+", default=[40, 120, 240])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len",    type=int, default=128)
    parser.add_argument("--dim",        type=int, default=128)
    parser.add_argument("--layers",     type=int, default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--verbose-compile", action="store_true")
    args = parser.parse_args()

    # ── 诊断 ──────────────────────────────────────────────────────
    print("诊断编译环境...")
    diag = run_diagnosis()
    print_diag(diag)

    if args.diag:
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        return

    # ── 编译 ──────────────────────────────────────────────────────
    cuda_ext_ok = False
    if args.force:
        print("  --force 跳过编译检查，假设 cuda_ext 已就绪")
        cuda_ext_ok = True
    elif diag["can_compile"]:
        print("  尝试 JIT 编译 binary_sta_fused_ext...")
        cuda_ext_ok = try_compile(verbose=args.verbose_compile)
        if cuda_ext_ok:
            print("  ✓ 编译成功！")
        else:
            print("  ✗ 编译失败，协议将跳过 cuda_ext 列（仍对比 sta_v2 vs packbits）")
    else:
        print("  ✗ 编译环境不完整，跳过 cuda_ext。请先运行: python fix_cuda_env.py")

    # ── 3×3 协议 ─────────────────────────────────────────────────
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    data   = load_corpus()
    cfg = dict(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        layers=args.layers,
        lr=args.lr,
    )

    VARIANTS: list[tuple[str, str, str]] = [
        ("sta_v2",    "sta_v2",    "packbits"),
        ("packbits",  "binary_sta","packbits"),
    ]
    if cuda_ext_ok:
        VARIANTS.append(("cuda_ext", "binary_sta", "cuda_ext"))

    raw: list[dict] = []
    total = len(args.steps) * len(args.seeds) * len(VARIANTS)
    done  = 0
    for steps in args.steps:
        for seed in args.seeds:
            for (label, attn_type, backend) in VARIANTS:
                done += 1
                print(f"  [{done}/{total}] variant={label}  steps={steps}  seed={seed} ...", end="", flush=True)
                t0 = time.perf_counter()
                rec = run_once(
                    variant=label,
                    attention_type=attn_type,
                    binary_backend=backend,
                    seed=seed,
                    steps=steps,
                    data=data,
                    device=device,
                    **cfg,
                )
                elapsed = time.perf_counter() - t0
                print(f"  tps={rec['tps']:.0f}  loss={rec['last_loss']:.4f}  "
                      f"backend_eff={rec['binary_backend_effective']}  [{elapsed:.1f}s]")
                raw.append(rec)

    # ── 汇总 ──────────────────────────────────────────────────────
    summary: list[dict] = []
    for steps in args.steps:
        item: dict = {"steps": steps}
        for label in ("sta_v2", "packbits", "cuda_ext"):
            ag = aggregate(raw, label, steps)
            if ag:
                item[label] = ag
        summary.append(item)

    # ── 决策 ──────────────────────────────────────────────────────
    decision = make_decision(summary)

    # ── 输出 ──────────────────────────────────────────────────────
    result = {
        "diagnosis":  diag,
        "config":     {"seeds": args.seeds, "steps": args.steps, **cfg},
        "cuda_ext_compiled": cuda_ext_ok,
        "raw_runs":   raw,
        "summary":    summary,
        "decision":   decision,
    }
    JSON_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(diag, summary, decision)

    print("\n" + "═" * 60)
    print(f"  整体判决: {decision['overall']}")
    print()
    for line in decision["recommendation"].splitlines():
        print(f"  {line}")
    print("═" * 60)
    print(f"\n  JSON 结果: {JSON_OUT}")
    print(f"  MD  报告: {MD_OUT}")
    print()


if __name__ == "__main__":
    main()
