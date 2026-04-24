"""Multi-seed, multi-duration protocol for STA variants with bottleneck profiling."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from h2q_evolution import H2Q_Evolution_Engine

WORKDIR = Path(__file__).resolve().parent
JSON_REPORT = WORKDIR / "train_compare_sta_protocol.json"
MD_REPORT = WORKDIR / "TRAIN_COMPARE_STA_PROTOCOL.md"


def load_corpus(path: Path, min_bytes: int = 1 << 20) -> bytes:
    if path.exists():
        data = path.read_bytes()
        if len(data) >= 4096:
            return data
    base = (b"MicroStream throughput profiling corpus. " * 4096)
    while len(base) < min_bytes:
        base = base + base
    return base[:min_bytes]


def sample_batch(data: bytes, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - (seq_len + 1)
    xs = []
    ys = []
    for _ in range(batch_size):
        s = random.randint(0, max_start)
        chunk = data[s : s + seq_len + 1]
        xs.append(torch.tensor(list(chunk[:-1]), dtype=torch.long))
        ys.append(torch.tensor(list(chunk[1:]), dtype=torch.long))
    return torch.stack(xs, 0).to(device), torch.stack(ys, 0).to(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = int(round((len(vs) - 1) * q))
    return float(vs[idx])


def run_once(
    variant_name: str,
    attention_type: str,
    binary_backend: str,
    seed: int,
    steps: int,
    data: bytes,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    dim: int,
    layers: int,
    lr: float,
) -> dict:
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
    losses = []
    step_times = []

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

    tokens = steps * batch_size * seq_len
    tps = tokens / max(elapsed, 1e-6)
    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    attn0 = model.blocks[0].attn
    return {
        "variant": variant_name,
        "seed": seed,
        "steps": steps,
        "avg_loss": sum(losses) / len(losses),
        "last_loss": losses[-1],
        "tokens_per_sec": tps,
        "peak_vram_mb": peak_vram,
        "step_time_ms_mean": sum(step_times) / len(step_times),
        "step_time_ms_p50": percentile(step_times, 0.50),
        "step_time_ms_p90": percentile(step_times, 0.90),
        "step_time_ms_p99": percentile(step_times, 0.99),
        "topology_sparsity": float(model.get_topology_sparsity()),
        "binary_backend_effective": getattr(attn0, "binary_backend", "n/a"),
        "cuda_ext_enabled": bool(getattr(attn0, "cuda_ext_enabled", False)),
    }


def profile_hotspots(
    attention_type: str,
    binary_backend: str,
    data: bytes,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    dim: int,
    layers: int,
    lr: float,
    steps: int = 12,
) -> dict:
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA unavailable"}

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

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            x, y = sample_batch(data, batch_size, seq_len, device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            prof.step()

    top = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
    lines = [ln for ln in top.splitlines() if ln.strip()]
    return {
        "status": "ok",
        "top_cuda_table": lines,
    }


def aggregate(records: list[dict], variant: str, steps: int) -> dict:
    rs = [r for r in records if r["variant"] == variant and r["steps"] == steps]
    if not rs:
        return {}

    def mean(key: str) -> float:
        return float(sum(r[key] for r in rs) / len(rs))

    backend_set = sorted({str(r.get("binary_backend_effective", "n/a")) for r in rs})
    ext_flags = sorted({bool(r.get("cuda_ext_enabled", False)) for r in rs})
    return {
        "variant": variant,
        "steps": steps,
        "runs": len(rs),
        "avg_loss": mean("avg_loss"),
        "last_loss": mean("last_loss"),
        "tokens_per_sec": mean("tokens_per_sec"),
        "peak_vram_mb": mean("peak_vram_mb"),
        "step_time_ms_mean": mean("step_time_ms_mean"),
        "step_time_ms_p90": mean("step_time_ms_p90"),
        "step_time_ms_p99": mean("step_time_ms_p99"),
        "topology_sparsity": mean("topology_sparsity"),
        "binary_backend_effective": backend_set,
        "cuda_ext_enabled": ext_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed STA protocol + bottleneck profiling")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--step-budgets", type=int, nargs="+", default=[40, 120, 240])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--binary-backend", type=str, default="packbits", choices=["packbits", "int8", "cuda_ext"])
    parser.add_argument("--corpus", type=str, default="data/open_corpus/open_corpus.txt")
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    corpus = load_corpus(WORKDIR / args.corpus)

    raw = []
    for steps in args.step_budgets:
        for seed in args.seeds:
            raw.append(
                run_once(
                    variant_name="sta_v2",
                    attention_type="sta_v2",
                    binary_backend="packbits",
                    seed=seed,
                    steps=steps,
                    data=corpus,
                    device=device,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    dim=args.dim,
                    layers=args.layers,
                    lr=args.lr,
                )
            )
            raw.append(
                run_once(
                    variant_name=f"binary_sta_{args.binary_backend}",
                    attention_type="binary_sta",
                    binary_backend=args.binary_backend,
                    seed=seed,
                    steps=steps,
                    data=corpus,
                    device=device,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    dim=args.dim,
                    layers=args.layers,
                    lr=args.lr,
                )
            )

    summary = []
    for steps in args.step_budgets:
        s0 = aggregate(raw, "sta_v2", steps)
        s1 = aggregate(raw, f"binary_sta_{args.binary_backend}", steps)
        if not s0 or not s1:
            continue
        summary.append(
            {
                "steps": steps,
                "sta_v2": s0,
                "binary": s1,
                "delta": {
                    "loss_last_binary_minus_sta": s1["last_loss"] - s0["last_loss"],
                    "tokens_per_sec_binary_over_sta": s1["tokens_per_sec"] / max(s0["tokens_per_sec"], 1e-6),
                    "peak_vram_mb_binary_minus_sta": s1["peak_vram_mb"] - s0["peak_vram_mb"],
                    "step_p99_ms_binary_minus_sta": s1["step_time_ms_p99"] - s0["step_time_ms_p99"],
                },
            }
        )

    hotspot_sta = profile_hotspots(
        attention_type="sta_v2",
        binary_backend="packbits",
        data=corpus,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        layers=args.layers,
        lr=args.lr,
    )
    hotspot_binary = profile_hotspots(
        attention_type="binary_sta",
        binary_backend=args.binary_backend,
        data=corpus,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        layers=args.layers,
        lr=args.lr,
    )

    report = {
        "device": str(device),
        "config": {
            "seeds": args.seeds,
            "step_budgets": args.step_budgets,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dim": args.dim,
            "layers": args.layers,
            "lr": args.lr,
            "binary_backend": args.binary_backend,
            "corpus": args.corpus,
        },
        "raw_runs": raw,
        "summary": summary,
        "hotspot_sta_v2": hotspot_sta,
        "hotspot_binary": hotspot_binary,
    }

    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Multi-seed Multi-duration Training Protocol",
        "",
        f"- device: {report['device']}",
        f"- config: {report['config']}",
        "",
        "## Summary",
    ]
    for item in summary:
        lines.append(f"- steps={item['steps']}")
        lines.append(
            f"  sta_v2: last_loss={item['sta_v2']['last_loss']:.6f}, tps={item['sta_v2']['tokens_per_sec']:.2f}, "
            f"vram={item['sta_v2']['peak_vram_mb']:.2f}, p99={item['sta_v2']['step_time_ms_p99']:.3f}"
        )
        lines.append(
            f"  binary: last_loss={item['binary']['last_loss']:.6f}, tps={item['binary']['tokens_per_sec']:.2f}, "
            f"vram={item['binary']['peak_vram_mb']:.2f}, p99={item['binary']['step_time_ms_p99']:.3f}"
        )
        lines.append(
            f"  delta: loss={item['delta']['loss_last_binary_minus_sta']:+.6f}, "
            f"tps_ratio={item['delta']['tokens_per_sec_binary_over_sta']:.4f}x, "
            f"vram={item['delta']['peak_vram_mb_binary_minus_sta']:+.2f}, "
            f"p99={item['delta']['step_p99_ms_binary_minus_sta']:+.3f}"
        )

    lines.extend(["", "## CUDA Hotspots: STA_V2"])
    if hotspot_sta.get("status") == "ok":
        lines.extend([f"- {ln}" for ln in hotspot_sta["top_cuda_table"][:20]])
    else:
        lines.append(f"- skipped: {hotspot_sta.get('reason', 'unknown')}")

    lines.extend(["", "## CUDA Hotspots: Binary"])
    if hotspot_binary.get("status") == "ok":
        lines.extend([f"- {ln}" for ln in hotspot_binary["top_cuda_table"][:20]])
    else:
        lines.append(f"- skipped: {hotspot_binary.get('reason', 'unknown')}")

    lines.extend([
        "",
        "## Recommendation Rule",
        "- If binary keeps loss within +0.05 and gives >=1.2x throughput in medium and long budgets, switch default to binary.",
        "- If p99 latency regresses >20% while throughput gain <10%, keep sta_v2 as default and continue kernel-level tuning.",
    ])

    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[report] json={JSON_REPORT.name}")
    print(f"[report] md={MD_REPORT.name}")


if __name__ == "__main__":
    main()
