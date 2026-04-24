"""End-to-end acceptance and benchmarking for binary STA integration."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import torch

from h2q_evolution import H2Q_Evolution_Engine
from sta_core_v2 import StereographicAttentionLayer, Stereographic_Attention_Layer_V2
from test_stereographic_attention_layer import (
    cuda_dummy_forward,
    cuda_stress_2048,
    evolution_check,
    small_correctness_check,
)

WORKDIR = Path(__file__).resolve().parent
JSON_REPORT = WORKDIR / "final_binary_sta_report.json"
MD_REPORT = WORKDIR / "FINAL_BINARY_STA_REPORT.md"


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "agi_joint_trainer_acceptance",
        str(WORKDIR / "agi_joint_trainer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _measure_layer(name: str, layer: torch.nn.Module, x: torch.Tensor, runs: int = 3) -> dict:
    device = x.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    layer.eval()
    timings_ms = []
    with torch.inference_mode():
        _ = layer(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(runs):
            t0 = time.perf_counter()
            out = layer(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "name": name,
        "output_shape": list(out.shape),
        "avg_time_ms": sum(timings_ms) / len(timings_ms),
        "min_time_ms": min(timings_ms),
        "max_time_ms": max(timings_ms),
        "peak_vram_mb": peak_mb,
        "last_sparsity": float(getattr(layer, "last_sparsity", 0.0)),
        "routing_density": float(getattr(layer, "last_routing_density", 0.0)),
    }


def _acceptance_core_checks() -> dict:
    small_correctness_check()
    evolution_check()
    cuda_dummy_forward()
    cuda_stress_2048()
    return {
        "small_correctness": True,
        "evolution_check": True,
        "cuda_dummy_forward": True,
        "cuda_stress_2048": True,
    }


def _h2q_binary_sta_smoke() -> dict:
    device = _device()
    torch.manual_seed(101)
    model = H2Q_Evolution_Engine(
        dim=128,
        num_layers=2,
        rank=8,
        max_seq_len=128,
        attention_type="binary_sta",
        binary_num_planes=64,
        binary_chunk_size=32,
    ).to(device)
    x = torch.randint(0, model.VOCAB, (2, 64), device=device)
    y = torch.randint(0, model.VOCAB, (2, 64), device=device)

    logits, loss = model(x, y)
    if loss is None:
        raise AssertionError("H2Q binary STA smoke loss is None")
    topo = model.get_topology_sparsity()

    with torch.no_grad():
        generated = model.generate(x[:1, :8], new_bytes=8)

    return {
        "device": str(device),
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "topology_sparsity": float(topo),
        "generated_shape": list(generated.shape),
    }


def _trainer_binary_sta_smoke() -> dict:
    trainer = _load_trainer_module()
    device = torch.device("cuda:0")
    torch.manual_seed(202)
    cfg = {
        "dim": 128,
        "factor_size": 32,
        "fixed_rank": 8,
        "depth": 4,
        "seq_len": 32,
        "batch_size": 2,
        "dropout_rate": 0.1,
        "axiom_lambda": 0.1,
        "shockwave_threshold": math.pi / 2,
        "sta_variant": "binary_sta",
        "binary_num_planes": 64,
        "binary_chunk_size": 16,
        "binary_routing_mode": "normalize",
        "binary_backend": "packbits",
        "binary_fused_chunk_compute": True,
        "hash_dim": 32,
        "num_buckets": 4,
        "hamming_thresh": 4,
        "lr": 3e-4,
        "weight_decay": 0.02,
        "grad_clip": 1.0,
        "total_chunks": 2,
        "chunk_size_mb": 1,
        "source_dir": r"E:\Datasets\FineWeb-Edu_Full",
        "buffer_dir": r"D:\H2Q_Cache_Zone",
        "checkpoint_path": "binary_sta_acceptance_ckpt.pt",
        "best_model_path": "binary_sta_acceptance_best.pt",
        "telemetry_csv": "binary_sta_acceptance_telemetry.csv",
        "supervise_every": 0,
        "supervise_gen_tokens": 64,
        "eval_window_multiplier": 10,
    }
    model = trainer.AGI_Accelerated_Transformer(cfg).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    y = torch.randint(0, 256, (2, 32), device=device)
    logits, loss = model(x, y)
    if loss is None:
        raise AssertionError("trainer binary STA smoke loss is None")
    loss.backward()
    stats = model.accel_stats()
    with torch.no_grad():
        generated = model.generate(x[:1, :4], 6)

    return {
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "stats": {k: float(v) for k, v in stats.items()},
        "generated_shape": list(generated.shape),
    }


def _benchmarks() -> dict:
    device = _device()
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA unavailable"}

    torch.manual_seed(303)
    x1024 = torch.randn(4, 1024, 768, device=device)
    x2048 = torch.randn(1, 2048, 768, device=device)

    sta_v2_1024 = Stereographic_Attention_Layer_V2(
        hidden_dim=768,
        rank=8,
        max_seq_len=1024,
        causal=True,
    ).to(device)
    binary_1024 = StereographicAttentionLayer(
        hidden_dim=768,
        num_planes=128,
        chunk_size=64,
        causal=True,
        routing_mode="normalize",
    ).to(device)

    sta_v2_2048 = Stereographic_Attention_Layer_V2(
        hidden_dim=768,
        rank=8,
        max_seq_len=2048,
        causal=True,
    ).to(device)
    binary_2048 = StereographicAttentionLayer(
        hidden_dim=768,
        num_planes=128,
        chunk_size=64,
        causal=True,
        routing_mode="normalize",
    ).to(device)

    bench_1024 = {
        "sta_v2": _measure_layer("sta_v2_1024", sta_v2_1024, x1024),
        "binary_sta": _measure_layer("binary_sta_1024", binary_1024, x1024),
    }
    bench_2048 = {
        "sta_v2": _measure_layer("sta_v2_2048", sta_v2_2048, x2048, runs=2),
        "binary_sta": _measure_layer("binary_sta_2048", binary_2048, x2048, runs=2),
    }

    return {
        "status": "ok",
        "seq1024": bench_1024,
        "seq2048": bench_2048,
    }


def _render_report(result: dict) -> None:
    JSON_REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Final Binary STA Report",
        "",
        "## Acceptance",
        f"- core_checks: {result['acceptance']['core_checks']}",
        f"- h2q_binary_sta: loss={result['acceptance']['h2q_smoke']['loss']:.6f}, topology_sparsity={result['acceptance']['h2q_smoke']['topology_sparsity']:.6f}",
        f"- trainer_binary_sta: loss={result['acceptance']['trainer_smoke']['loss']:.6f}, sta_sparsity_mean={result['acceptance']['trainer_smoke']['stats']['sta_sparsity_mean']:.6f}",
        "",
        "## Benchmark 1024",
    ]

    if result["benchmark"]["status"] == "ok":
        for name, stats in result["benchmark"]["seq1024"].items():
            lines.append(
                f"- {name}: avg_time_ms={stats['avg_time_ms']:.3f}, peak_vram_mb={stats['peak_vram_mb']:.2f}, last_sparsity={stats['last_sparsity']:.6f}"
            )
        lines.append("")
        lines.append("## Benchmark 2048")
        for name, stats in result["benchmark"]["seq2048"].items():
            lines.append(
                f"- {name}: avg_time_ms={stats['avg_time_ms']:.3f}, peak_vram_mb={stats['peak_vram_mb']:.2f}, last_sparsity={stats['last_sparsity']:.6f}"
            )
    else:
        lines.append(f"- skipped: {result['benchmark']['reason']}")

    lines.extend(
        [
            "",
            "## Analysis",
            result["analysis"],
            "",
        ]
    )
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    core_checks = _acceptance_core_checks()
    h2q_smoke = _h2q_binary_sta_smoke()
    trainer_smoke = _trainer_binary_sta_smoke()
    benchmark = _benchmarks()

    analysis = (
        "binary_sta completed three-level acceptance: core correctness/evolution/GPU long sequence passed; "
        "h2q_evolution and agi_joint_trainer are both configurable and passed smoke tests; "
        "benchmark provides direct comparison with sta_v2 for further default path decisions."
    )

    result = {
        "acceptance": {
            "core_checks": core_checks,
            "h2q_smoke": h2q_smoke,
            "trainer_smoke": trainer_smoke,
        },
        "benchmark": benchmark,
        "analysis": analysis,
    }
    _render_report(result)
    print(f"[report] json={JSON_REPORT.name}")
    print(f"[report] md={MD_REPORT.name}")


if __name__ == "__main__":
    main()
