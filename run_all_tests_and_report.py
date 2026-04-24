#!/usr/bin/env python3
"""
Run Phase1(A/B/C) and Phase2(2a/2b) quick experiments, then generate reports.
This script is designed for reproducible, isolated test runs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

WORKDIR = Path(__file__).resolve().parent
TRAINER = WORKDIR / "agi_joint_trainer.py"

REDLINE_LOSS_FACTOR = 1.01
REDLINE_TPS_FACTOR = 0.90
REDLINE_VRAM_MAX = 0.25

# Quick mode to complete end-to-end verification within a single session.
QUICK_TOTAL_CHUNKS = 2
QUICK_CHUNK_SIZE_MB = 1
QUICK_DIM = 128
QUICK_DEPTH = 4
QUICK_HASH_DIM = 32
QUICK_NUM_BUCKETS = 4
QUICK_HAMMING_THRESH = 4


@dataclass
class PlanResult:
    name: str
    telemetry: str
    checkpoint: str
    best_model: str
    metrics: Dict[str, float]


def run_cmd(args: List[str], label: str) -> None:
    print(f"\n[RUN] {label}")
    print(" ".join(args))
    proc = subprocess.run(args, cwd=str(WORKDIR), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def collect_metrics(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing telemetry: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Telemetry is empty: {csv_path}")
    window = df.tail(min(50, len(df)))
    return {
        "train_loss_mean": float(window["train_loss"].mean()),
        "val_loss_mean": float(window["val_loss"].mean()),
        "tokens_per_sec_mean": float(window["tokens_per_sec"].mean()),
        "vram_alloc_gb_mean": float(window["vram_alloc_gb"].mean()),
        "rows": int(len(window)),
    }


def load_baseline() -> Dict[str, float]:
    snap = WORKDIR / "baseline_snapshot.json"
    if snap.exists():
        data = json.loads(snap.read_text(encoding="utf-8"))
        val_base = float(data.get("val_loss_μ", 1.7873))
        tps_base = float(data.get("tokens_per_sec_μ", 18014))
    else:
        # fallback from current main telemetry
        df = pd.read_csv(WORKDIR / "agi_joint_telemetry.csv")
        w = df.tail(min(100, len(df)))
        val_base = float(w["val_loss"].mean())
        tps_base = float(w["tokens_per_sec"].mean())
    return {
        "val_base": val_base,
        "tps_base": tps_base,
        "val_max": val_base * REDLINE_LOSS_FACTOR,
        "tps_min": tps_base * REDLINE_TPS_FACTOR,
        "vram_max": REDLINE_VRAM_MAX,
    }


def pass_fail(metrics: Dict[str, float], red: Dict[str, float]) -> Dict[str, bool]:
    return {
        "loss_pass": metrics["val_loss_mean"] <= red["val_max"],
        "tps_pass": metrics["tokens_per_sec_mean"] >= red["tps_min"],
        "vram_pass": metrics["vram_alloc_gb_mean"] <= red["vram_max"],
    }


def char_stats(text: str) -> Dict[str, float]:
    if not text:
        return {
            "invalid_char_rate": 1.0,
            "repeat_bigram_rate": 1.0,
            "max_run": 0,
            "avg_word_len": 0.0,
            "readability_score": 0.0,
        }

    invalid = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    invalid_rate = invalid / len(text)

    # Bigram repeat ratio
    if len(text) > 2:
        bgs = [text[i : i + 2] for i in range(len(text) - 1)]
        uniq = len(set(bgs))
        repeat_rate = 1.0 - (uniq / len(bgs))
    else:
        repeat_rate = 0.0

    # Max run of identical chars
    run = 1
    max_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    words = [w for w in text.split(" ") if w]
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    score = (
        (1 - min(invalid_rate, 0.05) / 0.05) * 0.30
        + (1 - min(repeat_rate, 0.30) / 0.30) * 0.25
        + (1 - min(max_run, 20) / 20) * 0.25
        + (0.20 if (" " in text) else 0.0)
    )
    score = float(max(0.0, min(1.0, score)))

    return {
        "invalid_char_rate": float(invalid_rate),
        "repeat_bigram_rate": float(repeat_rate),
        "max_run": int(max_run),
        "avg_word_len": float(avg_word_len),
        "readability_score": score,
    }


def generation_eval(best_model_path: Path, prompts: List[str]) -> Dict[str, object]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("agi_joint_trainer_eval", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    device = torch.device("cuda:0")
    ckpt = torch.load(best_model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = mod.AGI_Accelerated_Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    samples = []
    scores = []
    with torch.no_grad():
        for p in prompts:
            ids = list(p.encode("utf-8"))
            ctx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(ctx, 160, temperature=0.8)
            text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
            st = char_stats(text)
            samples.append({"prompt": p, "sample": text[:220], **st})
            scores.append(st["readability_score"])

    return {
        "mean_readability_score": float(np.mean(scores)) if scores else 0.0,
        "samples": samples,
    }


def run_plan(name: str, overrides: Dict[str, str]) -> PlanResult:
    telemetry = f"telemetry_{name}.csv"
    ckpt = f"ckpt_{name}.pt"
    best = f"best_{name}.pt"

    args = [
        sys.executable,
        str(TRAINER),
        "--dim", str(QUICK_DIM),
        "--depth", str(QUICK_DEPTH),
        "--hash-dim", str(QUICK_HASH_DIM),
        "--num-buckets", str(QUICK_NUM_BUCKETS),
        "--hamming-thresh", str(QUICK_HAMMING_THRESH),
        "--total-chunks", str(QUICK_TOTAL_CHUNKS),
        "--chunk-size-mb", str(QUICK_CHUNK_SIZE_MB),
        "--checkpoint-path", ckpt,
        "--best-model-path", best,
        "--telemetry-csv", telemetry,
    ]
    for k, v in overrides.items():
        args.extend([k, str(v)])

    run_cmd(args, f"Phase run {name}")
    metrics = collect_metrics(WORKDIR / telemetry)
    return PlanResult(name=name, telemetry=telemetry, checkpoint=ckpt, best_model=best, metrics=metrics)


def main() -> None:
    red = load_baseline()

    phase1_runs = [
        ("A_baseline", {"--supervise-every": 10, "--eval-window-multiplier": 1000}),
        ("B_eval_down", {"--supervise-every": 10, "--eval-window-multiplier": 100}),
        ("C_no_supervise", {"--supervise-every": 0, "--eval-window-multiplier": 1000}),
    ]

    phase1_results: List[PlanResult] = []
    for n, o in phase1_runs:
        phase1_results.append(run_plan(n, o))

    phase1_summary = {}
    best_phase1 = None
    best_tps = -1.0
    for r in phase1_results:
        pf = pass_fail(r.metrics, red)
        row = {
            **r.metrics,
            **pf,
            "overall_pass": bool(all(pf.values())),
            "tps_gain_pct_vs_baseline": float((r.metrics["tokens_per_sec_mean"] / red["tps_base"] - 1) * 100),
            "val_loss_delta_vs_baseline": float(r.metrics["val_loss_mean"] - red["val_base"]),
            "telemetry": r.telemetry,
            "checkpoint": r.checkpoint,
            "best_model": r.best_model,
        }
        phase1_summary[r.name] = row
        if row["overall_pass"] and r.metrics["tokens_per_sec_mean"] > best_tps:
            best_tps = r.metrics["tokens_per_sec_mean"]
            best_phase1 = r

    # If no pass-all plan, fallback to highest tps under loss redline.
    if best_phase1 is None:
        candidates = [
            r for r in phase1_results
            if r.metrics["val_loss_mean"] <= red["val_max"] and r.metrics["vram_alloc_gb_mean"] <= red["vram_max"]
        ]
        best_phase1 = max(candidates, key=lambda x: x.metrics["tokens_per_sec_mean"]) if candidates else phase1_results[0]

    # Phase 2a/2b built on best Phase1 setup
    best_name = best_phase1.name
    best_overrides = dict(phase1_runs[[n for n, _ in phase1_runs].index(best_name)][1])

    phase2a = run_plan(
        "2a_char_baseline",
        {
            **best_overrides,
            "--seq-len": 128,
            "--batch-size": 24,
            "--lr": 3e-4,
        },
    )

    phase2b = run_plan(
        "2b_seq192",
        {
            **best_overrides,
            "--seq-len": 192,
            "--batch-size": 16,
            "--lr": 1.5e-4,
        },
    )

    prompts = ["The ", "Why ", "In this system, ", "Character-level model "]
    gen_2a = generation_eval(WORKDIR / phase2a.best_model, prompts)
    gen_2b = generation_eval(WORKDIR / phase2b.best_model, prompts)

    s2a = gen_2a["mean_readability_score"]
    s2b = gen_2b["mean_readability_score"]

    phase2_summary = {
        "2a": {
            **phase2a.metrics,
            "readability_score": s2a,
            "telemetry": phase2a.telemetry,
            "best_model": phase2a.best_model,
        },
        "2b": {
            **phase2b.metrics,
            "readability_score": s2b,
            "telemetry": phase2b.telemetry,
            "best_model": phase2b.best_model,
        },
        "delta_readability_2b_minus_2a": float(s2b - s2a),
    }

    # Final recommendation
    phase2_pass = (
        phase2b.metrics["val_loss_mean"] <= red["val_max"]
        and phase2b.metrics["tokens_per_sec_mean"] >= red["tps_min"]
        and (s2b - s2a) >= 0.2
    )

    recommendation = {
        "phase1_best_plan": best_name,
        "phase2b_pass": bool(phase2_pass),
        "action": (
            "Adopt Phase1 best plan + seq_len=192 profile"
            if phase2_pass
            else "Keep Phase1 best plan and hold seq_len=128; retry 2b with longer run"
        ),
        "notes": [
            "This run is QUICK mode (2 chunks per plan, 1MB/chunk) for end-to-end validation.",
            "For production decision, rerun winner plans with >=50 chunks.",
        ],
    }

    out = {
        "mode": "quick",
        "quick_config": {
            "total_chunks": QUICK_TOTAL_CHUNKS,
            "chunk_size_mb": QUICK_CHUNK_SIZE_MB,
            "dim": QUICK_DIM,
            "depth": QUICK_DEPTH,
            "hash_dim": QUICK_HASH_DIM,
            "num_buckets": QUICK_NUM_BUCKETS,
            "hamming_thresh": QUICK_HAMMING_THRESH,
        },
        "red_lines": red,
        "phase1": phase1_summary,
        "phase2": phase2_summary,
        "generation_samples": {
            "2a": gen_2a["samples"],
            "2b": gen_2b["samples"],
        },
        "recommendation": recommendation,
    }

    json_path = WORKDIR / "final_test_report.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown report
    md_lines = []
    md_lines.append("# Final Test Report (Phase1 + Phase2)")
    md_lines.append("")
    md_lines.append("## Scope")
    md_lines.append("- Completed Phase1 A/B/C throughput tests")
    md_lines.append("- Completed Phase2 2a/2b character-usability tests")
    md_lines.append("- Generated recommendation under resource red lines")
    md_lines.append("")
    md_lines.append("## Run Mode")
    md_lines.append(f"- quick mode: total_chunks={QUICK_TOTAL_CHUNKS}, chunk_size_mb={QUICK_CHUNK_SIZE_MB}")
    md_lines.append("")
    md_lines.append("## Red Lines")
    md_lines.append(f"- val_loss <= {red['val_max']:.4f}")
    md_lines.append(f"- tokens_per_sec >= {red['tps_min']:.0f}")
    md_lines.append(f"- vram_alloc_gb <= {red['vram_max']:.3f}")
    md_lines.append("")

    md_lines.append("## Phase1 Results")
    for n, s in phase1_summary.items():
        md_lines.append(
            f"- {n}: tps={s['tokens_per_sec_mean']:.0f} ({s['tps_gain_pct_vs_baseline']:+.1f}%), "
            f"val_loss={s['val_loss_mean']:.4f} ({s['val_loss_delta_vs_baseline']:+.4f}), "
            f"vram={s['vram_alloc_gb_mean']:.3f}, overall_pass={s['overall_pass']}"
        )
    md_lines.append("")

    md_lines.append("## Phase2 Results")
    md_lines.append(
        f"- 2a: tps={phase2a.metrics['tokens_per_sec_mean']:.0f}, val_loss={phase2a.metrics['val_loss_mean']:.4f}, "
        f"readability={s2a:.3f}"
    )
    md_lines.append(
        f"- 2b: tps={phase2b.metrics['tokens_per_sec_mean']:.0f}, val_loss={phase2b.metrics['val_loss_mean']:.4f}, "
        f"readability={s2b:.3f}"
    )
    md_lines.append(f"- readability delta (2b-2a) = {s2b - s2a:+.3f}")
    md_lines.append("")

    md_lines.append("## Recommendation")
    md_lines.append(f"- phase1_best_plan: {recommendation['phase1_best_plan']}")
    md_lines.append(f"- phase2b_pass: {recommendation['phase2b_pass']}")
    md_lines.append(f"- action: {recommendation['action']}")
    for note in recommendation["notes"]:
        md_lines.append(f"- note: {note}")

    (WORKDIR / "FINAL_TEST_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n[OK] Generated reports:")
    print(" - final_test_report.json")
    print(" - FINAL_TEST_REPORT.md")


if __name__ == "__main__":
    main()
