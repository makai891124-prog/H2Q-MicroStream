#!/usr/bin/env python3
"""Finalize report from already completed experiment artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

WORKDIR = Path(__file__).resolve().parent
TRAINER = WORKDIR / "agi_joint_trainer.py"


def load_baseline():
    snap = WORKDIR / "baseline_snapshot.json"
    if snap.exists():
        data = json.loads(snap.read_text(encoding="utf-8"))
        val_base = float(data.get("val_loss_μ", 1.7873))
        tps_base = float(data.get("tokens_per_sec_μ", 18014))
    else:
        df = pd.read_csv(WORKDIR / "agi_joint_telemetry.csv")
        w = df.tail(min(100, len(df)))
        val_base = float(w["val_loss"].mean())
        tps_base = float(w["tokens_per_sec"].mean())
    return {
        "val_base": val_base,
        "tps_base": tps_base,
        "val_max": val_base * 1.01,
        "tps_min": tps_base * 0.90,
        "vram_max": 0.25,
    }


def stats_from_csv(path: Path):
    df = pd.read_csv(path)
    w = df.tail(min(50, len(df)))
    return {
        "train_loss_mean": float(w["train_loss"].mean()),
        "val_loss_mean": float(w["val_loss"].mean()),
        "tokens_per_sec_mean": float(w["tokens_per_sec"].mean()),
        "vram_alloc_gb_mean": float(w["vram_alloc_gb"].mean()),
        "rows": int(len(w)),
    }


def char_stats(text: str):
    if not text:
        return {"invalid_char_rate": 1.0, "repeat_bigram_rate": 1.0, "max_run": 0, "readability_score": 0.0}
    invalid = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    invalid_rate = invalid / len(text)
    bgs = [text[i : i + 2] for i in range(max(0, len(text) - 1))]
    repeat = 1.0 - (len(set(bgs)) / len(bgs)) if bgs else 0.0
    run, max_run = 1, 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    score = (
        (1 - min(invalid_rate, 0.05) / 0.05) * 0.3
        + (1 - min(repeat, 0.3) / 0.3) * 0.3
        + (1 - min(max_run, 20) / 20) * 0.2
        + (0.2 if " " in text else 0.0)
    )
    return {
        "invalid_char_rate": float(invalid_rate),
        "repeat_bigram_rate": float(repeat),
        "max_run": int(max_run),
        "readability_score": float(max(0.0, min(1.0, score))),
    }


def generation_eval(best_model_path: Path, prompts):
    spec = importlib.util.spec_from_file_location("agi_joint_trainer_eval", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    device = torch.device("cuda:0")
    ckpt = torch.load(best_model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = mod.AGI_Accelerated_Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = []
    scores = []
    with torch.no_grad():
        for p in prompts:
            ids = list(p.encode("utf-8"))
            ctx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(ctx, 160)
            text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
            st = char_stats(text)
            results.append({"prompt": p, "sample": text[:220], **st})
            scores.append(st["readability_score"])

    return {"mean_readability_score": float(np.mean(scores)), "samples": results}


def main():
    red = load_baseline()

    phase1_files = {
        "A_baseline": WORKDIR / "telemetry_A_baseline.csv",
        "B_eval_down": WORKDIR / "telemetry_B_eval_down.csv",
        "C_no_supervise": WORKDIR / "telemetry_C_no_supervise.csv",
    }

    phase1 = {}
    best_name = None
    best_tps = -1.0
    for name, path in phase1_files.items():
        m = stats_from_csv(path)
        m["loss_pass"] = m["val_loss_mean"] <= red["val_max"]
        m["tps_pass"] = m["tokens_per_sec_mean"] >= red["tps_min"]
        m["vram_pass"] = m["vram_alloc_gb_mean"] <= red["vram_max"]
        m["overall_pass"] = m["loss_pass"] and m["tps_pass"] and m["vram_pass"]
        m["tps_gain_pct_vs_baseline"] = (m["tokens_per_sec_mean"] / red["tps_base"] - 1) * 100
        m["val_loss_delta_vs_baseline"] = m["val_loss_mean"] - red["val_base"]
        phase1[name] = m
        if m["overall_pass"] and m["tokens_per_sec_mean"] > best_tps:
            best_tps = m["tokens_per_sec_mean"]
            best_name = name

    if best_name is None:
        best_name = max(phase1, key=lambda n: phase1[n]["tokens_per_sec_mean"])

    phase2 = {
        "2a": stats_from_csv(WORKDIR / "telemetry_2a_char_baseline.csv"),
        "2b": stats_from_csv(WORKDIR / "telemetry_2b_seq192.csv"),
    }

    prompts = ["The ", "Why ", "Character-level model ", "In this experiment, "]
    g2a = generation_eval(WORKDIR / "best_2a_char_baseline.pt", prompts)
    g2b = generation_eval(WORKDIR / "best_2b_seq192.pt", prompts)
    phase2["2a"]["readability_score"] = g2a["mean_readability_score"]
    phase2["2b"]["readability_score"] = g2b["mean_readability_score"]
    phase2_delta = g2b["mean_readability_score"] - g2a["mean_readability_score"]

    phase2_pass = (
        phase2["2b"]["val_loss_mean"] <= red["val_max"]
        and phase2["2b"]["tokens_per_sec_mean"] >= red["tps_min"]
        and phase2_delta >= 0.2
    )

    rec = {
        "phase1_best_plan": best_name,
        "phase2b_pass": bool(phase2_pass),
        "action": (
            "Adopt phase1 best + seq_len=192 profile" if phase2_pass
            else "Keep phase1 best and hold seq_len=128; rerun 2b with longer chunks before promotion"
        ),
    }

    out = {
        "mode": "quick",
        "red_lines": red,
        "phase1": phase1,
        "phase2": {**phase2, "delta_readability_2b_minus_2a": phase2_delta},
        "generation_samples": {"2a": g2a["samples"], "2b": g2b["samples"]},
        "recommendation": rec,
    }

    (WORKDIR / "final_test_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Final Test Report (Completed)",
        "",
        "## Phase1 Summary",
    ]
    for k, v in phase1.items():
        lines.append(
            f"- {k}: tps={v['tokens_per_sec_mean']:.0f}, val_loss={v['val_loss_mean']:.4f}, "
            f"vram={v['vram_alloc_gb_mean']:.3f}, pass={v['overall_pass']}"
        )
    lines += [
        "",
        "## Phase2 Summary",
        f"- 2a: tps={phase2['2a']['tokens_per_sec_mean']:.0f}, val_loss={phase2['2a']['val_loss_mean']:.4f}, readability={phase2['2a']['readability_score']:.3f}",
        f"- 2b: tps={phase2['2b']['tokens_per_sec_mean']:.0f}, val_loss={phase2['2b']['val_loss_mean']:.4f}, readability={phase2['2b']['readability_score']:.3f}",
        f"- delta_readability (2b-2a): {phase2_delta:+.3f}",
        "",
        "## Recommendation",
        f"- phase1_best_plan: {best_name}",
        f"- phase2b_pass: {phase2_pass}",
        f"- action: {rec['action']}",
    ]
    (WORKDIR / "FINAL_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[OK] Generated final_test_report.json and FINAL_TEST_REPORT.md")


if __name__ == "__main__":
    main()
