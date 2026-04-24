"""
Dynamic strict acceptance for long-run H2Q telemetry.

Inputs:
- baseline snapshot json
- daemon telemetry csv (from local_evolution_daemon.py)
- optional hypotheses jsonl (from world_model_autopilot.py)

Outputs:
- acceptance_verdict.json
- acceptance_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List, Tuple


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_baseline(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "val_loss_max": float(data["val_loss_max"]),
        "tokens_per_sec_min": float(data["tokens_per_sec_min"]),
        "vram_alloc_max_gb": float(data["vram_alloc_max"]),
    }


def load_telemetry(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "step": _safe_float(r.get("T_Step", "nan")),
                    "ema": _safe_float(r.get("Causal_Loss_EMA", "nan")),
                    "sparsity": _safe_float(r.get("Topology_Sparsity", "nan")),
                    "svd": _safe_float(r.get("SVD_Entropy", "nan")),
                    "vram_mb": _safe_float(r.get("VRAM_Allocated_MB", "nan")),
                    "step_per_sec": _safe_float(r.get("StepPerSec", "nan")),
                }
            )
    return rows


def window_cvs(values: List[float], window: int) -> List[float]:
    out: List[float] = []
    if window <= 1:
        return out
    for i in range(0, len(values) - window + 1):
        w = values[i : i + window]
        mu = fmean(w)
        if abs(mu) < 1e-12:
            out.append(float("inf"))
        else:
            out.append(abs(pstdev(w) / mu))
    return out


def detect_phase_trigger_count(rows: List[Dict[str, float]], min_gap_rows: int = 5) -> int:
    """
    A simple phase-trigger proxy:
    count events where sparsity >= 0.5 and EMA reaches a new minimum,
    with a row-gap debounce.
    """
    best_ema = float("inf")
    count = 0
    last_idx = -10**9
    for idx, row in enumerate(rows):
        ema = row["ema"]
        sp = row["sparsity"]
        if math.isnan(ema) or math.isnan(sp):
            continue
        if ema < best_ema and sp >= 0.5 and (idx - last_idx) >= min_gap_rows:
            best_ema = ema
            count += 1
            last_idx = idx
        else:
            if ema < best_ema:
                best_ema = ema
    return count


def evaluate_hypotheses(path: Path) -> Tuple[bool, Dict[str, float]]:
    if not path.exists():
        return False, {
            "checked": 0,
            "supported": 0,
            "support_rate": 0.0,
        }

    checked = 0
    supported = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            status = str(obj.get("status", "")).lower()
            if status in {"supported", "falsified"}:
                checked += 1
                if status == "supported":
                    supported += 1

    support_rate = (supported / checked) if checked > 0 else 0.0
    passed = checked > 0 and support_rate >= (2.0 / 3.0)
    return passed, {
        "checked": checked,
        "supported": supported,
        "support_rate": support_rate,
    }


def build_report_md(result: Dict) -> str:
    def _fmt(v: object, digits: int = 6) -> str:
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return "N/A"
            return f"{v:.{digits}f}"
        if v is None:
            return "N/A"
        return str(v)

    a = result["gates"]["A"]
    b = result["gates"]["B"]
    c = result["gates"]["C"]
    d = result["gates"]["D"]

    lines = [
        "# Dynamic Acceptance Report",
        "",
        f"- UTC: {result['utc']}",
        f"- Telemetry: {result['telemetry_path']}",
        f"- Verdict: **{result['verdict']}**",
        "",
        "## Gate A (base thresholds)",
        f"- pass: {a['pass']}",
        f"- ema_last={_fmt(a['ema_last'])} <= val_loss_max={_fmt(a['val_loss_max'])}",
        f"- vram_last_gb={_fmt(a['vram_last_gb'])} <= vram_max_gb={_fmt(a['vram_max_gb'])}",
        f"- tokens_per_sec={_fmt(a['tokens_per_sec'], 2)} >= tokens_per_sec_min={_fmt(a['tokens_per_sec_min'], 2)}",
        f"- tps_available={a['tps_available']}",
        "",
        "## Gate B (window stability)",
        f"- pass: {b['pass']}",
        f"- max_loss_cv={_fmt(b['max_loss_cv'])} (threshold={_fmt(b['loss_cv_threshold'])})",
        f"- max_sparsity_cv={_fmt(b['max_sparsity_cv'])} (threshold={_fmt(b['sparsity_cv_threshold'])})",
        f"- window={b['window']}",
        "",
        "## Gate C (topology state)",
        f"- pass: {c['pass']}",
        f"- sparsity_peak={_fmt(c['sparsity_peak'])} (target={_fmt(c['sparsity_target'])})",
        f"- svd_last={_fmt(c['svd_last'])} (min={_fmt(c['svd_min'])})",
        f"- phase_trigger_count={c['phase_trigger_count']} (max={c['phase_trigger_max']})",
        "",
        "## Gate D (hypothesis support)",
        f"- pass: {d['pass']}",
        f"- checked={d['checked']}, supported={d['supported']}, support_rate={_fmt(d['support_rate'])}",
        "",
    ]
    return "\n".join(lines) + "\n"


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="Strict dynamic acceptance for daemon telemetry")
    p.add_argument("--baseline", type=str, default="baseline_snapshot.json")
    p.add_argument("--telemetry", type=str, required=True)
    p.add_argument("--hypotheses", type=str, default="autopilot_hypotheses.jsonl")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--loss-cv-threshold", type=float, default=0.10)
    p.add_argument("--sparsity-cv-threshold", type=float, default=0.20)
    p.add_argument("--sparsity-target", type=float, default=0.50)
    p.add_argument("--svd-min", type=float, default=1.60)
    p.add_argument("--phase-trigger-max", type=int, default=3)
    p.add_argument("--allow-missing-tps", action="store_true")
    p.add_argument("--output-json", type=str, default="acceptance_verdict.json")
    p.add_argument("--output-md", type=str, default="acceptance_report.md")
    args = p.parse_args()

    baseline = load_baseline(Path(args.baseline))
    rows = load_telemetry(Path(args.telemetry))
    if not rows:
        raise RuntimeError(f"Empty telemetry: {args.telemetry}")

    ema_values = [r["ema"] for r in rows if not math.isnan(r["ema"])]
    sp_values = [r["sparsity"] for r in rows if not math.isnan(r["sparsity"])]
    svd_values = [r["svd"] for r in rows if not math.isnan(r["svd"])]

    last = rows[-1]
    ema_last = last["ema"]
    vram_last_gb = last["vram_mb"] / 1024.0
    tps_available = not math.isnan(last["step_per_sec"])
    tokens_per_sec = (last["step_per_sec"] * float(args.seq_len)) if tps_available else float("nan")

    tps_pass = (
        tokens_per_sec >= baseline["tokens_per_sec_min"]
        if tps_available
        else bool(args.allow_missing_tps)
    )

    gate_a = {
        "pass": (
            ema_last <= baseline["val_loss_max"]
            and vram_last_gb <= baseline["vram_alloc_max_gb"]
            and tps_pass
        ),
        "ema_last": ema_last,
        "val_loss_max": baseline["val_loss_max"],
        "vram_last_gb": vram_last_gb,
        "vram_max_gb": baseline["vram_alloc_max_gb"],
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_min": baseline["tokens_per_sec_min"],
        "tps_available": tps_available,
    }

    loss_cvs = window_cvs(ema_values, args.window)
    sp_cvs = window_cvs(sp_values, args.window)
    max_loss_cv = max(loss_cvs) if loss_cvs else float("inf")
    max_sparsity_cv = max(sp_cvs) if sp_cvs else float("inf")

    gate_b = {
        "pass": (
            len(loss_cvs) > 0
            and len(sp_cvs) > 0
            and max_loss_cv <= args.loss_cv_threshold
            and max_sparsity_cv <= args.sparsity_cv_threshold
        ),
        "max_loss_cv": max_loss_cv,
        "max_sparsity_cv": max_sparsity_cv,
        "loss_cv_threshold": args.loss_cv_threshold,
        "sparsity_cv_threshold": args.sparsity_cv_threshold,
        "window": args.window,
    }

    sparsity_peak = max(sp_values) if sp_values else float("nan")
    svd_last = svd_values[-1] if svd_values else float("nan")
    phase_trigger_count = detect_phase_trigger_count(rows)

    gate_c = {
        "pass": (
            sparsity_peak >= args.sparsity_target
            and svd_last >= args.svd_min
            and phase_trigger_count <= args.phase_trigger_max
        ),
        "sparsity_peak": sparsity_peak,
        "sparsity_target": args.sparsity_target,
        "svd_last": svd_last,
        "svd_min": args.svd_min,
        "phase_trigger_count": phase_trigger_count,
        "phase_trigger_max": args.phase_trigger_max,
    }

    hypo_pass, hypo_stats = evaluate_hypotheses(Path(args.hypotheses))
    gate_d = {
        "pass": hypo_pass,
        **hypo_stats,
    }

    if gate_a["pass"] and gate_b["pass"] and gate_c["pass"] and gate_d["pass"]:
        verdict = "ACCEPT"
    elif gate_a["pass"] and gate_b["pass"] and gate_c["pass"]:
        verdict = "CONDITIONAL_ACCEPT"
    elif gate_a["pass"]:
        verdict = "RETEST"
    else:
        verdict = "REJECT"

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "telemetry_path": args.telemetry,
        "baseline_path": args.baseline,
        "gates": {
            "A": gate_a,
            "B": gate_b,
            "C": gate_c,
            "D": gate_d,
        },
    }

    Path(args.output_json).write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.output_md).write_text(build_report_md(result), encoding="utf-8")

    print(f"[acceptance] verdict={verdict}")
    print(f"[acceptance] json={args.output_json}")
    print(f"[acceptance] md={args.output_md}")


if __name__ == "__main__":
    main()
