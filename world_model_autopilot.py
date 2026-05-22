"""
world_model_autopilot.py
========================
Self-driving orchestration loop for H2Q world-model evolution.

Loop:
1) Build open corpus (arXiv + Hugging Face + GitHub metadata/text)
2) Run local evolution daemon for a bounded cycle
3) Analyze telemetry and derive next-cycle policy updates
4) Persist state/report and continue

This is an automation layer on top of local_evolution_daemon.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List


@dataclass
class CycleMetrics:
    cycle_id: int
    steps: int
    ema_min: float
    ema_last: float
    sparsity_mean_last10: float
    sparsity_max: float
    svd_entropy_last: float
    vram_mb_last: float
    phase_trigger_like: bool
    relation_density_last10: float
    hierarchy_ratio_last10: float
    self_ref_consistency_last10: float
    ungs_loss_last10: float
    axiom_residual: float

@dataclass
class HypothesisRecord:
    hypothesis_id: str
    cycle_proposed: int
    statement: str
    check_type: str
    baseline: float
    threshold: float
    status: str
    checked_cycle: int
    evidence: str


def run_cmd(cmd: List[str], cwd: str) -> None:
    p = subprocess.Popen(cmd, cwd=cwd)
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with code {code}: {' '.join(cmd)}")


def build_daemon_cmd(
    seq_len: int,
    lr: float,
    telemetry: str,
    steps_per_cycle: int,
    cycle_checkpoint: str,
    resume_checkpoint: str,
) -> List[str]:
    cmd = [
        "python",
        "local_evolution_daemon.py",
        "--source",
        "data/open_corpus/open_corpus.bin",
        "--seq-len",
        str(seq_len),
        "--lr",
        str(lr),
        "--final-checkpoint",
        cycle_checkpoint,
        "--telemetry-every",
        "1000",
        "--print-every",
        "1000",
        "--svd-every",
        "1000",
        "--cache-clear-every",
        "10000",
        "--telemetry-csv",
        telemetry,
        "--max-steps",
        str(steps_per_cycle),
    ]
    if resume_checkpoint:
        cmd.extend(["--resume", resume_checkpoint])
    return cmd


def load_telemetry(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def pick_float(row: dict, keys: List[str]) -> float:
    for k in keys:
        if k in row and row.get(k, "") != "":
            return to_float(row.get(k, "nan"))
    return float("nan")


def finite_mean(values: List[float], tail: int = 10) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    k = min(len(finite), max(1, tail))
    return statistics.fmean(finite[-k:])


def analyze_cycle(cycle_id: int, rows: List[dict]) -> CycleMetrics:
    if not rows:
        return CycleMetrics(
            cycle_id=cycle_id,
            steps=0,
            ema_min=float("nan"),
            ema_last=float("nan"),
            sparsity_mean_last10=float("nan"),
            sparsity_max=float("nan"),
            svd_entropy_last=float("nan"),
            vram_mb_last=float("nan"),
            phase_trigger_like=False,
            relation_density_last10=float("nan"),
            hierarchy_ratio_last10=float("nan"),
            self_ref_consistency_last10=float("nan"),
            ungs_loss_last10=float("nan"),
            axiom_residual=float("nan"),
        )

    ema = [pick_float(x, ["Causal_Loss_EMA", "val_loss"]) for x in rows]
    sp = [pick_float(x, ["Topology_Sparsity", "sta_sparsity"]) for x in rows]
    svd = [pick_float(x, ["SVD_Entropy"]) for x in rows]
    vram = []
    for x in rows:
        vram_mb = pick_float(x, ["VRAM_Allocated_MB"])
        if math.isnan(vram_mb):
            vram_gb = pick_float(x, ["vram_alloc_gb"])
            vram_mb = vram_gb * 1024.0 if not math.isnan(vram_gb) else float("nan")
        vram.append(vram_mb)
    rel = [pick_float(x, ["relation_density"]) for x in rows]
    hier = [pick_float(x, ["hierarchy_ratio"]) for x in rows]
    self_ref = [pick_float(x, ["self_ref_consistency"]) for x in rows]
    ungs = [pick_float(x, ["ungs_loss"]) for x in rows]
    step_vals = [pick_float(x, ["T_Step", "chunk"]) for x in rows]
    steps = [int(v) for v in step_vals if not math.isnan(v)]
    if not steps:
        steps = [len(rows)]

    k = min(10, len(sp))
    mean_last_k = statistics.fmean(sp[-k:]) if k > 0 else float("nan")
    phase_like = (min(ema) < ema[0]) and (max(sp) > 0.5)

    rel_last10 = finite_mean(rel, 10)
    hier_last10 = finite_mean(hier, 10)
    self_ref_last10 = finite_mean(self_ref, 10)
    ungs_last10 = finite_mean(ungs, 10)

    # Axiom residual: lower is better.
    residual_terms = []
    if not math.isnan(rel_last10):
        residual_terms.append(max(0.0, 0.08 - rel_last10))
    else:
        residual_terms.append(max(0.0, 0.5 - max(sp)))
    if not math.isnan(hier_last10):
        residual_terms.append(max(0.0, 0.03 - hier_last10))
    else:
        residual_terms.append(max(0.0, 1.6 - (svd[-1] if svd else 0.0)) / 10.0)
    if not math.isnan(self_ref_last10):
        residual_terms.append(max(0.0, 0.60 - self_ref_last10))
    if not math.isnan(ungs_last10):
        residual_terms.append(max(0.0, ungs_last10 - 0.80))

    axiom_residual = sum(residual_terms) / max(len(residual_terms), 1)

    return CycleMetrics(
        cycle_id=cycle_id,
        steps=steps[-1],
        ema_min=min(ema),
        ema_last=ema[-1],
        sparsity_mean_last10=mean_last_k,
        sparsity_max=max(sp),
        svd_entropy_last=svd[-1],
        vram_mb_last=vram[-1],
        phase_trigger_like=phase_like,
        relation_density_last10=rel_last10,
        hierarchy_ratio_last10=hier_last10,
        self_ref_consistency_last10=self_ref_last10,
        ungs_loss_last10=ungs_last10,
        axiom_residual=axiom_residual,
    )


def next_policy(prev_lr: float, metric: CycleMetrics) -> dict:
    # Axiom-residual-driven policy:
    # residual high -> reduce lr and increase structural pressure.
    # residual low  -> cautiously relax lr to avoid over-regularization.
    residual = metric.axiom_residual
    if math.isnan(residual):
        residual = 0.5

    if residual > 0.25:
        lr = max(1e-5, prev_lr * 0.82)
    elif residual > 0.10:
        lr = max(1e-5, prev_lr * 0.92)
    else:
        lr = min(1e-3, prev_lr * 1.03)

    structural_pressure = min(2.0, 1.0 + 2.0 * residual)
    ungs_closure_lambda = min(0.5, 0.05 + 0.08 * residual)
    ungs_encapsulation_lambda = min(0.5, 0.03 + 0.06 * residual)
    ungs_self_ref_lambda = min(0.5, 0.02 + 0.05 * residual)
    axiom_lambda = min(0.5, 0.10 + 0.10 * residual)

    return {
        "lr": lr,
        "target_axiom_residual": 0.10,
        "structural_pressure": structural_pressure,
        "suggest_ungs_closure_lambda": ungs_closure_lambda,
        "suggest_ungs_encapsulation_lambda": ungs_encapsulation_lambda,
        "suggest_ungs_self_ref_lambda": ungs_self_ref_lambda,
        "suggest_axiom_lambda": axiom_lambda,
        "phase_trigger_like": metric.phase_trigger_like,
        "policy_mode": "axiom_residual_driven",
    }

def generate_hypotheses(metric: CycleMetrics) -> List[dict]:
    hyps: List[dict] = []

    hyps.append(
        {
            "hypothesis_id": f"H-AXRES-{metric.cycle_id}",
            "cycle_proposed": metric.cycle_id,
            "statement": "If policy is residual-driven, next cycle axiom_residual should drop by at least 0.03",
            "check_type": "residual_drop",
            "baseline": metric.axiom_residual,
            "threshold": 0.03,
        }
    )

    hyps.append(
        {
            "hypothesis_id": f"H-SREF-{metric.cycle_id}",
            "cycle_proposed": metric.cycle_id,
            "statement": "Self-reference consistency should improve by at least 0.02",
            "check_type": "self_ref_improve",
            "baseline": metric.self_ref_consistency_last10,
            "threshold": 0.02,
        }
    )

    hyps.append(
        {
            "hypothesis_id": f"H-VRAM-{metric.cycle_id}",
            "cycle_proposed": metric.cycle_id,
            "statement": "VRAM allocation should remain approximately constant in next cycle",
            "check_type": "vram_stable",
            "baseline": metric.vram_mb_last,
            "threshold": 1.0,
        }
    )

    return hyps


def validate_hypotheses(pending: List[dict], metric: CycleMetrics) -> List[HypothesisRecord]:
    out: List[HypothesisRecord] = []

    for h in pending:
        h_id = h.get("hypothesis_id", "H-UNKNOWN")
        c = h.get("check_type", "")
        baseline = float(h.get("baseline", 0.0))
        threshold = float(h.get("threshold", 0.0))
        status = "falsified"
        evidence = ""

        if c == "ema_improve":
            delta = baseline - metric.ema_min
            status = "supported" if delta >= threshold else "falsified"
            evidence = f"baseline_ema_min={baseline:.6f}, next_ema_min={metric.ema_min:.6f}, delta={delta:.6f}"
        elif c == "residual_drop":
            delta = baseline - metric.axiom_residual
            status = "supported" if delta >= threshold else "falsified"
            evidence = f"baseline_residual={baseline:.6f}, next_residual={metric.axiom_residual:.6f}, delta={delta:.6f}"
        elif c == "self_ref_improve":
            delta = metric.self_ref_consistency_last10 - baseline
            status = "supported" if delta >= threshold else "falsified"
            evidence = f"baseline_self_ref={baseline:.6f}, next_self_ref={metric.self_ref_consistency_last10:.6f}, delta={delta:.6f}"
        elif c == "sparsity_peak":
            status = "supported" if metric.sparsity_max > threshold else "falsified"
            evidence = f"next_sparsity_max={metric.sparsity_max:.6f}, threshold={threshold:.6f}"
        elif c == "vram_stable":
            drift = abs(metric.vram_mb_last - baseline)
            status = "supported" if drift <= threshold else "falsified"
            evidence = f"baseline_vram={baseline:.4f}, next_vram={metric.vram_mb_last:.4f}, drift={drift:.4f}"
        else:
            evidence = "unknown check type"

        out.append(
            HypothesisRecord(
                hypothesis_id=h_id,
                cycle_proposed=int(h.get("cycle_proposed", 0)),
                statement=h.get("statement", ""),
                check_type=c,
                baseline=baseline,
                threshold=threshold,
                status=status,
                checked_cycle=metric.cycle_id,
                evidence=evidence,
            )
        )

    return out


def append_report(
    path: str,
    metric: CycleMetrics,
    policy: dict,
    validated: List[HypothesisRecord],
    proposed: List[dict],
) -> None:
    lines = [
        f"## Cycle {metric.cycle_id}",
        f"- UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Steps: {metric.steps}",
        f"- EMA min: {metric.ema_min:.6f}",
        f"- EMA last: {metric.ema_last:.6f}",
        f"- Sparsity max: {metric.sparsity_max:.4f}",
        f"- Sparsity mean(last10): {metric.sparsity_mean_last10:.4f}",
        f"- SVD entropy(last): {metric.svd_entropy_last:.4f}",
        f"- VRAM MB(last): {metric.vram_mb_last:.4f}",
        f"- Axiom residual(last10): {metric.axiom_residual:.6f}",
        f"- Relation density(last10): {metric.relation_density_last10:.6f}",
        f"- Hierarchy ratio(last10): {metric.hierarchy_ratio_last10:.6f}",
        f"- Self-ref consistency(last10): {metric.self_ref_consistency_last10:.6f}",
        f"- UNGS loss(last10): {metric.ungs_loss_last10:.6f}",
        f"- Phase-like trigger observed: {metric.phase_trigger_like}",
        f"- Next policy: {json.dumps(policy, ensure_ascii=False)}",
        f"- Hypothesis checks: {len(validated)}",
    ]

    for rec in validated:
        lines.append(
            f"  - [{rec.status}] {rec.hypothesis_id}: {rec.statement} | {rec.evidence}"
        )

    lines.append(f"- Hypothesis proposed: {len(proposed)}")
    for h in proposed:
        lines.append(
            f"  - [proposed] {h.get('hypothesis_id')}: {h.get('statement')}"
        )

    lines.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-driving H2Q world model orchestrator")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps-per-cycle", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--initial-lr", type=float, default=3e-4)
    parser.add_argument("--arxiv-max", type=int, default=200)
    parser.add_argument("--arxiv-pdf-max", type=int, default=24)
    parser.add_argument("--arxiv-pdf-pages", type=int, default=6)
    parser.add_argument("--arxiv-pdf-max-bytes", type=int, default=12582912)
    parser.add_argument("--hf-max", type=int, default=40)
    parser.add_argument("--target-mb", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)
    state_path = os.path.join(ws, "autopilot_state.json")
    report_path = os.path.join(ws, "autopilot_report.md")
    hypo_log_path = os.path.join(ws, "autopilot_hypotheses.jsonl")

    lr = args.initial_lr
    pending_hypotheses: List[dict] = []
    resume_checkpoint = ""
    if os.path.isfile(state_path):
        try:
            old = json.load(open(state_path, "r", encoding="utf-8"))
            lr = float(old.get("lr", lr))
            pending_hypotheses = old.get("pending_hypotheses", [])
            saved_resume = old.get("last_checkpoint", "")
            if saved_resume and os.path.isfile(saved_resume):
                resume_checkpoint = saved_resume
        except Exception:
            pass

    for cycle in range(1, args.cycles + 1):
        telemetry = os.path.join(ws, f"evolution_telemetry_cycle_{cycle}.csv")
        cycle_checkpoint = os.path.join(ws, f"h2q_cycle_{cycle}.pt")

        run_cmd(
            [
                "python",
                "build_open_corpus.py",
                "--arxiv-max",
                str(args.arxiv_max),
                "--arxiv-pdf-max",
                str(args.arxiv_pdf_max),
                "--arxiv-pdf-pages",
                str(args.arxiv_pdf_pages),
                "--arxiv-pdf-max-bytes",
                str(args.arxiv_pdf_max_bytes),
                "--hf-max",
                str(args.hf_max),
                "--target-mb",
                str(args.target_mb),
                "--timeout",
                str(args.timeout),
            ],
            cwd=ws,
        )

        run_cmd(
            build_daemon_cmd(
                seq_len=args.seq_len,
                lr=lr,
                telemetry=telemetry,
                steps_per_cycle=args.steps_per_cycle,
                cycle_checkpoint=cycle_checkpoint,
                resume_checkpoint=resume_checkpoint,
            ),
            cwd=ws,
        )

        if os.path.isfile(cycle_checkpoint):
            resume_checkpoint = cycle_checkpoint

        rows = load_telemetry(telemetry)
        metric = analyze_cycle(cycle, rows)
        policy = next_policy(lr, metric)
        lr = policy["lr"]

        validated = validate_hypotheses(pending_hypotheses, metric)
        proposed = generate_hypotheses(metric)
        pending_hypotheses = proposed

        with open(hypo_log_path, "a", encoding="utf-8") as f:
            for rec in validated:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            for h in proposed:
                f.write(
                    json.dumps(
                        {
                            "status": "proposed",
                            "cycle": cycle,
                            **h,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        json.dump(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle,
                "lr": lr,
                "last_checkpoint": resume_checkpoint,
                "last_metric": asdict(metric),
                "last_policy": policy,
                "pending_hypotheses": pending_hypotheses,
            },
            open(state_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        append_report(report_path, metric, policy, validated, proposed)

        # Small cooldown to avoid immediate restart jitter.
        time.sleep(2)

    print("[autopilot] finished")
    print(f"[autopilot] report={report_path}")
    print(f"[autopilot] state={state_path}")


if __name__ == "__main__":
    main()
