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
        )

    ema = [to_float(x["Causal_Loss_EMA"]) for x in rows]
    sp = [to_float(x["Topology_Sparsity"]) for x in rows]
    svd = [to_float(x["SVD_Entropy"]) for x in rows]
    vram = [to_float(x["VRAM_Allocated_MB"]) for x in rows]
    steps = [int(float(x["T_Step"])) for x in rows]

    k = min(10, len(sp))
    mean_last_k = statistics.fmean(sp[-k:]) if k > 0 else float("nan")
    phase_like = (min(ema) < ema[0]) and (max(sp) > 0.5)

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
    )


def next_policy(prev_lr: float, metric: CycleMetrics) -> dict:
    # Simple self-correction policy:
    # - if topology sparsity remains low, increase structural pressure by reducing LR slightly
    # - if entropy collapses too much, relax by increasing LR slightly
    lr = prev_lr
    if metric.sparsity_max < 0.2:
        lr = max(1e-5, prev_lr * 0.85)
    elif metric.svd_entropy_last < 1.6:
        lr = min(1e-3, prev_lr * 1.1)

    return {
        "lr": lr,
        "target_sparsity": 0.5,
        "phase_trigger_like": metric.phase_trigger_like,
    }

def generate_hypotheses(metric: CycleMetrics) -> List[dict]:
    hyps: List[dict] = []

    hyps.append(
        {
            "hypothesis_id": f"H-EMA-{metric.cycle_id}",
            "cycle_proposed": metric.cycle_id,
            "statement": "If high sparsity appears, next cycle EMA minimum should improve by at least 0.05",
            "check_type": "ema_improve",
            "baseline": metric.ema_min,
            "threshold": 0.05,
        }
    )

    hyps.append(
        {
            "hypothesis_id": f"H-SP-{metric.cycle_id}",
            "cycle_proposed": metric.cycle_id,
            "statement": "If SVD entropy stays healthy, next cycle should reach sparsity > 0.5 at least once",
            "check_type": "sparsity_peak",
            "baseline": metric.svd_entropy_last,
            "threshold": 0.5,
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
