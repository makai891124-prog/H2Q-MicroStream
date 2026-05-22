from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with code {code}: {' '.join(cmd)}")


def load_last_row(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    out: Dict[str, float] = {}
    for k, v in last.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def summarize_run(telemetry_csv: Path) -> Dict[str, float]:
    main_last = load_last_row(telemetry_csv)
    core_last = load_last_row(Path(str(telemetry_csv).replace(".csv", "_core.csv")))

    return {
        "train_loss_last": main_last.get("train_loss", float("nan")),
        "val_loss_last": main_last.get("val_loss", float("nan")),
        "sta_sparsity_last": main_last.get("sta_sparsity", float("nan")),
        "tcrh_connectivity_last": main_last.get("tcrh_connectivity", float("nan")),
        "mahler_order_last": main_last.get("mahler_dominant_order", float("nan")),
        "ungs_loss_last": core_last.get("ungs_loss", main_last.get("ungs_loss", float("nan"))),
        "relation_density_last": core_last.get("relation_density", main_last.get("relation_density", float("nan"))),
        "hierarchy_ratio_last": core_last.get("hierarchy_ratio", main_last.get("hierarchy_ratio", float("nan"))),
        "self_ref_consistency_last": core_last.get("self_ref_consistency", main_last.get("self_ref_consistency", float("nan"))),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Quick A/B launcher for UNGS toggle on AGI V2 trainer")
    p.add_argument("--workspace", type=str, default=".")
    p.add_argument("--total-chunks", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--chunk-size-mb", type=int, default=10)
    p.add_argument("--eval-window-multiplier", type=int, default=1000)
    p.add_argument("--supervise-every", type=int, default=0)
    p.add_argument("--adaptive-control-enabled", type=int, default=1)
    p.add_argument("--source-dir", type=str, default=r"E:\Datasets\FineWeb-Edu_Full")
    p.add_argument("--buffer-dir", type=str, default=r"D:\H2Q_Cache_Zone")
    p.add_argument("--output-json", type=str, default="ungs_ab_quick_report.json")
    args = p.parse_args()

    ws = Path(args.workspace).resolve()

    control_csv = ws / "telemetry_ab_control.csv"
    ungs_csv = ws / "telemetry_ab_ungs.csv"

    common = [
        "python",
        "agi_joint_v2_trainer.py",
        "--total-chunks",
        str(args.total_chunks),
        "--seed",
        str(args.seed),
        "--dim",
        str(args.dim),
        "--depth",
        str(args.depth),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--chunk-size-mb",
        str(args.chunk_size_mb),
        "--eval-window-multiplier",
        str(args.eval_window_multiplier),
        "--supervise-every",
        str(args.supervise_every),
        "--adaptive-control-enabled",
        str(args.adaptive_control_enabled),
        "--source-dir",
        args.source_dir,
        "--buffer-dir",
        args.buffer_dir,
    ]

    run_cmd(
        common
        + [
            "--checkpoint-path",
            "ab_control.pt",
            "--best-model-path",
            "ab_control_best.pt",
            "--telemetry-csv",
            str(control_csv),
            "--ungs-enabled",
            "0",
        ],
        ws,
    )

    run_cmd(
        common
        + [
            "--checkpoint-path",
            "ab_ungs.pt",
            "--best-model-path",
            "ab_ungs_best.pt",
            "--telemetry-csv",
            str(ungs_csv),
            "--ungs-enabled",
            "1",
        ],
        ws,
    )

    control = summarize_run(control_csv)
    ungs = summarize_run(ungs_csv)

    delta = {
        "delta_val_loss": control["val_loss_last"] - ungs["val_loss_last"],
        "delta_relation_density": ungs["relation_density_last"] - control["relation_density_last"],
        "delta_hierarchy_ratio": ungs["hierarchy_ratio_last"] - control["hierarchy_ratio_last"],
        "delta_self_ref_consistency": ungs["self_ref_consistency_last"] - control["self_ref_consistency_last"],
    }

    report = {
        "control": control,
        "ungs": ungs,
        "delta": delta,
    }

    out_path = ws / args.output_json
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ab] report={out_path}")


if __name__ == "__main__":
    main()
