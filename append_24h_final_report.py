from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Dict, List


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_rollup(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run_telemetry(root: Path, run_name: str) -> List[Path]:
    return sorted(root.glob(f"evolution_telemetry_{run_name}_seg*.csv"))


def summarize_run(files: List[Path]) -> Dict[str, float]:
    if not files:
        return {
            "segments": 0,
            "ema_last_mean": float("nan"),
            "sparsity_peak_max": float("nan"),
            "svd_last_mean": float("nan"),
            "vram_last_gb_mean": float("nan"),
            "tps_last_mean": float("nan"),
        }

    ema_last_vals = []
    sparsity_peak_vals = []
    svd_last_vals = []
    vram_last_gb_vals = []
    tps_last_vals = []

    for fp in files:
        rows = []
        with fp.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)

        if not rows:
            continue

        ema = [_safe_float(x.get("Causal_Loss_EMA", "nan")) for x in rows]
        sp = [_safe_float(x.get("Topology_Sparsity", "nan")) for x in rows]
        svd = [_safe_float(x.get("SVD_Entropy", "nan")) for x in rows]
        vram_mb = [_safe_float(x.get("VRAM_Allocated_MB", "nan")) for x in rows]
        sps = [_safe_float(x.get("StepPerSec", "nan")) for x in rows]

        if ema:
            ema_last_vals.append(ema[-1])
        if sp:
            sparsity_peak_vals.append(max(sp))
        if svd:
            svd_last_vals.append(svd[-1])
        if vram_mb:
            vram_last_gb_vals.append(vram_mb[-1] / 1024.0)
        if sps:
            tps_last_vals.append(sps[-1] * 1024.0)

    def _mean(xs: List[float]) -> float:
        return fmean(xs) if xs else float("nan")

    return {
        "segments": len(files),
        "ema_last_mean": _mean(ema_last_vals),
        "sparsity_peak_max": max(sparsity_peak_vals) if sparsity_peak_vals else float("nan"),
        "svd_last_mean": _mean(svd_last_vals),
        "vram_last_gb_mean": _mean(vram_last_gb_vals),
        "tps_last_mean": _mean(tps_last_vals),
    }


def render_auto_section(rollup: dict, main_stats: dict, ctrl_stats: dict) -> str:
    overall = rollup.get("overall_verdict", "RETEST")
    comp = rollup.get("main_vs_ctrl", {})

    lines: List[str] = []
    lines.append("### 9.6 24小时主跑与对照最终汇总（自动生成）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 全局最终裁决：{overall}")
    lines.append("")
    lines.append("#### 9.6.1 主跑 main24h 概览")
    lines.append(f"- 分段数：{main_stats['segments']}")
    lines.append(f"- EMA(last)均值：{main_stats['ema_last_mean']:.6f}")
    lines.append(f"- Sparsity峰值（跨段最大）：{main_stats['sparsity_peak_max']:.6f}")
    lines.append(f"- SVD(last)均值：{main_stats['svd_last_mean']:.6f}")
    lines.append(f"- VRAM(last)均值(GB)：{main_stats['vram_last_gb_mean']:.6f}")
    lines.append(f"- TPS(last)均值（StepPerSec×1024）：{main_stats['tps_last_mean']:.2f}")
    lines.append("")
    lines.append("#### 9.6.2 对照 ctrl24h 概览")
    lines.append(f"- 分段数：{ctrl_stats['segments']}")
    lines.append(f"- EMA(last)均值：{ctrl_stats['ema_last_mean']:.6f}")
    lines.append(f"- Sparsity峰值（跨段最大）：{ctrl_stats['sparsity_peak_max']:.6f}")
    lines.append(f"- SVD(last)均值：{ctrl_stats['svd_last_mean']:.6f}")
    lines.append(f"- VRAM(last)均值(GB)：{ctrl_stats['vram_last_gb_mean']:.6f}")
    lines.append(f"- TPS(last)均值（StepPerSec×1024）：{ctrl_stats['tps_last_mean']:.2f}")
    lines.append("")
    lines.append("#### 9.6.3 严格对比结论")
    lines.append(f"- 状态：{comp.get('status', 'DATA_INCOMPLETE')}")
    lines.append(f"- 结论：{comp.get('final_conclusion', 'RETEST')}")
    if comp.get("status") == "OK":
        lines.append(f"- 主跑最终裁决：{comp.get('main_final_verdict')}")
        lines.append(f"- 对照最终裁决：{comp.get('ctrl_final_verdict')}")
        lines.append(f"- 主跑接受率：{comp.get('main_accept_rate')}")
        lines.append(f"- 对照接受率：{comp.get('ctrl_accept_rate')}")
    else:
        lines.append(f"- 原因：{comp.get('reason', 'insufficient evidence')}")
    lines.append("")
    lines.append("#### 9.6.4 产物索引")
    lines.append("- acceptance_rollup.json")
    lines.append("- acceptance_rollup.md")
    lines.append("- acceptance_main24h_seg*.json")
    lines.append("- acceptance_ctrl24h_seg*.json")
    lines.append("")
    return "\n".join(lines)


def upsert_auto_section(report_path: Path, section: str) -> None:
    start_marker = "<!-- AUTO_24H_SUMMARY_START -->"
    end_marker = "<!-- AUTO_24H_SUMMARY_END -->"

    original = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    block = f"{start_marker}\n{section}\n{end_marker}\n"

    if start_marker in original and end_marker in original:
        pre = original.split(start_marker)[0]
        post = original.split(end_marker)[-1]
        new_text = pre + block + post.lstrip("\n")
    else:
        sep = "\n" if original.endswith("\n") else "\n\n"
        new_text = original + sep + block

    report_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    rollup = load_rollup(root / "acceptance_rollup.json")

    main_stats = summarize_run(collect_run_telemetry(root, "main24h"))
    ctrl_stats = summarize_run(collect_run_telemetry(root, "ctrl24h"))

    section = render_auto_section(rollup, main_stats, ctrl_stats)
    upsert_auto_section(root / "FINAL_ANALYSIS_REPORT_CN.md", section)

    print("[ok] updated FINAL_ANALYSIS_REPORT_CN.md with AUTO_24H summary block")


if __name__ == "__main__":
    main()
