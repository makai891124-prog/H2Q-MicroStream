from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ACCEPTANCE_RE = re.compile(r"^acceptance_(?P<run>.+?)_seg(?P<seg>\d+)(?:_v(?P<v>\d+))?\.json$")


@dataclass
class SegmentResult:
    run: str
    seg: int
    version: int
    file: str
    verdict: str
    gates: Dict[str, dict]


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_best_file(files: List[Tuple[Path, int]]) -> Path:
    # highest version wins; if tie, longest stem (usually more specific), then latest mtime
    files = sorted(files, key=lambda x: (x[1], len(x[0].stem), x[0].stat().st_mtime), reverse=True)
    return files[0][0]


def collect_segments(root: Path) -> List[SegmentResult]:
    grouped: Dict[Tuple[str, int], List[Tuple[Path, int]]] = {}
    for p in root.glob("acceptance_*.json"):
        m = ACCEPTANCE_RE.match(p.name)
        if not m:
            continue
        run = m.group("run")
        seg = int(m.group("seg"))
        v = int(m.group("v") or 0)
        grouped.setdefault((run, seg), []).append((p, v))

    out: List[SegmentResult] = []
    for (run, seg), cand in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        chosen = _pick_best_file(cand)
        data = _safe_load_json(chosen)
        if not data:
            continue
        out.append(
            SegmentResult(
                run=run,
                seg=seg,
                version=max(v for _, v in cand),
                file=chosen.name,
                verdict=str(data.get("verdict", "UNKNOWN")),
                gates=data.get("gates", {}),
            )
        )
    return out


def aggregate_run_verdict(verdicts: List[str]) -> str:
    # strict priority: REJECT > RETEST > CONDITIONAL_ACCEPT > ACCEPT
    if not verdicts:
        return "RETEST"
    s = set(verdicts)
    if "REJECT" in s:
        return "REJECT"
    if "RETEST" in s:
        return "RETEST"
    if "CONDITIONAL_ACCEPT" in s:
        return "CONDITIONAL_ACCEPT"
    if s == {"ACCEPT"}:
        return "ACCEPT"
    return "RETEST"


def aggregate_all(segments: List[SegmentResult]) -> dict:
    runs: Dict[str, dict] = {}
    for seg in segments:
        r = runs.setdefault(
            seg.run,
            {
                "segment_count": 0,
                "segments": [],
                "verdicts": [],
                "gate_pass_counts": {"A": 0, "B": 0, "C": 0, "D": 0},
            },
        )
        r["segment_count"] += 1
        r["verdicts"].append(seg.verdict)

        gate_pass = {}
        for g in ["A", "B", "C", "D"]:
            passed = bool(seg.gates.get(g, {}).get("pass", False))
            gate_pass[g] = passed
            if passed:
                r["gate_pass_counts"][g] += 1

        r["segments"].append(
            {
                "seg": seg.seg,
                "file": seg.file,
                "verdict": seg.verdict,
                "gate_pass": gate_pass,
                "ema_last": seg.gates.get("A", {}).get("ema_last"),
                "sparsity_peak": seg.gates.get("C", {}).get("sparsity_peak"),
                "support_rate": seg.gates.get("D", {}).get("support_rate"),
            }
        )

    for run, info in runs.items():
        info["final_verdict"] = aggregate_run_verdict(info.pop("verdicts"))
        info["segments"] = sorted(info["segments"], key=lambda x: x["seg"])

    available_finals = [v["final_verdict"] for v in runs.values()]
    overall_verdict = aggregate_run_verdict(available_finals)

    comparison = strict_compare_main_vs_ctrl(runs)

    return {
        "runs": runs,
        "overall_verdict": overall_verdict,
        "main_vs_ctrl": comparison,
    }


def strict_compare_main_vs_ctrl(runs: Dict[str, dict]) -> dict:
    main = runs.get("main24h")
    ctrl = runs.get("ctrl24h")

    if not main or not ctrl:
        return {
            "status": "DATA_INCOMPLETE",
            "final_conclusion": "RETEST",
            "reason": "missing main24h and/or ctrl24h acceptance segments",
        }

    # strict compare by pass-rate and severity
    def pass_rate(run_info: dict) -> float:
        segs = run_info.get("segments", [])
        if not segs:
            return 0.0
        passed = sum(1 for x in segs if x.get("verdict") == "ACCEPT")
        return passed / len(segs)

    main_rate = pass_rate(main)
    ctrl_rate = pass_rate(ctrl)

    if main_rate > ctrl_rate and main.get("final_verdict") in {"ACCEPT", "CONDITIONAL_ACCEPT"}:
        concl = "MAIN_BETTER"
    elif ctrl_rate > main_rate and ctrl.get("final_verdict") in {"ACCEPT", "CONDITIONAL_ACCEPT"}:
        concl = "CTRL_BETTER"
    elif main.get("final_verdict") == ctrl.get("final_verdict"):
        concl = "NO_SIGNIFICANT_DIFFERENCE"
    else:
        concl = "RETEST"

    return {
        "status": "OK",
        "final_conclusion": concl,
        "main_final_verdict": main.get("final_verdict"),
        "ctrl_final_verdict": ctrl.get("final_verdict"),
        "main_accept_rate": main_rate,
        "ctrl_accept_rate": ctrl_rate,
    }


def main() -> None:
    root = Path.cwd()
    segments = collect_segments(root)
    summary = aggregate_all(segments)

    out_json = root / "acceptance_rollup.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Acceptance Rollup")
    lines.append("")
    lines.append(f"- Overall verdict: **{summary['overall_verdict']}**")
    lines.append("")

    runs = summary.get("runs", {})
    if not runs:
        lines.append("- No segmented acceptance files found.")
    else:
        for run_name, info in sorted(runs.items()):
            lines.append(f"## Run: {run_name}")
            lines.append(f"- final verdict: **{info['final_verdict']}**")
            lines.append(f"- segment count: {info['segment_count']}")
            lines.append(f"- gate pass counts: {info['gate_pass_counts']}")
            lines.append("")
            for seg in info.get("segments", []):
                lines.append(
                    f"- seg{seg['seg']}: verdict={seg['verdict']}, "
                    f"gate_pass={seg['gate_pass']}, file={seg['file']}"
                )
            lines.append("")

    comp = summary.get("main_vs_ctrl", {})
    lines.append("## Main vs Ctrl (strict)")
    lines.append(f"- status: {comp.get('status')}")
    lines.append(f"- final conclusion: **{comp.get('final_conclusion')}**")
    if comp.get("status") == "OK":
        lines.append(f"- main_final_verdict: {comp.get('main_final_verdict')}")
        lines.append(f"- ctrl_final_verdict: {comp.get('ctrl_final_verdict')}")
        lines.append(f"- main_accept_rate: {comp.get('main_accept_rate')}")
        lines.append(f"- ctrl_accept_rate: {comp.get('ctrl_accept_rate')}")
    else:
        lines.append(f"- reason: {comp.get('reason')}")

    out_md = root / "acceptance_rollup.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_json.name}")
    print(f"[ok] wrote {out_md.name}")


if __name__ == "__main__":
    main()
