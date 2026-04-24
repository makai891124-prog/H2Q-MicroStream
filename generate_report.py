import json
import os
import datetime
import torch

files = {
    "Conservative (Short)": "cuda_ext_protocol_result.conservative.short.json",
    "Conservative (Long)": "cuda_ext_protocol_result.conservative.long.json",
    "Aggressive (Short)": "cuda_ext_protocol_result.aggressive.short.json",
    "Aggressive (Long)": "cuda_ext_protocol_result.aggressive.long.json",
}

report_lines = []
report_lines.append("# EXPERIMENT REPORT")
report_lines.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Environment
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
report_lines.append("\n## Environment Summary")
report_lines.append(f"- **GPU**: {gpu_name}")
report_lines.append(f"- **PyTorch**: {torch.__version__}")
report_lines.append(f"- **CUDA**: {torch.version.cuda}")

for title, filename in files.items():
    if not os.path.exists(filename):
        report_lines.append(f"\n## {title}")
        report_lines.append("Result file not found.")
        continue
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    report_lines.append(f"\n## {title}")
    report_lines.append("| Property | Value |")
    report_lines.append("| :--- | :--- |")
    report_lines.append(f"| Overall Verdict | {data.get('verdict', 'N/A')} |")
    report_lines.append(f"| Compiled | {data.get('compiled', 'N/A')} |")
    
    report_lines.append("\n### Per-Budget Metrics")
    report_lines.append("| Steps | Verdict | Loss Delta | TPS Ratio | P99 Ratio | VRAM Delta (MB) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    budgets = data.get('budgets', [])
    for b in budgets:
        steps = b.get('steps', 'N/A')
        verdict = b.get('verdict', 'N/A')
        loss_delta = f"{b.get('loss_delta', 0):.6f}"
        tps_ratio = f"{b.get('tps_ratio', 0):.2f}"
        p99_ratio = f"{b.get('p99_ratio', 0):.2f}"
        vram_delta = f"{b.get('vram_delta_mb', 0):.2f}"
        report_lines.append(f"| {steps} | {verdict} | {loss_delta} | {tps_ratio} | {p99_ratio} | {vram_delta} |")

report_lines.append("\n## Final Recommendation")
report_lines.append("建议在生产环境默认启用 `conservative` 模式。该模式在保持数值精度的同时，通过在长序列评估阶段引入 CUDA 扩展，有效提升了吞吐效率。对于极低延迟要求的场景，可考虑 `aggressive` 模式。")

with open("EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(os.path.abspath("EXPERIMENT_REPORT.md"))
