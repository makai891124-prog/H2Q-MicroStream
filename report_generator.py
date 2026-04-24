import json
import os

files = [
    "cuda_ext_protocol_result.conservative.short.json",
    "cuda_ext_protocol_result.conservative.long.json",
    "cuda_ext_protocol_result.aggressive.short.json",
    "cuda_ext_protocol_result.aggressive.long.json"
]

report_data = []
all_switch = True

for filename in files:
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            report_data.append((filename, data))
            if data.get("decision", {}).get("overall") != "SWITCH":
                all_switch = False
    else:
        print(f"File not found: {filename}")
        all_switch = False

if not report_data:
    print("No data found.")
    exit(1)

# Use the first file for common diagnosis and config info
first_file_data = report_data[0][1]
diagnosis = first_file_data.get("diagnosis", {})
steps = first_file_data.get("config", {}).get("steps", "N/A")
cuda_ext_compiled = first_file_data.get("cuda_ext_compiled", "N/A")

report_md = f"# EXPERIMENT_REPORT\n\n"
report_md += "## Diagnosis\n"
report_md += f"- **Device**: {diagnosis.get('device', 'N/A')}\n"
report_md += f"- **Torch Version**: {diagnosis.get('torch_version', 'N/A')}\n"
report_md += f"- **Torch CUDA Version**: {diagnosis.get('torch_cuda_version', 'N/A')}\n"
report_md += f"- **Steps**: {steps}\n"
report_md += f"- **CUDA Extension Compiled**: {cuda_ext_compiled}\n\n"

for filename, data in report_data:
    report_md += f"## File: {filename}\n"
    report_md += f"- **Overall Decision**: {data.get('decision', {}).get('overall', 'N/A')}\n\n"
    report_md += "| Steps | Verdict | Loss Delta | TPS Ratio | P99 Ratio | VRAM Delta (MB) |\n"
    report_md += "|-------|---------|------------|-----------|-----------|-----------------|\n"
    
    per_budget = data.get("decision", {}).get("per_budget", [])
    for row in per_budget:
        report_md += f"| {row.get('steps')} | {row.get('verdict')} | {row.get('loss_delta'):.6f} | {row.get('tps_ratio'):.4f} | {row.get('p99_ratio'):.4f} | {row.get('vram_delta_mb'):.2f} |\n"
    report_md += "\n"

if all_switch:
    recommendation = "建议默认切换至 cuda_ext (Recommend switch default to cuda_ext)"
else:
    recommendation = "建议保持默认 packbits 并使用 conservative inference-long 策略 (Recommend keep default packbits and use conservative inference-long policy)"

report_md += "## Final Recommendation\n"
report_md += f"{recommendation}\n"

with open("EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print(os.path.abspath("EXPERIMENT_REPORT.md"))
print(recommendation)
