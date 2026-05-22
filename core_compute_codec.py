from __future__ import annotations

import csv
import os
from typing import Dict

ORTHO_LOSS_SCALE = 0.01


def estimate_ce_from_total_loss(train_loss: float, ortho_loss: float, axiom_lambda: float) -> float:
    """Estimate CE term from total loss = CE + axiom_lambda * ortho * 0.01."""
    return float(train_loss) - float(axiom_lambda) * float(ortho_loss) * ORTHO_LOSS_SCALE


def compute_core_metrics(
    *,
    train_loss: float,
    val_loss: float,
    ortho_loss: float,
    tokens_per_sec: float,
    axiom_lambda: float,
    stats: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Compute unified derived metrics for trainer telemetry sidecar."""
    ce_estimate = estimate_ce_from_total_loss(train_loss, ortho_loss, axiom_lambda)
    reg_uplift = float(train_loss) - ce_estimate
    out: Dict[str, float] = {
        "ce_estimate": ce_estimate,
        "regularization_uplift": reg_uplift,
        "generalization_gap": float(val_loss) - float(train_loss),
        "throughput_ktok_s": float(tokens_per_sec) / 1000.0,
    }

    stats = stats or {}
    out.update(
        {
            "struct128_enabled_ratio": float(stats.get("struct128_enabled_ratio", 0.0)),
            "struct128_mean_sim_sem": float(stats.get("struct128_mean_sim_sem", 0.0)),
            "struct128_mean_sim_struct": float(stats.get("struct128_mean_sim_struct", 0.0)),
            "struct128_entropy_or_row_norm": float(stats.get("struct128_entropy_or_row_norm", 0.0)),
            "struct128_effective_k": float(stats.get("struct128_effective_k", 0.0)),
            "ungs_loss": float(stats.get("ungs_loss", 0.0)),
            "relation_density": float(stats.get("relation_density", 0.0)),
            "hierarchy_ratio": float(stats.get("hierarchy_ratio", 0.0)),
            "self_ref_consistency": float(stats.get("self_ref_consistency", 0.0)),
            "axiom_residual": float(stats.get("axiom_residual", 0.0)),
            "structural_pressure": float(stats.get("structural_pressure", 0.0)),
            "lr_dynamic": float(stats.get("lr_dynamic", 0.0)),
            "axiom_lambda_dynamic": float(stats.get("axiom_lambda_dynamic", 0.0)),
            "ungs_closure_lambda_dynamic": float(stats.get("ungs_closure_lambda_dynamic", 0.0)),
            "ungs_encapsulation_lambda_dynamic": float(stats.get("ungs_encapsulation_lambda_dynamic", 0.0)),
            "ungs_self_ref_lambda_dynamic": float(stats.get("ungs_self_ref_lambda_dynamic", 0.0)),
            "control_phase": float(stats.get("control_phase", 0.0)),
            "control_phase_scale": float(stats.get("control_phase_scale", 0.0)),
            "val_worse_streak": float(stats.get("val_worse_streak", 0.0)),
            "val_protection_active": float(stats.get("val_protection_active", 0.0)),
            "val_protection_triggered": float(stats.get("val_protection_triggered", 0.0)),
        }
    )
    return out


def to_core_telemetry_path(base_telemetry_path: str) -> str:
    root, ext = os.path.splitext(base_telemetry_path)
    if not ext:
        ext = ".csv"
    return f"{root}_core{ext}"


class CoreTelemetryCSV:
    FIELDS = [
        "timestamp",
        "chunk",
        "ce_estimate",
        "regularization_uplift",
        "generalization_gap",
        "throughput_ktok_s",
        "struct128_enabled_ratio",
        "struct128_mean_sim_sem",
        "struct128_mean_sim_struct",
        "struct128_entropy_or_row_norm",
        "struct128_effective_k",
        "ungs_loss",
        "relation_density",
        "hierarchy_ratio",
        "self_ref_consistency",
        "axiom_residual",
        "structural_pressure",
        "lr_dynamic",
        "axiom_lambda_dynamic",
        "ungs_closure_lambda_dynamic",
        "ungs_encapsulation_lambda_dynamic",
        "ungs_self_ref_lambda_dynamic",
        "control_phase",
        "control_phase_scale",
        "val_worse_streak",
        "val_protection_active",
        "val_protection_triggered",
    ]

    def __init__(self, path: str):
        self.path = path
        self._rows_since_flush = 0
        self._flush_every = max(1, int(os.environ.get("CORE_TELEMETRY_FLUSH_EVERY", "8")))
        exists = os.path.exists(path)
        self.fp = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDS)
        if not exists or os.path.getsize(path) == 0:
            self.writer.writeheader()
            self.fp.flush()

    def write(self, *, timestamp: str, chunk: int, metrics: Dict[str, float]) -> None:
        row = {
            "timestamp": timestamp,
            "chunk": int(chunk),
            "ce_estimate": f"{metrics.get('ce_estimate', 0.0):.6f}",
            "regularization_uplift": f"{metrics.get('regularization_uplift', 0.0):.6f}",
            "generalization_gap": f"{metrics.get('generalization_gap', 0.0):.6f}",
            "throughput_ktok_s": f"{metrics.get('throughput_ktok_s', 0.0):.3f}",
            "struct128_enabled_ratio": f"{metrics.get('struct128_enabled_ratio', 0.0):.4f}",
            "struct128_mean_sim_sem": f"{metrics.get('struct128_mean_sim_sem', 0.0):.6f}",
            "struct128_mean_sim_struct": f"{metrics.get('struct128_mean_sim_struct', 0.0):.6f}",
            "struct128_entropy_or_row_norm": f"{metrics.get('struct128_entropy_or_row_norm', 0.0):.6f}",
            "struct128_effective_k": f"{metrics.get('struct128_effective_k', 0.0):.6f}",
            "ungs_loss": f"{metrics.get('ungs_loss', 0.0):.6f}",
            "relation_density": f"{metrics.get('relation_density', 0.0):.6f}",
            "hierarchy_ratio": f"{metrics.get('hierarchy_ratio', 0.0):.6f}",
            "self_ref_consistency": f"{metrics.get('self_ref_consistency', 0.0):.6f}",
            "axiom_residual": f"{metrics.get('axiom_residual', 0.0):.6f}",
            "structural_pressure": f"{metrics.get('structural_pressure', 0.0):.6f}",
            "lr_dynamic": f"{metrics.get('lr_dynamic', 0.0):.8f}",
            "axiom_lambda_dynamic": f"{metrics.get('axiom_lambda_dynamic', 0.0):.6f}",
            "ungs_closure_lambda_dynamic": f"{metrics.get('ungs_closure_lambda_dynamic', 0.0):.6f}",
            "ungs_encapsulation_lambda_dynamic": f"{metrics.get('ungs_encapsulation_lambda_dynamic', 0.0):.6f}",
            "ungs_self_ref_lambda_dynamic": f"{metrics.get('ungs_self_ref_lambda_dynamic', 0.0):.6f}",
            "control_phase": f"{metrics.get('control_phase', 0.0):.0f}",
            "control_phase_scale": f"{metrics.get('control_phase_scale', 0.0):.6f}",
            "val_worse_streak": f"{metrics.get('val_worse_streak', 0.0):.0f}",
            "val_protection_active": f"{metrics.get('val_protection_active', 0.0):.0f}",
            "val_protection_triggered": f"{metrics.get('val_protection_triggered', 0.0):.0f}",
        }
        self.writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every:
            self.fp.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        try:
            if self._rows_since_flush > 0:
                self.fp.flush()
            self.fp.close()
        except Exception:
            pass
