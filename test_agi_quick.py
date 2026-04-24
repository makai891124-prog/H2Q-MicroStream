"""Quick smoke-test for agi_joint_trainer.py"""
import sys, math, torch, types

sys.path.insert(0, ".")
sys.path.insert(0, "H2Q-Single")

# 用 importlib 加载模块，阻止触发 __main__ 入口
import importlib.util
spec = importlib.util.spec_from_file_location("agi_joint_trainer", "agi_joint_trainer.py")
_mod = importlib.util.module_from_spec(spec)
_mod.__name__ = "agi_joint_trainer"   # 非 __main__，阻止 main() 执行
spec.loader.exec_module(_mod)
AGI_Accelerated_Transformer = _mod.AGI_Accelerated_Transformer
AccelTelemetry = _mod.AccelTelemetry

CFG_SMALL = {
    "dim": 128, "factor_size": 32, "fixed_rank": 8,
    "depth": 4, "seq_len": 32, "batch_size": 2,
    "dropout_rate": 0.1, "axiom_lambda": 0.1,
    "shockwave_threshold": math.pi / 2,
    "hash_dim": 32, "num_buckets": 4, "hamming_thresh": 4,
    "lr": 3e-4, "weight_decay": 0.02, "grad_clip": 1.0,
    "total_chunks": 10, "chunk_size_mb": 1,
    "source_dir": r"E:\Datasets\FineWeb-Edu_Full",
    "buffer_dir": r"D:\H2Q_Cache_Zone",
    "checkpoint_path": "agi_joint_test.pt",
    "best_model_path": "agi_joint_test_best.pt",
    "telemetry_csv": "agi_joint_test_telemetry.csv",
    "supervise_every": 0, "supervise_gen_tokens": 64,
}

dev = torch.device("cuda:0")
model = AGI_Accelerated_Transformer(CFG_SMALL).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")

# Forward pass
x = torch.randint(0, 256, (2, 32), device=dev)
y = torch.randint(0, 256, (2, 32), device=dev)
logits, loss = model(x, y)
print(f"logits: {logits.shape}  loss: {loss.item():.4f}")

# Backward pass
loss.backward()
print("backward OK")

# Accel stats
s = model.accel_stats()
print(f"STA sparsity: {s['sta_sparsity_mean']*100:.1f}%")
print(f"TCRH connectivity: {s['tcrh_conn_mean']*100:.1f}%")
print(f"ortho_loss: {s['ortho_loss']:.4f}")

# Generate
with torch.no_grad():
    ctx = torch.zeros(1, 1, dtype=torch.long, device=dev)
    out = model.generate(ctx, 10)
    print(f"generate shape: {out.shape}")

print("=== ALL CHECKS PASSED ===")
