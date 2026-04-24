"""Validation script for StereographicAttentionLayer."""

import torch

from sta_core_v2 import StereographicAttentionLayer


def _naive_routing(codes: torch.Tensor, causal: bool = False) -> torch.Tensor:
    num_planes = codes.size(-1)
    xor_bits = torch.bitwise_xor(codes.unsqueeze(2), codes.unsqueeze(1))
    mismatch = xor_bits.sum(dim=-1, dtype=torch.int32)
    routing = (num_planes - mismatch).to(dtype=torch.float32) / float(num_planes)

    if causal:
        seq_len = codes.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=codes.device, dtype=torch.bool))
        routing = routing.masked_fill(~mask.unsqueeze(0), 0.0)

    routing = routing / routing.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return routing


def small_correctness_check() -> None:
    torch.manual_seed(7)
    layer = StereographicAttentionLayer(
        hidden_dim=32,
        num_planes=16,
        chunk_size=5,
        causal=True,
        routing_mode="normalize",
    )
    x = torch.randn(2, 16, 32)

    with torch.no_grad():
        _, codes = layer.encode_topology(x)
        routing_chunked = layer._chunked_similarity(codes, x.dtype)
        routing_naive = _naive_routing(codes, causal=True)

    max_diff = (routing_chunked - routing_naive).abs().max().item()
    if max_diff > 1e-6:
        raise AssertionError(f"chunked routing mismatch: max_diff={max_diff}")
    print(f"[check] routing correctness OK, max_diff={max_diff:.6e}")


def evolution_check() -> None:
    torch.manual_seed(11)
    layer = StereographicAttentionLayer(
        hidden_dim=32,
        num_planes=16,
        chunk_size=4,
        causal=False,
        routing_mode="normalize",
        evolution_noise_std=0.0,
    )
    layer.train()

    x = torch.randn(2, 8, 32) * 0.01
    x[..., 0] = 5.0

    before = layer.addressing_planes.clone()
    _ = layer(x)
    after = layer.addressing_planes

    if layer.last_invalid_plane_count <= 0:
        raise AssertionError("expected at least one invalid plane during evolution check")
    if torch.equal(before, after):
        raise AssertionError("addressing planes did not evolve in training mode")

    layer.eval()
    frozen_before = layer.addressing_planes.clone()
    _ = layer(x)
    if not torch.equal(frozen_before, layer.addressing_planes):
        raise AssertionError("addressing planes changed in eval mode")

    print(
        f"[check] evolution OK, invalid_planes={layer.last_invalid_plane_count}, "
        f"row_sum_mean={layer.last_routing_row_sum_mean:.6f}"
    )


def cuda_dummy_forward(batch: int = 4, seq_len: int = 1024, dim: int = 768, planes: int = 128) -> None:
    if not torch.cuda.is_available():
        print("[check] CUDA unavailable, skipped large dummy forward")
        return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(23)

    layer = StereographicAttentionLayer(
        hidden_dim=dim,
        num_planes=planes,
        chunk_size=64,
        causal=True,
        routing_mode="normalize",
    ).to(device)
    layer.eval()

    x = torch.randn(batch, seq_len, dim, device=device)
    with torch.inference_mode():
        out = layer(x)
    torch.cuda.synchronize(device)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(
        f"[check] cuda dummy forward OK, input=({batch}, {seq_len}, {dim}), "
        f"planes={planes}, output={tuple(out.shape)}, peak_vram_mb={peak_mb:.2f}, "
        f"routing_row_sum_mean={layer.last_routing_row_sum_mean:.6f}"
    )


def cuda_stress_2048() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(29)

    layer = StereographicAttentionLayer(
        hidden_dim=768,
        num_planes=128,
        chunk_size=64,
        causal=True,
        routing_mode="normalize",
    ).to(device)
    layer.eval()

    x = torch.randn(1, 2048, 768, device=device)
    with torch.inference_mode():
        out = layer(x)
    torch.cuda.synchronize(device)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(
        f"[check] cuda 2048 stress OK, output={tuple(out.shape)}, peak_vram_mb={peak_mb:.2f}"
    )


if __name__ == "__main__":
    small_correctness_check()
    evolution_check()
    cuda_dummy_forward()
    cuda_stress_2048()
    print("=== StereographicAttentionLayer validation passed ===")