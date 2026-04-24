"""Training-level comparison between sta_v2 and binary_sta variants."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from h2q_evolution import H2Q_Evolution_Engine

WORKDIR = Path(__file__).resolve().parent
JSON_REPORT = WORKDIR / "train_compare_sta_variants.json"
MD_REPORT = WORKDIR / "TRAIN_COMPARE_STA_VARIANTS.md"


def char_stats(text: str) -> dict:
    if not text:
        return {
            "invalid_char_rate": 1.0,
            "repeat_bigram_rate": 1.0,
            "max_run": 0,
            "readability_score": 0.0,
        }

    invalid = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    invalid_rate = invalid / len(text)
    bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1))]
    repeat_bigram_rate = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    run = 1
    max_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    score = (
        (1.0 - min(invalid_rate, 0.05) / 0.05) * 0.35
        + (1.0 - min(repeat_bigram_rate, 0.35) / 0.35) * 0.35
        + (1.0 - min(max_run, 20) / 20.0) * 0.20
        + (0.10 if " " in text else 0.0)
    )
    return {
        "invalid_char_rate": float(invalid_rate),
        "repeat_bigram_rate": float(repeat_bigram_rate),
        "max_run": int(max_run),
        "readability_score": float(max(0.0, min(1.0, score))),
    }


def load_corpus(path: Path, min_bytes: int = 1 << 20) -> bytes:
    if path.exists():
        data = path.read_bytes()
        if len(data) >= 4096:
            return data
    # fallback synthetic corpus
    base = (b"The quick brown fox jumps over the lazy dog. " * 2048)
    while len(base) < min_bytes:
        base = base + base
    return base[:min_bytes]


def sample_batch(data: bytes, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    import random

    max_start = len(data) - (seq_len + 1)
    if max_start <= 0:
        raise RuntimeError("corpus too short for seq_len")

    xs = []
    ys = []
    for _ in range(batch_size):
        start = random.randint(0, max_start)
        chunk = data[start : start + seq_len + 1]
        x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(list(chunk[1:]), dtype=torch.long)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def run_variant(
    name: str,
    attention_type: str,
    data: bytes,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    dim: int,
    layers: int,
    lr: float,
) -> dict:
    torch.manual_seed(42)
    model = H2Q_Evolution_Engine(
        dim=dim,
        num_layers=layers,
        rank=8,
        max_seq_len=seq_len,
        attention_type=attention_type,
        binary_num_planes=128,
        binary_chunk_size=64,
        binary_routing_mode="normalize",
        binary_backend="packbits",
        binary_fused_chunk_compute=True,
    ).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = sample_batch(data, batch_size, seq_len, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    tokens = steps * batch_size * seq_len
    tps = tokens / max(elapsed, 1e-6)
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    model.eval()
    prompt = torch.tensor([[84, 104, 101, 32]], dtype=torch.long, device=device)  # "The "
    with torch.no_grad():
        out = model.generate(prompt, new_bytes=64)
    text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    quality = char_stats(text)

    return {
        "variant": name,
        "attention_type": attention_type,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dim": dim,
        "layers": layers,
        "avg_loss": sum(losses) / len(losses),
        "last_loss": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "tokens_per_sec": tps,
        "peak_vram_mb": peak_vram_mb,
        "topology_sparsity": float(model.get_topology_sparsity()),
        "generation_text": text,
        "generation_quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-level sta_v2 vs binary_sta comparison")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus", type=str, default="data/open_corpus/open_corpus.txt")
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    corpus = load_corpus(WORKDIR / args.corpus)

    sta_v2 = run_variant(
        "sta_v2",
        "sta_v2",
        corpus,
        device,
        args.steps,
        args.batch_size,
        args.seq_len,
        args.dim,
        args.layers,
        args.lr,
    )
    binary_sta = run_variant(
        "binary_sta_packbits",
        "binary_sta",
        corpus,
        device,
        args.steps,
        args.batch_size,
        args.seq_len,
        args.dim,
        args.layers,
        args.lr,
    )

    result = {
        "device": str(device),
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dim": args.dim,
            "layers": args.layers,
            "lr": args.lr,
            "corpus": args.corpus,
        },
        "sta_v2": sta_v2,
        "binary_sta_packbits": binary_sta,
        "delta": {
            "loss_last_binary_minus_sta": binary_sta["last_loss"] - sta_v2["last_loss"],
            "tokens_per_sec_binary_over_sta": binary_sta["tokens_per_sec"] / max(sta_v2["tokens_per_sec"], 1e-6),
            "peak_vram_mb_binary_minus_sta": binary_sta["peak_vram_mb"] - sta_v2["peak_vram_mb"],
            "readability_binary_minus_sta": binary_sta["generation_quality"]["readability_score"]
            - sta_v2["generation_quality"]["readability_score"],
        },
    }

    JSON_REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Training Compare: sta_v2 vs binary_sta(packbits)",
        "",
        f"- device: {result['device']}",
        f"- config: {result['config']}",
        "",
        "## sta_v2",
        f"- avg_loss: {sta_v2['avg_loss']:.6f}",
        f"- last_loss: {sta_v2['last_loss']:.6f}",
        f"- tokens_per_sec: {sta_v2['tokens_per_sec']:.2f}",
        f"- peak_vram_mb: {sta_v2['peak_vram_mb']:.2f}",
        f"- readability: {sta_v2['generation_quality']['readability_score']:.4f}",
        "",
        "## binary_sta(packbits)",
        f"- avg_loss: {binary_sta['avg_loss']:.6f}",
        f"- last_loss: {binary_sta['last_loss']:.6f}",
        f"- tokens_per_sec: {binary_sta['tokens_per_sec']:.2f}",
        f"- peak_vram_mb: {binary_sta['peak_vram_mb']:.2f}",
        f"- readability: {binary_sta['generation_quality']['readability_score']:.4f}",
        "",
        "## Delta",
        f"- loss_last_binary_minus_sta: {result['delta']['loss_last_binary_minus_sta']:+.6f}",
        f"- tokens_per_sec_binary_over_sta: {result['delta']['tokens_per_sec_binary_over_sta']:.4f}x",
        f"- peak_vram_mb_binary_minus_sta: {result['delta']['peak_vram_mb_binary_minus_sta']:+.2f}",
        f"- readability_binary_minus_sta: {result['delta']['readability_binary_minus_sta']:+.4f}",
        "",
        "## Notes",
        "- This is a short controlled training experiment for directional comparison.",
        "- For final model decision, rerun with longer steps and fixed seeds over multiple trials.",
    ]
    MD_REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[report] json={JSON_REPORT.name}")
    print(f"[report] md={MD_REPORT.name}")


if __name__ == "__main__":
    main()
