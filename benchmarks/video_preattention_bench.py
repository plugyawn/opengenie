from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact LTX video self-attention preattention benchmark.")
    parser.add_argument("--backend", choices=["torch", "triton", "triton-separate"], default="triton")
    parser.add_argument("--tokens", type=int, default=34680)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check-error", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dim = args.heads * args.head_dim
    q = torch.randn(1, args.tokens, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    cos, sin = _make_rope(args.tokens, args.heads, args.head_dim, device)
    eps = 1e-6

    if args.backend == "torch":
        call = lambda: _torch_preattention(q, k, q_weight, k_weight, eps, cos, sin)  # noqa: E731
    elif args.backend == "triton":
        from ltx_serve.triton_ltx_ops import fused_video_qk_rmsnorm_rope

        call = lambda: fused_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, eps, cos, sin)  # noqa: E731
    else:
        from ltx_serve.triton_ltx_ops import fused_video_qk_rmsnorm_rope_separate

        call = lambda: fused_video_qk_rmsnorm_rope_separate(q, k, q_weight, k_weight, eps, cos, sin)  # noqa: E731

    error = _error_against_torch(q, k, q_weight, k_weight, eps, cos, sin, call) if args.check_error else None
    times_ms = _measure(call, args.warmup, args.runs)
    q_out, k_out = call()
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "backend": args.backend,
        "tokens": args.tokens,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": str(q_out.dtype),
        "q_shape": list(q_out.shape),
        "k_shape": list(k_out.shape),
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_ms": {
            "min": min(times_ms),
            "p50": _percentile(times_ms, 0.50),
            "p90": _percentile(times_ms, 0.90),
            "max": max(times_ms),
            "mean": sum(times_ms) / len(times_ms),
            "all": times_ms,
        },
        "error": error,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def _make_rope(tokens: int, heads: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Representative split-RoPE cos/sin tensor with the same shape and dtype as LTX.
    freqs = torch.randn(1, heads, tokens, head_dim // 2, device=device, dtype=torch.float32)
    return freqs.cos().to(torch.bfloat16).contiguous(), freqs.sin().to(torch.bfloat16).contiguous()


def _torch_preattention(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.nn.functional.rms_norm(q, (q.shape[-1],), weight=q_weight, eps=eps)
    k = torch.nn.functional.rms_norm(k, (k.shape[-1],), weight=k_weight, eps=eps)
    return _apply_split_rope(q, cos, sin), _apply_split_rope(k, cos, sin)


def _apply_split_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    b, tokens, dim = x.shape
    heads = cos.shape[1]
    head_dim = dim // heads
    half = head_dim // 2
    xh = x.reshape(b, tokens, heads, head_dim).transpose(1, 2)
    first = xh[..., :half]
    second = xh[..., half:]
    out_first = first * cos - second * sin
    out_second = second * cos + first * sin
    return torch.cat([out_first, out_second], dim=-1).transpose(1, 2).reshape(b, tokens, dim)


def _measure(call: object, warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        _ = call()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = call()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _error_against_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    call: object,
) -> dict[str, float]:
    ref_q, ref_k = _torch_preattention(q, k, q_weight, k_weight, eps, cos, sin)
    out_q, out_k = call()
    torch.cuda.synchronize()
    diff_q = (out_q.float() - ref_q.float()).abs()
    diff_k = (out_k.float() - ref_k.float()).abs()
    diff = torch.maximum(diff_q, diff_k)
    denom = torch.maximum(ref_q.float().abs(), ref_k.float().abs()).clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
        "mean_rel": float((diff / denom).mean().item()),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


if __name__ == "__main__":
    main()
