from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX video self-attention microbenchmark.")
    parser.add_argument(
        "--backend",
        choices=["fa3-bf16", "fa3-bf16-out", "fa3-bf16-qkvpacked", "fa3-bf16-qkvpacked-pack", "fa3-fp8"],
        default="fa3-bf16",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=34680)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--sm-margin", type=int, default=0)
    parser.add_argument("--attention-chunk", type=int, default=0)
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="Lab-only local-attention window. The default -1 keeps exact dense attention.",
    )
    parser.add_argument(
        "--window-right",
        type=int,
        default=-1,
        help="Lab-only local-attention window. The default -1 keeps exact dense attention.",
    )
    parser.add_argument("--fp8-scale", choices=["tensor", "head"], default="head")
    parser.add_argument("--fp8-quant-kernel", choices=["pytorch", "triton"], default="pytorch")
    parser.add_argument("--include-fp8-quant", action="store_true")
    parser.add_argument("--check-error", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    import flash_attn_interface

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    q = torch.randn(args.batch, args.tokens, args.heads, args.head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if args.backend == "fa3-bf16":
        call = lambda: flash_attn_interface.flash_attn_func(  # noqa: E731
            q,
            k,
            v,
            window_size=(args.window_left, args.window_right),
            num_splits=args.num_splits,
            sm_margin=args.sm_margin,
            attention_chunk=args.attention_chunk,
        )
        error: dict[str, float] | None = _error_against_bf16(flash_attn_interface, q, k, v, call) if args.check_error else None
    elif args.backend == "fa3-bf16-out":
        out = torch.empty_like(q)
        call = lambda: _fa3_bf16_preallocated_call(flash_attn_interface, q, k, v, out, args)  # noqa: E731
        error = _error_against_bf16(flash_attn_interface, q, k, v, call) if args.check_error else None
    elif args.backend == "fa3-bf16-qkvpacked":
        qkv = torch.stack((q, k, v), dim=2).contiguous()
        call = lambda: flash_attn_interface.flash_attn_qkvpacked_func(  # noqa: E731
            qkv,
            window_size=(args.window_left, args.window_right),
            sm_margin=args.sm_margin,
            attention_chunk=args.attention_chunk,
        )
        error = _error_against_bf16(flash_attn_interface, q, k, v, call) if args.check_error else None
    elif args.backend == "fa3-bf16-qkvpacked-pack":
        call = lambda: flash_attn_interface.flash_attn_qkvpacked_func(  # noqa: E731
            torch.stack((q, k, v), dim=2).contiguous(),
            window_size=(args.window_left, args.window_right),
            sm_margin=args.sm_margin,
            attention_chunk=args.attention_chunk,
        )
        error = _error_against_bf16(flash_attn_interface, q, k, v, call) if args.check_error else None
    else:
        if args.include_fp8_quant:
            call = lambda: _fp8_attention_call(flash_attn_interface, q, k, v, args)  # noqa: E731
        else:
            q8, q_descale = _to_fp8(q, args.fp8_scale)
            k8, k_descale = _to_fp8(k, args.fp8_scale)
            v8, v_descale = _to_fp8(v, args.fp8_scale)
            call = lambda: _fa3_call(  # noqa: E731
                flash_attn_interface,
                q8,
                k8,
                v8,
                args,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        error = _error_against_bf16(flash_attn_interface, q, k, v, call) if args.check_error else None

    times_ms = _measure(call, args.warmup, args.runs)
    output = call()
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "backend": args.backend,
        "batch": args.batch,
        "tokens": args.tokens,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": str(output.dtype),
        "shape": list(output.shape),
        "warmup": args.warmup,
        "runs": args.runs,
        "num_splits": args.num_splits,
        "sm_margin": args.sm_margin,
        "attention_chunk": args.attention_chunk,
        "window_left": args.window_left,
        "window_right": args.window_right,
        "fp8_scale": args.fp8_scale if args.backend == "fa3-fp8" else None,
        "fp8_quant_kernel": args.fp8_quant_kernel if args.backend == "fa3-fp8" else None,
        "include_fp8_quant": args.include_fp8_quant if args.backend == "fa3-fp8" else None,
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


def _to_fp8(x: torch.Tensor, scale_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if scale_mode == "tensor":
        scale = (x.float().abs().amax().clamp_min(1e-12) / fp8_max).reshape(1, 1)
        q = torch.clamp(x.float() / scale.reshape(1, 1, 1, 1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return q, scale.to(torch.float32).expand(x.shape[0], x.shape[2]).contiguous()
    if scale_mode == "head":
        scale = x.float().abs().amax(dim=(1, 3)).clamp_min(1e-12) / fp8_max
        q = torch.clamp(x.float() / scale[:, None, :, None], -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return q, scale.to(torch.float32).contiguous()
    raise ValueError(f"unknown scale mode: {scale_mode}")


def _to_fp8_with_kernel(x: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if args.fp8_quant_kernel == "triton":
        if args.fp8_scale != "head":
            raise ValueError("--fp8-quant-kernel triton currently supports only --fp8-scale head")
        from ltx_serve.triton_ltx_ops import fused_fp8_quantize_e4m3_per_head_4d

        return fused_fp8_quantize_e4m3_per_head_4d(x)
    return _to_fp8(x, args.fp8_scale)


def _fp8_attention_call(
    flash_attn_interface: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    q8, q_descale = _to_fp8_with_kernel(q, args)
    k8, k_descale = _to_fp8_with_kernel(k, args)
    v8, v_descale = _to_fp8_with_kernel(v, args)
    return _fa3_call(
        flash_attn_interface,
        q8,
        k8,
        v8,
        args,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


def _fa3_bf16_preallocated_call(
    flash_attn_interface: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    softmax_scale = q.shape[-1] ** -0.5
    y, _softmax_lse, *_rest = flash_attn_interface.flash_attn_3_gpu.fwd(
        q,
        k,
        v,
        None,
        None,
        None,
        out,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        softmax_scale,
        False,
        args.window_left,
        args.window_right,
        args.attention_chunk,
        0.0,
        True,
        None,
        args.num_splits,
        None,
        args.sm_margin,
    )
    return y


def _fa3_call(
    flash_attn_interface: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
    *,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> torch.Tensor:
    return flash_attn_interface.flash_attn_func(
        q,
        k,
        v,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=(args.window_left, args.window_right),
        num_splits=args.num_splits,
        sm_margin=args.sm_margin,
        attention_chunk=args.attention_chunk,
    )


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


def _error_against_bf16(
    flash_attn_interface: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fp8_call: object,
) -> dict[str, float]:
    ref = flash_attn_interface.flash_attn_func(q, k, v)
    out = fp8_call()
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
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
