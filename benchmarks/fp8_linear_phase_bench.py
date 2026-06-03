from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP8 linear phase benchmark for LTX H100 fusion triage.")
    parser.add_argument("--rows", type=int, default=22440)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    from ltx_serve.triton_ltx_ops import fused_fp8_quantize_e4m3

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = torch.randn((args.rows, args.in_features), device=device, dtype=torch.bfloat16)
    input_scale = torch.tensor(0.05, device=device, dtype=torch.float32)
    weight_scale = torch.tensor(0.05, device=device, dtype=torch.float32)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weight_bf16 = torch.randn((args.out_features, args.in_features), device=device, dtype=torch.bfloat16) * 0.05
    weight = torch.clamp(weight_bf16.float() / weight_scale, -fp8_max, fp8_max).to(torch.float8_e4m3fn)

    def quantize() -> torch.Tensor:
        return fused_fp8_quantize_e4m3(x, input_scale)

    qinput = quantize()
    torch.cuda.synchronize()

    def matmul_only() -> torch.Tensor:
        return torch._scaled_mm(
            qinput,
            weight.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    def quantize_plus_matmul() -> torch.Tensor:
        q = fused_fp8_quantize_e4m3(x, input_scale)
        return torch._scaled_mm(
            q,
            weight.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    quant_ms = _measure(quantize, args.warmup, args.runs)
    matmul_ms = _measure(matmul_only, args.warmup, args.runs)
    combined_ms = _measure(quantize_plus_matmul, args.warmup, args.runs)
    output = quantize_plus_matmul()
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "rows": args.rows,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "dtype": str(output.dtype),
        "warmup": args.warmup,
        "runs": args.runs,
        "quantize_ms": _summary(quant_ms),
        "matmul_only_ms": _summary(matmul_ms),
        "quantize_plus_matmul_ms": _summary(combined_ms),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def _measure(fn: object, warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(values)
    return {
        "min": min(values),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "max": max(values),
        "mean": sum(values) / len(values),
        "all": values,
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return sorted_values[idx]


if __name__ == "__main__":
    main()
