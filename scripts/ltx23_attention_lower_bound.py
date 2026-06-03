from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-2.3 dense video self-attention FLOP lower bound.")
    parser.add_argument("--tokens", type=int, default=34_680)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--measured-attention-s", type=float, default=16.346)
    parser.add_argument(
        "--h100-bf16-tc-peak-tflops",
        type=float,
        default=756.0,
        help="Optimistic dense BF16/FP16 tensor-core peak for H100 PCIe.",
    )
    parser.add_argument("--target-total-s", type=float, default=5.0)
    parser.add_argument(
        "--target-attention-budget-s",
        type=float,
        default=1.58,
        help="Illustrative attention budget if decode/encode/projections keep the rest of the 5s request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flops_per_call = 4 * args.tokens * args.tokens * args.width
    calls = args.layers * args.steps
    total_flops = flops_per_call * calls
    result = {
        "tokens": args.tokens,
        "width": args.width,
        "layers": args.layers,
        "steps": args.steps,
        "video_self_attention_calls": calls,
        "flops_per_call_tflop": flops_per_call / 1e12,
        "total_video_self_attention_pflop": total_flops / 1e15,
        "measured_attention_s": args.measured_attention_s,
        "measured_effective_tflops": total_flops / args.measured_attention_s / 1e12,
        "h100_bf16_tc_peak_tflops": args.h100_bf16_tc_peak_tflops,
        "optimistic_peak_lower_bound_s": total_flops / (args.h100_bf16_tc_peak_tflops * 1e12),
        "target_total_s": args.target_total_s,
        "required_tflops_if_attention_used_entire_target": total_flops / args.target_total_s / 1e12,
        "target_attention_budget_s": args.target_attention_budget_s,
        "required_tflops_for_attention_budget": total_flops / args.target_attention_budget_s / 1e12,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
