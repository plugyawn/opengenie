from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ltx_serve.buckets import BUCKETS
from ltx_serve.schemas import BenchmarkResult


DEFAULT_PROMPT = (
    "A vertical cinematic reel of neon reflections on rain-soaked pavement, "
    "slow push-in, realistic motion, synchronized ambient city audio."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LTX2 FastVideo benchmark and emit JSON.")
    parser.add_argument("--bucket", default="fast_5s_v1", choices=sorted(BUCKETS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", default=os.getenv("LTX2_MODEL_PATH", "FastVideo/LTX2-Distilled-Diffusers"))
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--attention-backend", default=os.getenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA"))
    parser.add_argument("--quant", choices=["none", "absmax-fp8", "nvfp4"], default="none")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bucket = BUCKETS[args.bucket]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports are intentionally inside main so local non-GPU tests do not need
    # FastVideo installed.
    import torch
    import torch._inductor.config
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.utils import maybe_download_model

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

    model_root = maybe_download_model(args.model_path)
    pipeline_config = PipelineConfig.from_pretrained(model_root)
    if args.quant == "absmax-fp8":
        from fastvideo.layers.quantization.absmax_fp8 import AbsMaxFP8Config

        pipeline_config.dit_config.quant_config = AbsMaxFP8Config()
    elif args.quant == "nvfp4":
        from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

        pipeline_config.dit_config.quant_config = NVFP4Config()

    torch_compile_kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
    }
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        pipeline_config=pipeline_config,
        enable_torch_compile=args.compile,
        enable_torch_compile_text_encoder=args.compile,
        enable_torch_compile_vae=args.compile,
        torch_compile_kwargs=torch_compile_kwargs,
        torch_compile_kwargs_vae=torch_compile_kwargs,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    measured: list[BenchmarkResult] = []
    try:
        total_runs = args.warmup + args.runs
        for idx in range(total_runs):
            output_path = output_dir / f"{args.bucket}_run_{idx + 1}.mp4"
            if output_path.exists():
                output_path.unlink()
            start = time.perf_counter()
            result = generator.generate_video(
                prompt=args.prompt,
                output_path=str(output_path),
                fps=bucket.fps,
                seed=10,
                save_video=args.save_video,
                guidance_scale=args.guidance_scale,
                height=bucket.height,
                width=bucket.width,
                num_frames=bucket.frames,
                num_inference_steps=args.steps,
            )
            e2e_runtime_s = time.perf_counter() - start
            runtime_s = float(result.get("generation_time", e2e_runtime_s)) if isinstance(result, dict) else e2e_runtime_s
            stage_times = _extract_stage_times(result)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_name = torch.cuda.get_device_name(0)
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                gpu_name = None
                peak_vram_gb = None

            bench = BenchmarkResult(
                backend=f"fastvideo:{args.quant}:compile={args.compile}:attn={args.attention_backend}",
                bucket=bucket.name,
                prompt=args.prompt,
                output_path=str(output_path) if args.save_video else None,
                runtime_s=runtime_s,
                e2e_runtime_s=e2e_runtime_s,
                target_duration_s=bucket.duration_s,
                realtime_factor=e2e_runtime_s / bucket.duration_s,
                faster_than_realtime=e2e_runtime_s < bucket.duration_s,
                gpu_name=gpu_name,
                peak_vram_gb=peak_vram_gb,
                stage_times=stage_times,
            )
            print(bench.model_dump_json())
            if idx >= args.warmup:
                measured.append(bench)
    finally:
        generator.shutdown()

    summary_path = output_dir / f"{args.bucket}_summary.json"
    summary = {
        "bucket": bucket.name,
        "runs": [m.model_dump() for m in measured],
        "best_e2e_runtime_s": min((m.e2e_runtime_s for m in measured), default=None),
        "best_realtime_factor": min((m.realtime_factor for m in measured), default=None),
        "faster_than_realtime": any(m.faster_than_realtime for m in measured),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2))


def _extract_stage_times(result: object) -> dict[str, float]:
    if not isinstance(result, dict):
        return {}
    logging_info = result.get("logging_info")
    stages = getattr(logging_info, "stages", None)
    if not stages:
        return {}
    return {str(name): float(metrics.get("execution_time", 0.0)) for name, metrics in stages.items()}


if __name__ == "__main__":
    main()
