from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from ltx_serve.buckets import BUCKETS


DEFAULT_PROMPT = (
    "A vertical cinematic reel of neon reflections on rain-soaked pavement, "
    "slow push-in, realistic motion, synchronized ambient city audio."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the source FastVideo LTX2 pipeline benchmark.")
    parser.add_argument("--model-path", default=os.getenv("FASTVIDEO_LTX2_MODEL", "FastVideo/LTX2-Distilled-Diffusers"))
    parser.add_argument("--bucket", default="premium_25fps_5s_v1", choices=sorted(BUCKETS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="outputs/fastvideo_source")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output-type", choices=["pil", "latent"], default="latent")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--attention-backend", default=os.getenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA"))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-vae", action="store_true")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

    bucket = BUCKETS[args.bucket]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch._inductor.config
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.utils import maybe_download_model

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    model_root = maybe_download_model(args.model_path)
    pipeline_config = PipelineConfig.from_pretrained(model_root)
    compile_kwargs = {"backend": "inductor", "fullgraph": True, "dynamic": False}
    load_start = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        pipeline_config=pipeline_config,
        output_type=args.output_type,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_layerwise_offload=False,
        enable_stage_verification=False,
        enable_torch_compile=args.torch_compile,
        enable_torch_compile_vae=args.torch_compile_vae,
        torch_compile_kwargs=compile_kwargs,
        torch_compile_kwargs_vae=compile_kwargs,
        ltx2_vae_tiling=args.vae_tiling,
    )
    load_s = time.perf_counter() - load_start

    measured: list[dict[str, Any]] = []
    total_runs = args.warmup + args.runs
    try:
        for idx in range(total_runs):
            output_path = output_dir / f"{args.bucket}_{args.output_type}_run_{idx + 1}.mp4"
            if output_path.exists():
                output_path.unlink()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.perf_counter()
            result = generator.generate_video(
                prompt=args.prompt,
                output_path=str(output_path),
                save_video=args.save_video,
                return_frames=False,
                fps=bucket.final_fps,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                height=bucket.height,
                width=bucket.width,
                num_frames=bucket.frames,
                num_inference_steps=args.steps,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_s = time.perf_counter() - start

            stages = {}
            logging_info = result.get("logging_info") if isinstance(result, dict) else None
            if logging_info is not None and getattr(logging_info, "stages", None):
                stages = {
                    name: {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
                    for name, metrics in logging_info.stages.items()
                }

            runtime_s = float(result.get("e2e_latency") or wall_s) if isinstance(result, dict) else wall_s
            row = {
                "backend": f"fastvideo-source:{model_root}:attention={args.attention_backend}:output={args.output_type}:compile={args.torch_compile}:vae_compile={args.torch_compile_vae}",
                "bucket": bucket.name,
                "output_path": result.get("video_path") if isinstance(result, dict) else None,
                "wall_s": wall_s,
                "generation_time": result.get("generation_time") if isinstance(result, dict) else None,
                "runtime_s": runtime_s,
                "target_duration_s": bucket.output_duration_s,
                "realtime_factor": runtime_s / bucket.output_duration_s,
                "faster_than_realtime": runtime_s < bucket.output_duration_s,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
                "load_s": load_s,
                "stages": stages,
            }
            print(json.dumps(row, indent=2, default=str))
            if idx >= args.warmup:
                measured.append(row)
    finally:
        generator.shutdown()

    summary = {
        "model_path": args.model_path,
        "bucket": bucket.name,
        "runs": measured,
        "best_runtime_s": min((float(x["runtime_s"]) for x in measured), default=None),
        "best_realtime_factor": min((float(x["realtime_factor"]) for x in measured), default=None),
        "faster_than_realtime": any(bool(x["faster_than_realtime"]) for x in measured),
    }
    summary_path = output_dir / f"{args.bucket}_{args.output_type}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
