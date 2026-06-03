from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from benchmarks.diffusers_ltx23_benchmark import DEFAULT_MODEL, DEFAULT_PROMPT, _save_output
from ltx_serve.buckets import BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile high-level LTX-2.3 Diffusers phases.")
    parser.add_argument("--bucket", default="premium_25fps_5s_v1", choices=sorted(BUCKETS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/profiles")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--stg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--output-type", choices=["pil", "latent"], default="pil")
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bucket = BUCKETS[args.bucket]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import DiffusionPipeline
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from diffusers.utils import export_to_video

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, device_map=args.device_map)
    if args.vae_tiling and hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
    if args.vae_slicing and hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
    load_s = time.perf_counter() - load_start

    timings: dict[str, list[float]] = defaultdict(list)
    _wrap_callable(pipe, "encode_prompt", "encode_prompt", timings, torch)
    _wrap_callable(pipe, "prepare_latents", "prepare_video_latents", timings, torch)
    _wrap_callable(pipe, "prepare_audio_latents", "prepare_audio_latents", timings, torch)
    _wrap_module(pipe, "connectors", "connectors", timings, torch)
    _wrap_module(pipe, "transformer", "transformer", timings, torch)
    _wrap_callable(getattr(pipe, "vae", None), "decode", "video_vae_decode", timings, torch)
    _wrap_callable(getattr(pipe, "audio_vae", None), "decode", "audio_vae_decode", timings, torch)
    _wrap_module(pipe, "vocoder", "vocoder", timings, torch)
    _wrap_callable(getattr(pipe, "video_processor", None), "postprocess_video", "video_postprocess", timings, torch)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)
    start = time.perf_counter()
    output = pipe(
        prompt=args.prompt,
        height=bucket.height,
        width=bucket.width,
        num_frames=bucket.frames,
        frame_rate=float(bucket.fps),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        stg_scale=args.stg_scale,
        generator=generator,
        output_type=args.output_type,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pipeline_s = time.perf_counter() - start

    output_path = None
    save_s = 0.0
    if args.save_video and args.output_type == "pil":
        output_path = output_dir / f"{args.bucket}_profile.mp4"
        save_start = time.perf_counter()
        _save_output(output, output_path, bucket, pipe, export_to_video, encode_video)
        save_s = time.perf_counter() - save_start

    phase_summary = {
        name: {
            "calls": len(values),
            "total_s": sum(values),
            "max_s": max(values),
            "mean_s": sum(values) / len(values),
        }
        for name, values in sorted(timings.items())
        if values
    }

    result = {
        "model_path": args.model_path,
        "bucket": bucket.name,
        "output_type": args.output_type,
        "output_path": str(output_path) if output_path else None,
        "target_output_duration_s": bucket.output_duration_s,
        "pipeline_s": pipeline_s,
        "save_s": save_s,
        "e2e_s": pipeline_s + save_s,
        "realtime_factor": (pipeline_s + save_s) / bucket.output_duration_s,
        "faster_than_realtime": (pipeline_s + save_s) < bucket.output_duration_s,
        "load_s": load_s,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "phase_summary": phase_summary,
    }
    path = output_dir / f"{args.bucket}_{args.output_type}_profile.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"profile_path": str(path), **result}, indent=2))


def _wrap_module(owner: object, attr_name: str, label: str, timings: dict[str, list[float]], torch: object) -> None:
    module = getattr(owner, attr_name, None)
    if module is not None:
        _wrap_callable(module, "forward", label, timings, torch)


def _wrap_callable(owner: object | None, attr_name: str, label: str, timings: dict[str, list[float]], torch: object) -> None:
    if owner is None or not hasattr(owner, attr_name):
        return
    original = getattr(owner, attr_name)
    if not callable(original):
        return

    def timed(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[label].append(time.perf_counter() - started)
        return result

    setattr(owner, attr_name, timed)


if __name__ == "__main__":
    main()
