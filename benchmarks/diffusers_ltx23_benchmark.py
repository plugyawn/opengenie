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
DEFAULT_MODEL = os.getenv("LTX23_MODEL_PATH", "diffusers/LTX-2.3-Distilled-Diffusers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LTX-2.3 Diffusers benchmark and emit JSON.")
    parser.add_argument("--bucket", default="fast_5s_v1", choices=sorted(BUCKETS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--stg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
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

    load_started = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    if args.vae_tiling and hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
    if args.vae_slicing and hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
    load_s = time.perf_counter() - load_started

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    measured: list[BenchmarkResult] = []
    total_runs = args.warmup + args.runs
    for idx in range(total_runs):
        output_path = output_dir / f"{args.bucket}_run_{idx + 1}.mp4"
        if output_path.exists():
            output_path.unlink()

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
            output_type="pil",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_s = time.perf_counter() - start

        save_s = 0.0
        if args.save_video:
            save_started = time.perf_counter()
            _save_output(output, output_path, bucket, pipe, export_to_video, encode_video)
            save_s = time.perf_counter() - save_started
        e2e_runtime_s = generation_s + save_s

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_name = None
            peak_vram_gb = None

        bench = BenchmarkResult(
            backend=(
                f"diffusers:{args.model_path}:dtype={args.dtype}:steps={args.steps}:"
                f"guidance={args.guidance_scale}:stg={args.stg_scale}:"
                f"vae_tiling={args.vae_tiling}:vae_slicing={args.vae_slicing}"
            ),
            bucket=bucket.name,
            prompt=args.prompt,
            output_path=str(output_path) if args.save_video else None,
            runtime_s=e2e_runtime_s,
            e2e_runtime_s=e2e_runtime_s,
            target_duration_s=bucket.output_duration_s,
            realtime_factor=e2e_runtime_s / bucket.output_duration_s,
            faster_than_realtime=e2e_runtime_s < bucket.output_duration_s,
            gpu_name=gpu_name,
            peak_vram_gb=peak_vram_gb,
            stage_times={"load_s": load_s, "generation_s": generation_s, "save_s": save_s},
        )
        print(bench.model_dump_json())
        if idx >= args.warmup:
            measured.append(bench)

    summary = {
        "model_path": args.model_path,
        "bucket": bucket.name,
        "runs": [m.model_dump() for m in measured],
        "best_e2e_runtime_s": min((m.e2e_runtime_s for m in measured), default=None),
        "best_realtime_factor": min((m.realtime_factor for m in measured), default=None),
        "faster_than_realtime": any(m.faster_than_realtime for m in measured),
    }
    summary_path = output_dir / f"{args.bucket}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2))


def _save_output(output: object, output_path: Path, bucket: object, pipe: object, export_to_video: object, encode_video: object) -> None:
    frames = getattr(output, "frames", None)
    audio = getattr(output, "audio", None)
    if frames is None:
        raise RuntimeError("pipeline output did not include frames")
    video = frames[0] if isinstance(frames, (list, tuple)) else frames
    video = _center_crop_video(video, int(bucket.final_width), int(bucket.final_height))
    video = _trim_video(video, int(bucket.final_frames))

    if audio is not None:
        audio_sample_rate = _audio_sample_rate(pipe)
        audio_tensor = audio[0] if hasattr(audio, "ndim") and getattr(audio, "ndim") > 1 else audio
        encode_video(video, fps=bucket.final_fps, audio=audio_tensor, audio_sample_rate=audio_sample_rate, output_path=str(output_path))
    else:
        export_to_video(video, str(output_path), fps=bucket.final_fps)


def _center_crop_video(video: object, target_width: int, target_height: int) -> object:
    if isinstance(video, list) and video and hasattr(video[0], "crop"):
        width, height = video[0].size
        if width == target_width and height == target_height:
            return video
        left = max((width - target_width) // 2, 0)
        top = max((height - target_height) // 2, 0)
        return [frame.crop((left, top, left + target_width, top + target_height)) for frame in video]
    return video


def _trim_video(video: object, target_frames: int) -> object:
    if isinstance(video, list) and len(video) > target_frames:
        return video[:target_frames]
    return video


def _audio_sample_rate(pipe: object) -> int:
    for owner_name in ("vocoder", "audio_vae"):
        owner = getattr(pipe, owner_name, None)
        config = getattr(owner, "config", None)
        for key in ("sample_rate", "sampling_rate", "audio_sample_rate"):
            value = getattr(config, key, None) if config is not None else None
            if value:
                return int(value)
    return 44100


if __name__ == "__main__":
    main()
