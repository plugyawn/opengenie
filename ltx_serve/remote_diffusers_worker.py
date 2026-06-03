from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from benchmarks.diffusers_ltx23_benchmark import _save_output
from ltx_serve.buckets import BUCKETS, PRODUCTION_BUCKET_NAME, choose_bucket
from ltx_serve.schemas import GenerateRequest


MODEL_PATH = os.getenv("LTX23_MODEL_PATH", "diffusers/LTX-2.3-Distilled-Diffusers")
FORCED_BUCKET = os.getenv("LTX_WORKER_BUCKET", PRODUCTION_BUCKET_NAME)
OUTPUT_DIR = Path(os.getenv("LTX_WORKER_OUTPUT_DIR", Path.cwd() / "outputs" / "worker"))
PUBLIC_BASE_URL = os.getenv("LTX_PUBLIC_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
STEPS = int(os.getenv("LTX_WORKER_STEPS", "1"))
GUIDANCE_SCALE = float(os.getenv("LTX_WORKER_GUIDANCE_SCALE", "1.0"))
STG_SCALE = float(os.getenv("LTX_WORKER_STG_SCALE", "0.0"))
DTYPE_NAME = os.getenv("LTX_WORKER_DTYPE", "bf16")
DEVICE_MAP = os.getenv("LTX_WORKER_DEVICE_MAP", "cuda")
VAE_TILING = os.getenv("LTX_WORKER_VAE_TILING", "1") not in {"0", "false", "False"}
VAE_SLICING = os.getenv("LTX_WORKER_VAE_SLICING", "1") not in {"0", "false", "False"}


app = FastAPI(title="LTX-2.3 Prime Diffusers Worker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

_pipe: Any | None = None
_torch: Any | None = None
_export_to_video: Any | None = None
_encode_video: Any | None = None
_load_s: float | None = None
_lock = threading.Lock()


class WorkerGenerateResponse(BaseModel):
    job_id: str
    bucket: str
    output_url: str
    output_path: str
    runtime_s: float
    target_duration_s: float
    realtime_factor: float
    faster_than_realtime: bool
    generation_s: float
    save_s: float
    width: int
    height: int
    frames: int
    fps: int
    peak_vram_gb: float | None = None
    load_s: float | None = None


@app.on_event("startup")
def startup() -> None:
    load_pipeline()


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ok": _pipe is not None,
        "model_path": MODEL_PATH,
        "forced_bucket": FORCED_BUCKET,
        "steps": STEPS,
        "dtype": DTYPE_NAME,
        "load_s": _load_s,
    }


@app.post("/generate", response_model=WorkerGenerateResponse)
def generate(req: GenerateRequest) -> WorkerGenerateResponse:
    pipe, torch, export_to_video, encode_video = load_pipeline()
    bucket = BUCKETS[FORCED_BUCKET] if FORCED_BUCKET else choose_bucket(req.duration_s, req.tier)
    seed = req.seed if req.seed is not None else 10
    job_id = uuid.uuid4().hex[:12]
    output_path = OUTPUT_DIR / f"{job_id}_{bucket.name}.mp4"

    with _lock:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        start = time.perf_counter()
        output = pipe(
            prompt=req.prompt,
            height=bucket.height,
            width=bucket.width,
            num_frames=bucket.frames,
            frame_rate=float(bucket.fps),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            stg_scale=STG_SCALE,
            generator=generator,
            output_type="pil",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_s = time.perf_counter() - start
        save_start = time.perf_counter()
        _save_output(output, output_path, bucket, pipe, export_to_video, encode_video)
        save_s = time.perf_counter() - save_start
        runtime_s = time.perf_counter() - start
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    return WorkerGenerateResponse(
        job_id=job_id,
        bucket=bucket.name,
        output_url=f"{PUBLIC_BASE_URL}/outputs/{output_path.name}",
        output_path=str(output_path),
        runtime_s=runtime_s,
        target_duration_s=bucket.output_duration_s,
        realtime_factor=runtime_s / bucket.output_duration_s,
        faster_than_realtime=runtime_s < bucket.output_duration_s,
        generation_s=generation_s,
        save_s=save_s,
        width=bucket.final_width,
        height=bucket.final_height,
        frames=bucket.final_frames,
        fps=bucket.final_fps,
        peak_vram_gb=peak_vram_gb,
        load_s=_load_s,
    )


def load_pipeline() -> tuple[Any, Any, Any, Any]:
    global _pipe, _torch, _export_to_video, _encode_video, _load_s
    if _pipe is not None:
        return _pipe, _torch, _export_to_video, _encode_video

    import torch
    from diffusers import DiffusionPipeline
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from diffusers.utils import export_to_video

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[DTYPE_NAME]

    started = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=DEVICE_MAP,
    )
    if VAE_TILING and hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
    if VAE_SLICING and hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()

    _load_s = time.perf_counter() - started
    _pipe = pipe
    _torch = torch
    _export_to_video = export_to_video
    _encode_video = encode_video
    return _pipe, _torch, _export_to_video, _encode_video


def main() -> None:
    port = int(os.getenv("PORT", "9000"))
    uvicorn.run("ltx_serve.remote_diffusers_worker:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
