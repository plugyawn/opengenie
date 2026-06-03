from __future__ import annotations

import argparse
import atexit
import math
import os
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from benchmarks.official_ltx23_benchmark import (
    _BatchSimpleDenoiser,
    _attach_audio_to_encoder,
    _build_compilation_config,
    _build_image_conditionings,
    _build_quantization_policy,
    _encode_prompt_contexts,
    _encode_video_stream_probe,
    _finish_video_stream_probe,
    _frame_block_to_uint8_numpy,
    _patch_attention_shape_profile,
    _patch_compile_for_fp8_linear,
    _patch_compile_for_static_shapes,
    _patch_flash_attention3_options,
    _patch_hot_pipeline_blocks,
    _patch_sampler_progress,
    _patch_torch_addcmul_residuals,
    _patch_triton_adazero,
    _patch_triton_audio_ada_values,
    _patch_triton_cross_attn_adaln,
    _patch_triton_ffn_gelu_fp8_quant,
    _patch_triton_fp8_quantize,
    _patch_triton_residual_gate,
    _patch_triton_simple_residual_gate,
    _patch_triton_video_ada_values,
    _patch_triton_video_ffn_out_bias_residual,
    _patch_triton_video_out_bias_residual,
    _patch_triton_video_preattention,
    _patch_triton_video_qk_bias_preattention,
    _patch_triton_video_qk_grouped_mm,
    _patch_triton_video_qkv_grouped_mm,
    _patch_triton_video_qkv_quant_reuse,
    _patch_triton_video_msa_branch,
    _patch_triton_video_text_adaln,
    _patch_triton_video_text_context_adaln,
    _patch_rope_embedding_cache,
    _patch_uniform_timestep_adaln,
    _pin_attention,
    _residual_cache_config_from_args,
    _resolve_attention,
    _run_stage_batch,
    _sigmas,
    _spd_config_from_args,
    _start_video_stream_probe,
    _video_decode_tiling_config,
)
from ltx_serve.buckets import (
    BUCKETS,
    LIVE_BASE_BUCKET,
    LIVE_BASE_BUCKET_NAME,
    LIVE_BUCKET,
    LIVE_BUCKET_NAME,
    LIVE_OVERLAP3_BUCKET,
    LIVE_OVERLAP3_BUCKET_NAME,
    LIVE_OVERLAP_BUCKET,
    LIVE_OVERLAP_BUCKET_NAME,
    PRODUCTION_BUCKET,
    ROLLING_2S_BUCKET,
    ROLLING_4S_BUCKET,
    resolve_bucket,
)
from ltx_serve.latent_continuity import LatentTailConditioning
from ltx_serve.schemas import GenerateRequest


DEFAULT_CHECKPOINT_PATH = "/home/ubuntu/models/LTX-2.3-fp8/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA_ROOT = (
    "/home/ubuntu/.cache/huggingface/hub/models--FastVideo--LTX2-Distilled-Diffusers/"
    "snapshots/0762ece944ea65f45cd3318981423e1670ff7225/text_encoder/gemma"
)

OUTPUT_DIR = Path(os.getenv("LTX_WORKER_OUTPUT_DIR", Path.cwd() / "outputs" / "official_worker"))
PUBLIC_BASE_URL = os.getenv("LTX_PUBLIC_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
LOAD_ON_STARTUP = os.getenv("LTX_WORKER_LOAD_ON_STARTUP", "1") not in {"0", "false", "False"}


app = FastAPI(title="LTX-2.3 Official Exact Worker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

_engine: "OfficialEngine | None" = None
_engine_lock = threading.Lock()
_request_lock = threading.Lock()
_encode_executor = ThreadPoolExecutor(max_workers=int(os.getenv("LTX_WORKER_ENCODE_BACKGROUND_WORKERS", "2")))
_encode_jobs: dict[str, "BackgroundEncodeState"] = {}
_encode_jobs_lock = threading.Lock()


class BackgroundEncodeState:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.done = threading.Event()
        self.stats: dict[str, float | int | str] | None = None
        self.error: str | None = None


class _FullFrameOutputBucket:
    def __init__(self, bucket: Any) -> None:
        self._bucket = bucket

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bucket, name)

    @property
    def final_frames(self) -> int:
        return int(self._bucket.frames)

    @property
    def output_duration_s(self) -> float:
        return float(self.final_frames) / float(self.final_fps)


class _FrameCountOutputBucket:
    def __init__(self, bucket: Any, final_frames: int) -> None:
        self._bucket = bucket
        self._final_frames = int(final_frames)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bucket, name)

    @property
    def final_frames(self) -> int:
        return self._final_frames

    @property
    def output_duration_s(self) -> float:
        return float(self.final_frames) / float(self.final_fps)


class WorkerGenerateResponse(BaseModel):
    job_id: str
    bucket: str
    output_url: str
    output_path: str
    media_type: str
    runtime_s: float
    target_duration_s: float
    realtime_factor: float
    faster_than_realtime: bool
    time_to_first_video_byte_s: float
    first_byte_realtime_factor: float
    prompt_s: float
    denoise_s: float
    decode_video_s: float
    encode_first_byte_s: float
    encode_video_s: float
    decode_audio_s: float
    encode_bytes: int
    encode_detached: bool = False
    encode_complete: bool = True
    peak_vram_gb: float | None = None
    load_s: float | None = None
    spd: dict[str, Any] | None = None
    residual_cache: dict[str, Any] | None = None
    stage_times: dict[str, float]


def _apply_request_residual_cache_overrides(args: argparse.Namespace, req: GenerateRequest) -> None:
    mode = req.residual_cache_mode
    if mode is None:
        return
    args.residual_cache_mode = mode
    if mode == "off":
        args.residual_cache_mag_ratios = ""
        args.residual_cache_force_skip_steps = ""
        return
    if not (req.allow_quality_risk_residual_cache or args.allow_quality_risk_residual_cache):
        raise HTTPException(status_code=400, detail="residual cache requires explicit quality-risk opt in")
    args.allow_quality_risk_residual_cache = True
    if req.residual_cache_threshold is not None:
        args.residual_cache_threshold = float(req.residual_cache_threshold)
    if req.residual_cache_max_skips is not None:
        args.residual_cache_max_skips = int(req.residual_cache_max_skips)
    if req.residual_cache_retention_ratio is not None:
        args.residual_cache_retention_ratio = float(req.residual_cache_retention_ratio)
    if req.residual_cache_metric_element_stride is not None:
        args.residual_cache_metric_element_stride = int(req.residual_cache_metric_element_stride)
    if req.residual_cache_force_skip_steps is not None:
        args.residual_cache_force_skip_steps = ",".join(str(int(step)) for step in req.residual_cache_force_skip_steps)
    elif mode != "force-skip":
        args.residual_cache_force_skip_steps = ""
    if req.residual_cache_mag_ratios is not None:
        args.residual_cache_mag_ratios = ",".join(f"{float(ratio):.12g}" for ratio in req.residual_cache_mag_ratios)
    elif mode != "magcache":
        args.residual_cache_mag_ratios = ""
    if mode == "magcache" and not str(args.residual_cache_mag_ratios).strip():
        raise HTTPException(status_code=400, detail="magcache mode requires residual_cache_mag_ratios")
    if mode == "force-skip" and not str(args.residual_cache_force_skip_steps).strip():
        raise HTTPException(status_code=400, detail="force-skip mode requires residual_cache_force_skip_steps")


def _apply_request_spd_overrides(args: argparse.Namespace, req: GenerateRequest, bucket: Any) -> None:
    if req.spd is None:
        return
    if not req.spd:
        args.spd = False
        return
    if not req.allow_quality_risk_spd:
        raise HTTPException(status_code=400, detail="SPD is lab-only and requires explicit quality-risk opt in")
    if req.image_conditioning_path or req.continuation_video_path:
        raise HTTPException(status_code=400, detail="SPD request override is currently pure T2V only")
    args.spd = True
    if req.spd_scale is not None:
        args.spd_scale = float(req.spd_scale)
    if req.spd_transition_step is not None:
        args.spd_transition_step = int(req.spd_transition_step)
    if req.spd_mid_scale is not None:
        args.spd_mid_scale = float(req.spd_mid_scale)
    if req.spd_mid_transition_step is not None:
        args.spd_mid_transition_step = int(req.spd_mid_transition_step)
    if req.spd_taper is not None:
        args.spd_taper = int(req.spd_taper)
    if req.spd_initial_dct_downscale is not None:
        args.spd_initial_dct_downscale = bool(req.spd_initial_dct_downscale)
    if req.spd_highfreq_noise is not None:
        args.spd_highfreq_noise = bool(req.spd_highfreq_noise)
    if req.spd_highfreq_source is not None:
        args.spd_highfreq_source = str(req.spd_highfreq_source)
    if args.steps != 8:
        raise HTTPException(status_code=400, detail="SPD override is implemented only for the 8-step distilled schedule")
    try:
        _spd_config_from_args(args, bucket)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class OfficialEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bucket = BUCKETS[args.bucket]
        self.pipeline: Any | None = None
        self.transformer: Any | None = None
        self.transformer_context: AbstractContextManager[Any] | None = None
        self.torch: Any | None = None
        self.GaussianNoiser: Any | None = None
        self.ModalitySpec: Any | None = None
        self.sigmas: Any | None = None
        self.resolved_attention: str | None = None
        self.load_s: float | None = None
        self.prompt_context_cache: OrderedDict[tuple[str, int, str, str], tuple[Any, Any]] = OrderedDict()

    def load(self) -> None:
        if self.pipeline is not None:
            return

        started = time.perf_counter()
        import torch
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
        from ltx_pipelines.utils.types import ModalitySpec

        _apply_quality_preserving_patches(self.args)
        quantization_policy = _build_quantization_policy(self.args.quantization, self.args.checkpoint_path)
        compilation_config = _build_compilation_config(self.args)
        device = torch.device(self.args.device)

        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=self.args.checkpoint_path,
            gemma_root=self.args.gemma_root,
            loras=[],
            quantization=quantization_policy,
            compilation_config=compilation_config,
            device=device,
        )
        _patch_hot_pipeline_blocks(self.args, pipeline)
        self.resolved_attention = _resolve_attention(self.args.attention)
        pipeline.stage = _pin_attention(pipeline.stage, self.resolved_attention)

        transformer_context = pipeline.stage.model_context()
        transformer = transformer_context.__enter__()
        sigmas = _sigmas(self.args.steps, DISTILLED_SIGMAS).to(dtype=torch.float32, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.pipeline = pipeline
        self.transformer_context = transformer_context
        self.transformer = transformer
        self.torch = torch
        self.GaussianNoiser = GaussianNoiser
        self.ModalitySpec = ModalitySpec
        self.sigmas = sigmas
        self.load_s = time.perf_counter() - started

    def close(self) -> None:
        if self.transformer_context is not None:
            self.transformer_context.__exit__(None, None, None)
            self.transformer_context = None

    def generate(self, req: GenerateRequest, job_dir: Path) -> WorkerGenerateResponse:
        self.load()
        assert self.pipeline is not None
        assert self.transformer is not None
        assert self.torch is not None
        assert self.GaussianNoiser is not None
        assert self.ModalitySpec is not None
        assert self.sigmas is not None

        try:
            requested_bucket = resolve_bucket(req.duration_s, req.tier, req.bucket)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        live_aliases = {LIVE_BASE_BUCKET_NAME, LIVE_OVERLAP_BUCKET_NAME, LIVE_OVERLAP3_BUCKET_NAME}
        live_overlap_alias = requested_bucket.name in live_aliases and self.bucket.name in live_aliases
        if requested_bucket.name != self.bucket.name and not live_overlap_alias:
            raise HTTPException(status_code=400, detail=f"worker is pinned to {self.bucket.name}")

        torch = self.torch
        args = argparse.Namespace(**vars(self.args))
        args.prompt = req.prompt
        args.seed = req.seed if req.seed is not None else self.args.seed
        args.image_conditioning_path = req.image_conditioning_path
        args.image_conditioning_frame_idx = req.image_conditioning_frame_idx
        args.image_conditioning_strength = req.image_conditioning_strength
        args.image_conditioning_crf = req.image_conditioning_crf
        _apply_request_residual_cache_overrides(args, req)
        _apply_request_spd_overrides(args, req, self.bucket)
        job_dir.mkdir(parents=True, exist_ok=True)
        job_id = job_dir.name
        requested_continuation_frames = 0
        continuation_start_frame = 0
        latent_video_tail_frames = 0
        latent_effective_continuation_frames = 0
        continuation_extract_s = 0.0
        latent_continuation_s = 0.0
        latent_save_s = 0.0

        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            request_start = time.perf_counter()
            prompt_s, video_context, audio_context, prompt_cache_hit = self._encode_prompt_contexts_cached(args, torch)
            video_conditionings: list[Any] = []
            audio_conditionings: list[Any] = []
            if req.continuation_video_path and req.continuation_frames > 0 and req.continuation_mode == "latent":
                requested_continuation_frames = req.continuation_frames
                (
                    latent_continuation_s,
                    video_conditionings,
                    audio_conditionings,
                    latent_video_tail_frames,
                    latent_effective_continuation_frames,
                ) = _build_latent_continuation_conditionings(
                    video_path=Path(req.continuation_video_path),
                    bucket=self.bucket,
                    torch=torch,
                    device=args.device,
                    dtype=getattr(self.pipeline, "dtype", getattr(self.pipeline, "_dtype", torch.bfloat16)),
                    frames=req.continuation_frames,
                    strength=req.continuation_strength,
                )
                continuation_start_frame = latent_effective_continuation_frames
            elif req.continuation_video_path and req.continuation_frames > 0:
                requested_continuation_frames = req.continuation_frames
                continuation_extract_s, args.image_conditioning_spec = _prepare_continuation_specs(
                    video_path=Path(req.continuation_video_path),
                    job_dir=job_dir,
                    frames=req.continuation_frames,
                    strength=req.continuation_strength,
                    crf=req.continuation_crf,
                )
                continuation_start_frame = req.continuation_frames
            image_conditioning_s, image_conditionings = _build_image_conditionings(args, self.pipeline, self.bucket, torch)
            video_conditionings.extend(image_conditionings)
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            noiser = self.GaussianNoiser(generator=generator)
            spd_stats: dict[str, Any] = {}
            residual_cache_stats: dict[str, Any] = {}

            denoise_start = time.perf_counter()
            video_state, audio_state = _run_stage_batch(
                stage=self.pipeline.stage,
                transformer=self.transformer,
                denoiser=_BatchSimpleDenoiser(
                    video_context,
                    audio_context,
                    args.batch_size,
                    residual_cache_config=_residual_cache_config_from_args(args),
                    residual_cache_stats=residual_cache_stats,
                ),
                sigmas=self.sigmas,
                noiser=noiser,
                width=self.bucket.width,
                height=self.bucket.height,
                frames=self.bucket.frames,
                fps=float(self.bucket.fps),
                video=self.ModalitySpec(context=video_context, conditionings=video_conditionings),
                audio=self.ModalitySpec(context=audio_context, conditionings=audio_conditionings),
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size or args.batch_size,
                uniform_timestep_adaln=args.uniform_timestep_adaln,
                spd_config=_spd_config_from_args(args, self.bucket) if args.spd else None,
                spd_generator=generator,
                spd_stats=spd_stats,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            denoise_s = time.perf_counter() - denoise_start

            latent_save_start = time.perf_counter()
            _save_continuation_latents(torch, job_dir, video_state, audio_state)
            latent_save_s = time.perf_counter() - latent_save_start

            post_denoise_start = time.perf_counter()
            emitted_start_frame = continuation_start_frame
            emit_overlap_frames = bool(
                args.emit_overlap_frames
                and "overlap" in getattr(self.bucket, "name", "")
                and int(self.bucket.frames) > int(self.bucket.final_frames)
            )
            if emit_overlap_frames:
                emitted_start_frame = 0
            elif (
                emitted_start_frame == 0
                and "overlap" in getattr(self.bucket, "name", "")
                and int(self.bucket.frames) > int(self.bucket.final_frames)
            ):
                emitted_start_frame = int(self.bucket.frames) - int(self.bucket.final_frames)
            latent_emit_to_tail = bool(
                args.latent_continuation_emit_to_tail
                and latent_effective_continuation_frames > 0
                and not emit_overlap_frames
            )
            if emit_overlap_frames:
                encode_bucket = _FullFrameOutputBucket(self.bucket)
            elif latent_emit_to_tail:
                encode_bucket = _FrameCountOutputBucket(self.bucket, int(self.bucket.frames) - emitted_start_frame)
            else:
                encode_bucket = self.bucket
            audio_trim_start_s = float(emitted_start_frame) / float(self.bucket.final_fps)
            prestarted_encoder = None
            decoded_audio = None
            decode_audio_s = 0.0
            should_mux_audio = bool(args.decode_audio and req.audio and args.encode_mux_audio)
            if args.encode_prestart and args.encode_video_stream:
                if should_mux_audio and args.encode_audio_pipe:
                    prestarted_encoder = _start_video_stream_probe(
                        encode_bucket,
                        job_dir,
                        0,
                        args.encode_codec,
                        args.encode_container,
                        x264_preset=args.encode_x264_preset,
                        x264_crf=args.encode_x264_crf,
                        x264_params=args.encode_x264_params,
                        encode_threads=args.encode_threads,
                        low_latency_mux=args.encode_low_latency_mux,
                        movflags=args.encode_movflags,
                        audio_pipe=True,
                        audio_trim_start_s=audio_trim_start_s,
                    )
                    decode_audio_start = time.perf_counter()
                    decoded_audio = self.pipeline.audio_decoder(audio_state.latent)
                    _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    decode_audio_s = time.perf_counter() - decode_audio_start
                    _attach_audio_to_encoder(prestarted_encoder, decoded_audio, trim_start_s=audio_trim_start_s)
                else:
                    if should_mux_audio:
                        decode_audio_start = time.perf_counter()
                        decoded_audio = self.pipeline.audio_decoder(audio_state.latent)
                        _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        decode_audio_s = time.perf_counter() - decode_audio_start
                    prestarted_encoder = _start_video_stream_probe(
                        encode_bucket,
                        job_dir,
                        0,
                        args.encode_codec,
                        args.encode_container,
                        x264_preset=args.encode_x264_preset,
                        x264_crf=args.encode_x264_crf,
                        x264_params=args.encode_x264_params,
                        encode_threads=args.encode_threads,
                        low_latency_mux=args.encode_low_latency_mux,
                        movflags=args.encode_movflags,
                        audio=decoded_audio,
                        audio_trim_start_s=audio_trim_start_s,
                    )

            decode_video_start = time.perf_counter()
            tiling_config = _video_decode_tiling_config(args)
            first_video_chunk = None
            first_video_chunk_s = 0.0
            for chunk_index, chunk in enumerate(
                self.pipeline.video_decoder(video_state.latent, tiling_config=tiling_config, generator=generator)
            ):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if first_video_chunk is None:
                    first_video_chunk = chunk
                    first_video_chunk_s = time.perf_counter() - decode_video_start
                if chunk_index > 0:
                    raise RuntimeError("stream worker currently expects a single decoded video chunk")
            if first_video_chunk is None:
                raise RuntimeError("video decoder produced no chunks")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_video_s = time.perf_counter() - decode_video_start

            if should_mux_audio and decoded_audio is None:
                decode_audio_start = time.perf_counter()
                decoded_audio = self.pipeline.audio_decoder(audio_state.latent)
                _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_audio_s = time.perf_counter() - decode_audio_start

            encode_detached = bool(args.encode_detach_after_first_byte and prestarted_encoder is not None)
            encode_complete = not encode_detached
            if prestarted_encoder is not None and encode_detached:
                background_start_frame = emitted_start_frame
                if args.encode_cpu_materialize_before_detach:
                    first_video_chunk = _trim_crop_video_chunk(first_video_chunk, encode_bucket, emitted_start_frame)
                    first_video_chunk = _frame_block_to_uint8_numpy(first_video_chunk, torch)
                    background_start_frame = 0
                state = _register_background_encode(job_id, prestarted_encoder.output_path)
                _encode_executor.submit(
                    _finish_background_encode,
                    state,
                    prestarted_encoder,
                    first_video_chunk,
                    encode_bucket,
                    args.encode_feed_mode,
                    args.encode_frame_batch,
                    False,
                    background_start_frame,
                )
                first_byte_abs_s = _wait_for_first_byte(
                    prestarted_encoder,
                    timeout_s=args.encode_first_byte_timeout_s,
                )
                encode_stats = {
                    "encode_video_s": time.perf_counter() - prestarted_encoder.encode_start,
                    "encode_first_byte_s": first_byte_abs_s - prestarted_encoder.encode_start,
                    "encode_first_byte_abs_s": first_byte_abs_s,
                    "encode_bytes": prestarted_encoder.byte_count[0],
                    "output_path": str(prestarted_encoder.output_path),
                }
            elif prestarted_encoder is not None:
                encode_stats = _finish_video_stream_probe(
                    prestarted_encoder,
                    first_video_chunk,
                    encode_bucket,
                    feed_mode=args.encode_feed_mode,
                    frame_batch=args.encode_frame_batch,
                    start_frame=emitted_start_frame,
                )
            else:
                encode_stats = _encode_video_stream_probe(
                    first_video_chunk,
                    encode_bucket,
                    job_dir,
                    0,
                    args.encode_codec,
                    args.encode_container,
                    feed_mode=args.encode_feed_mode,
                    frame_batch=args.encode_frame_batch,
                    x264_preset=args.encode_x264_preset,
                    x264_crf=args.encode_x264_crf,
                    x264_params=args.encode_x264_params,
                    encode_threads=args.encode_threads,
                    low_latency_mux=args.encode_low_latency_mux,
                    movflags=args.encode_movflags,
                    audio=decoded_audio,
                    audio_trim_start_s=audio_trim_start_s,
                    start_frame=emitted_start_frame,
                )
            encode_video_s = float(encode_stats["encode_video_s"])
            encode_first_byte_s = float(encode_stats["encode_first_byte_s"])
            encode_bytes = int(encode_stats["encode_bytes"])
            output_path = Path(str(encode_stats["output_path"])).resolve()

            runtime_s = time.perf_counter() - request_start
            first_byte_after_denoise_s = (
                max(0.0, float(encode_stats["encode_first_byte_abs_s"]) - post_denoise_start)
                if encode_stats.get("encode_first_byte_abs_s") is not None
                else first_video_chunk_s + (decode_audio_s if should_mux_audio else 0.0) + encode_first_byte_s
            )
            time_to_first_video_byte_s = prompt_s + denoise_s + first_byte_after_denoise_s
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

        rel_path = output_path.relative_to(OUTPUT_DIR.resolve())
        output_url = (
            f"{PUBLIC_BASE_URL}/streams/{job_id}"
            if encode_detached
            else f"{PUBLIC_BASE_URL}/outputs/{rel_path.as_posix()}"
        )
        stage_times = {
            "prompt_s": prompt_s,
            "prompt_cache_hit": 1.0 if prompt_cache_hit else 0.0,
            "continuation_extract_s": continuation_extract_s,
            "latent_continuation_s": latent_continuation_s,
            "latent_save_s": latent_save_s,
            "image_conditioning_s": image_conditioning_s,
            "requested_continuation_frames": float(requested_continuation_frames),
            "continuation_frames": float(continuation_start_frame),
            "latent_video_tail_frames": float(latent_video_tail_frames),
            "latent_effective_continuation_frames": float(latent_effective_continuation_frames),
            "latent_emit_to_tail": 1.0 if latent_emit_to_tail else 0.0,
            "emitted_start_frame": float(emitted_start_frame),
            "emitted_frames": float(encode_bucket.final_frames),
            "emit_overlap_frames": 1.0 if emit_overlap_frames else 0.0,
            "denoise_s": denoise_s,
            "decode_video_s": decode_video_s,
            "first_video_chunk_s": first_video_chunk_s,
            "encode_first_byte_s": encode_first_byte_s,
            "encode_video_s": encode_video_s,
            "decode_audio_s": decode_audio_s,
            "first_byte_after_denoise_s": first_byte_after_denoise_s,
            "encode_detached": 1.0 if encode_detached else 0.0,
        }
        if args.residual_cache_mode != "off":
            stage_times["residual_cache_computed_steps"] = float(len(residual_cache_stats.get("computed_steps") or []))
            stage_times["residual_cache_skipped_steps"] = float(len(residual_cache_stats.get("skipped_steps") or []))
        return WorkerGenerateResponse(
            job_id=job_id,
            bucket=self.bucket.name,
            output_url=output_url,
            output_path=str(output_path),
            media_type="video/mp4" if args.encode_container == "frag-mp4" else "video/MP2T",
            runtime_s=runtime_s,
            target_duration_s=encode_bucket.output_duration_s,
            realtime_factor=runtime_s / encode_bucket.output_duration_s,
            faster_than_realtime=time_to_first_video_byte_s < encode_bucket.output_duration_s,
            time_to_first_video_byte_s=time_to_first_video_byte_s,
            first_byte_realtime_factor=time_to_first_video_byte_s / encode_bucket.output_duration_s,
            prompt_s=prompt_s,
            denoise_s=denoise_s,
            decode_video_s=decode_video_s,
            encode_first_byte_s=encode_first_byte_s,
            encode_video_s=encode_video_s,
            decode_audio_s=decode_audio_s,
            encode_bytes=encode_bytes,
            encode_detached=encode_detached,
            encode_complete=encode_complete,
            peak_vram_gb=peak_vram_gb,
            load_s=self.load_s,
            stage_times=stage_times,
            spd=spd_stats if args.spd else None,
            residual_cache=residual_cache_stats if args.residual_cache_mode != "off" else None,
        )

    def _encode_prompt_contexts_cached(self, args: argparse.Namespace, torch: Any) -> tuple[float, Any, Any, bool]:
        cache_size = int(os.getenv("LTX_WORKER_PROMPT_CACHE_SIZE", "16"))
        key = (str(args.prompt), int(args.batch_size), str(args.device), str(self.bucket.name))
        if cache_size > 0 and key in self.prompt_context_cache:
            started = time.perf_counter()
            video_context, audio_context = self.prompt_context_cache[key]
            self.prompt_context_cache.move_to_end(key)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter() - started, video_context, audio_context, True

        assert self.pipeline is not None
        prompt_s, video_context, audio_context = _encode_prompt_contexts(args, self.pipeline, torch)
        if cache_size > 0:
            self.prompt_context_cache[key] = (video_context, audio_context)
            while len(self.prompt_context_cache) > cache_size:
                self.prompt_context_cache.popitem(last=False)
        return prompt_s, video_context, audio_context, False


def _prepare_continuation_specs(
    *,
    video_path: Path,
    job_dir: Path,
    frames: int,
    strength: float,
    crf: int,
) -> tuple[float, list[str]]:
    started = time.perf_counter()
    if frames <= 0:
        return 0.0, []
    if not video_path.exists():
        raise FileNotFoundError(f"continuation video path does not exist: {video_path}")

    frame_dir = job_dir / "continuation_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()
    frame_pattern = frame_dir / "frame_%05d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vsync",
            "0",
            str(frame_pattern),
        ],
        check=True,
    )
    extracted = sorted(frame_dir.glob("frame_*.png"))
    if len(extracted) < frames:
        raise RuntimeError(f"needed {frames} continuation frames, extracted only {len(extracted)} from {video_path}")
    tail_frames = extracted[-frames:]
    specs = [f"{path}:{idx}:{strength}:{crf}" for idx, path in enumerate(tail_frames)]
    return time.perf_counter() - started, specs


def _build_latent_continuation_conditionings(
    *,
    video_path: Path,
    bucket: object,
    torch: Any,
    device: str,
    dtype: Any,
    frames: int,
    strength: float,
) -> tuple[float, list[Any], list[Any], int, int]:
    started = time.perf_counter()
    if frames <= 0:
        return 0.0, [], [], 0, 0

    job_dir = video_path.parent
    video_latent_path = job_dir / "video_latent.pt"
    audio_latent_path = job_dir / "audio_latent.pt"
    if not video_latent_path.exists():
        raise FileNotFoundError(f"latent continuation requested but missing {video_latent_path}")

    target_device = torch.device(device)
    video_latent = torch.load(video_latent_path, map_location="cpu").to(device=target_device, dtype=dtype)
    video_tail_frames = _video_latent_frames_for_requested_overlap(frames)
    effective_frames = _decoded_frames_for_video_latent_frames(video_tail_frames)
    if video_latent.shape[2] < video_tail_frames:
        raise RuntimeError(
            f"needed {video_tail_frames} video latent continuation frames, found {video_latent.shape[2]}"
        )
    video_tail = video_latent[:, :, -video_tail_frames:, :, :].contiguous()
    video_conditionings: list[Any] = [
        LatentTailConditioning(latent=video_tail, start_idx=0, strength=strength),
    ]

    audio_conditionings: list[Any] = []
    if audio_latent_path.exists():
        audio_latent = torch.load(audio_latent_path, map_location="cpu").to(device=target_device, dtype=dtype)
        overlap_s = float(effective_frames) / float(bucket.final_fps)
        audio_latents_per_second = 25.0
        audio_tail_frames = max(1, round(overlap_s * audio_latents_per_second))
        # Continuation chunks trim from the start but emit through the generated
        # tail, so the next audio prefix should be anchored to the final latent.
        audio_stop = int(audio_latent.shape[2])
        audio_start = max(audio_stop - audio_tail_frames, 0)
        if audio_stop > audio_start:
            audio_tail = audio_latent[:, :, audio_start:audio_stop, :].contiguous()
            audio_conditionings.append(LatentTailConditioning(latent=audio_tail, start_idx=0, strength=strength))

    return time.perf_counter() - started, video_conditionings, audio_conditionings, video_tail_frames, effective_frames


def _video_latent_frames_for_requested_overlap(frames: int) -> int:
    """Return the current latent-prefix count used for a requested pixel overlap.

    LTX video latents decode as one first frame plus 8 decoded frames per
    additional temporal latent. A request for 22 pixel frames therefore maps to
    3 latent frames, which can reproduce 17 decoded frames. The worker reports
    and trims by that effective count instead of pretending the full requested
    pixel overlap was reproduced.
    """

    if frames <= 0:
        return 0
    return max(1, math.ceil(float(frames) / 8.0))


def _decoded_frames_for_video_latent_frames(latent_frames: int) -> int:
    if latent_frames <= 0:
        return 0
    return 1 + 8 * (int(latent_frames) - 1)


def _save_continuation_latents(torch: Any, job_dir: Path, video_state: Any | None, audio_state: Any | None) -> None:
    if video_state is not None:
        torch.save(video_state.latent.detach().cpu(), job_dir / "video_latent.pt")
    if audio_state is not None:
        torch.save(audio_state.latent.detach().cpu(), job_dir / "audio_latent.pt")


def _register_background_encode(job_id: str, output_path: Path) -> BackgroundEncodeState:
    state = BackgroundEncodeState(output_path)
    with _encode_jobs_lock:
        _encode_jobs[job_id] = state
    return state


def _wait_for_background_encode(job_id: str, *, timeout_s: float) -> BackgroundEncodeState:
    deadline = time.perf_counter() + timeout_s
    while True:
        with _encode_jobs_lock:
            state = _encode_jobs.get(job_id)
        if state is not None:
            return state
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise HTTPException(status_code=404, detail="stream not found")
        time.sleep(min(0.02, remaining))


def _trim_crop_video_chunk(chunk: Any, bucket: Any, start_frame: int) -> Any:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    frames = int(min(max(int(chunk.shape[0]) - start_frame, 0), bucket.final_frames))
    if frames <= 0:
        raise ValueError("start_frame drops all decoded frames")
    crop_top = max((int(chunk.shape[1]) - int(bucket.final_height)) // 2, 0)
    crop_left = max((int(chunk.shape[2]) - int(bucket.final_width)) // 2, 0)
    return chunk[
        start_frame : start_frame + frames,
        crop_top : crop_top + int(bucket.final_height),
        crop_left : crop_left + int(bucket.final_width),
        :,
    ]


def _finish_background_encode(
    state: BackgroundEncodeState,
    session: Any,
    first_video_chunk: object,
    bucket: object,
    feed_mode: str,
    frame_batch: int,
    cpu_materialize: bool,
    start_frame: int,
) -> None:
    try:
        state.stats = _finish_video_stream_probe(
            session,
            first_video_chunk,
            bucket,
            feed_mode=feed_mode,
            frame_batch=frame_batch,
            cpu_materialize=cpu_materialize,
            start_frame=start_frame,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced by /streams.
        state.error = repr(exc)
    finally:
        state.done.set()


def _wait_for_first_byte(session: Any, *, timeout_s: float) -> float:
    deadline = time.perf_counter() + timeout_s
    while True:
        if not session.first_byte_queue.empty():
            return float(session.first_byte_queue.get_nowait())
        if session.proc.poll() is not None:
            stderr = session.proc.stderr.read().decode("utf-8", errors="replace") if session.proc.stderr is not None else ""
            raise RuntimeError(f"ffmpeg exited before first stream byte: {stderr[-2000:]}")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting {timeout_s:.1f}s for first encoded video byte")
        time.sleep(min(0.01, remaining))


def _follow_background_file(state: BackgroundEncodeState) -> Any:
    offset = 0
    while not state.output_path.exists() and not state.done.is_set():
        time.sleep(0.01)
    with state.output_path.open("rb") as file:
        while True:
            file.seek(offset)
            data = file.read(65536)
            if data:
                offset += len(data)
                yield data
                continue
            if state.done.is_set():
                if state.error is not None:
                    raise RuntimeError(state.error)
                break
            time.sleep(0.02)


def build_engine_args() -> argparse.Namespace:
    def env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.lower() not in {"0", "false", "no", "off"}

    bucket_name = os.getenv("LTX_WORKER_BUCKET", LIVE_BUCKET_NAME)

    return argparse.Namespace(
        bucket=bucket_name,
        prompt="",
        image_conditioning_path=None,
        image_conditioning_frame_idx=0,
        image_conditioning_strength=1.0,
        image_conditioning_crf=33,
        image_conditioning_spec=[],
        checkpoint_path=os.getenv("LTX23_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH),
        gemma_root=os.getenv("LTX23_GEMMA_ROOT", DEFAULT_GEMMA_ROOT),
        output_dir=str(OUTPUT_DIR),
        quantization=os.getenv("LTX_WORKER_QUANTIZATION", "fp8-scaled-mm"),
        attention=os.getenv("LTX_WORKER_ATTENTION", "h100-fa3"),
        steps=int(os.getenv("LTX_WORKER_STEPS", "8")),
        batch_size=1,
        max_batch_size=1,
        runs=1,
        warmup=0,
        seed=int(os.getenv("LTX_WORKER_SEED", "10")),
        compile=False,
        compile_mode="reduce-overhead",
        compile_backend="inductor",
        compile_fullgraph=False,
        compile_dynamic="none",
        compile_dynamic_marking=False,
        compile_trace_fp8_linear=False,
        triton_adazero=True,
        triton_fp8_quant=True,
        fp8_scaled_mm_bias_epilogue=False,
        triton_fp8_bias_add=True,
        triton_ffn_gelu_fp8_quant=False,
        triton_cross_attn_adaln=False,
        triton_video_preattention=True,
        triton_video_preattention_checks=0,
        triton_video_preattention_mode=os.getenv("LTX_WORKER_TRITON_VIDEO_PREATTENTION_MODE", "dual"),
        triton_video_ada_values=True,
        triton_audio_ada_values=env_flag("LTX_WORKER_TRITON_AUDIO_ADA_VALUES", True),
        triton_video_text_adaln=True,
        triton_video_text_context_adaln=False,
        cache_rope_embeddings=os.getenv("LTX_WORKER_CACHE_ROPE_EMBEDDINGS", "0") not in {"0", "false", "False"},
        uniform_timestep_adaln=os.getenv("LTX_WORKER_UNIFORM_TIMESTEP_ADALN", "0") not in {"0", "false", "False"},
        triton_video_out_bias_residual=True,
        triton_video_ffn_out_bias_residual=False,
        triton_video_qk_bias_preattention=False,
        triton_video_qkv_quant_reuse=False,
        triton_video_qkv_grouped_mm=False,
        triton_video_qkv_grouped_mm_checks=0,
        triton_video_qk_grouped_mm=False,
        triton_video_qk_grouped_mm_checks=0,
        triton_video_qkv_packed_linear=False,
        triton_video_qkv_packed_requant=False,
        triton_video_qkv_packed_requant_checks=0,
        allow_quality_risk_packed_qkv_requant=False,
        triton_video_msa_branch=os.getenv("LTX_WORKER_TRITON_VIDEO_MSA_BRANCH", "0") not in {"0", "false", "False"},
        triton_video_msa_branch_tokens=os.getenv("LTX_WORKER_TRITON_VIDEO_MSA_BRANCH_TOKENS", ""),
        triton_video_msa_branch_mode=os.getenv("LTX_WORKER_TRITON_VIDEO_MSA_BRANCH_MODE", "generic"),
        triton_video_msa_branch_profile=False,
        allow_quality_risk_msa_bf16_out=False,
        torch_addcmul_residuals=False,
        triton_residual_gate=False,
        triton_simple_residual_gate=True,
        allow_quality_risk_pointwise_fusions=False,
        hot_prompt_encoder=True,
        hot_decoders=True,
        prompt_inside_runs=True,
        fa3_num_splits=int(os.getenv("LTX_WORKER_FA3_NUM_SPLITS", "1")),
        fa3_attention_chunk=int(os.getenv("LTX_WORKER_FA3_ATTENTION_CHUNK", "0")),
        fa3_sm_margin=int(os.getenv("LTX_WORKER_FA3_SM_MARGIN", "0")),
        fa3_video_window_left=int(os.getenv("LTX_WORKER_FA3_VIDEO_WINDOW_LEFT", "-1")),
        fa3_video_window_right=int(os.getenv("LTX_WORKER_FA3_VIDEO_WINDOW_RIGHT", "-1")),
        fa3_video_window_checks=0,
        allow_quality_risk_windowed_attention=False,
        fa3_video_fp8_attention=False,
        fa3_video_prealloc_output=os.getenv("LTX_WORKER_FA3_VIDEO_PREALLOC_OUTPUT", "0") not in {"0", "false", "False"},
        fa3_video_fp8_components="qkv",
        allow_quality_risk_fp8_video_attention=False,
        fa3_video_fp8_checks=0,
        attention_shape_profile=False,
        block_profile=False,
        decode_video=True,
        decode_video_tiling=os.getenv("LTX_WORKER_DECODE_VIDEO_TILING", "none"),
        decode_chunk_profile=False,
        encode_video_stream=True,
        encode_codec=os.getenv("LTX_WORKER_ENCODE_CODEC", "libx264"),
        encode_container=os.getenv("LTX_WORKER_ENCODE_CONTAINER", "frag-mp4"),
        encode_feed_mode=os.getenv("LTX_WORKER_ENCODE_FEED_MODE", "frame-stream"),
        encode_frame_batch=int(os.getenv("LTX_WORKER_ENCODE_FRAME_BATCH", "2")),
        encode_x264_preset=os.getenv("LTX_WORKER_ENCODE_X264_PRESET", "ultrafast"),
        encode_x264_crf=os.getenv("LTX_WORKER_ENCODE_X264_CRF", "19"),
        encode_x264_params=os.getenv("LTX_WORKER_ENCODE_X264_PARAMS", ""),
        encode_threads=int(os.getenv("LTX_WORKER_ENCODE_THREADS", "0")),
        encode_low_latency_mux=os.getenv("LTX_WORKER_ENCODE_LOW_LATENCY_MUX", "0") not in {"0", "false", "False"},
        encode_movflags=os.getenv(
            "LTX_WORKER_ENCODE_MOVFLAGS",
            "frag_every_frame+empty_moov+default_base_moof+omit_tfhd_offset+separate_moof",
        ),
        encode_prestart=os.getenv("LTX_WORKER_ENCODE_PRESTART", "1") not in {"0", "false", "False"},
        encode_detach_after_first_byte=os.getenv("LTX_WORKER_ENCODE_DETACH_AFTER_FIRST_BYTE", "1")
        not in {"0", "false", "False"},
        encode_cpu_materialize_before_detach=os.getenv("LTX_WORKER_ENCODE_CPU_MATERIALIZE_BEFORE_DETACH", "1")
        not in {"0", "false", "False"},
        encode_first_byte_timeout_s=float(os.getenv("LTX_WORKER_ENCODE_FIRST_BYTE_TIMEOUT_S", "10.0")),
        encode_audio_pipe=os.getenv("LTX_WORKER_ENCODE_AUDIO_PIPE", "0") not in {"0", "false", "False"},
        encode_mux_audio=os.getenv("LTX_WORKER_ENCODE_MUX_AUDIO", "1") not in {"0", "false", "False"},
        decode_audio=os.getenv("LTX_WORKER_DECODE_AUDIO", "1") not in {"0", "false", "False"},
        emit_overlap_frames=env_flag("LTX_WORKER_EMIT_OVERLAP_FRAMES", False),
        latent_continuation_emit_to_tail=env_flag("LTX_WORKER_LATENT_CONTINUATION_EMIT_TO_TAIL", True),
        sampler_progress=os.getenv("LTX_WORKER_SAMPLER_PROGRESS", "0") not in {"0", "false", "False"},
        torch_profile=False,
        profile_dir=None,
        profile_row_limit=40,
        cuda_profiler_range=False,
        save_latents=False,
        spd=env_flag("LTX_WORKER_SPD", False),
        spd_scale=float(os.getenv("LTX_WORKER_SPD_SCALE", "0.5")),
        spd_transition_step=int(os.getenv("LTX_WORKER_SPD_TRANSITION_STEP", "5")),
        spd_mid_scale=(
            float(os.environ["LTX_WORKER_SPD_MID_SCALE"])
            if os.getenv("LTX_WORKER_SPD_MID_SCALE")
            else None
        ),
        spd_mid_transition_step=(
            int(os.environ["LTX_WORKER_SPD_MID_TRANSITION_STEP"])
            if os.getenv("LTX_WORKER_SPD_MID_TRANSITION_STEP")
            else None
        ),
        spd_transform=os.getenv("LTX_WORKER_SPD_TRANSFORM", "dct"),
        spd_taper=int(os.getenv("LTX_WORKER_SPD_TAPER", "8")),
        spd_initial_dct_downscale=env_flag("LTX_WORKER_SPD_INITIAL_DCT_DOWNSCALE", True),
        spd_highfreq_noise=env_flag("LTX_WORKER_SPD_HIGHFREQ_NOISE", True),
        spd_highfreq_source=os.getenv("LTX_WORKER_SPD_HIGHFREQ_SOURCE", "fresh"),
        residual_cache_mode=os.getenv("LTX_WORKER_RESIDUAL_CACHE_MODE", "off"),
        allow_quality_risk_residual_cache=env_flag("LTX_WORKER_ALLOW_QUALITY_RISK_RESIDUAL_CACHE", False),
        residual_cache_threshold=float(os.getenv("LTX_WORKER_RESIDUAL_CACHE_THRESHOLD", "0.06")),
        residual_cache_max_skips=int(os.getenv("LTX_WORKER_RESIDUAL_CACHE_MAX_SKIPS", "1")),
        residual_cache_retention_ratio=float(os.getenv("LTX_WORKER_RESIDUAL_CACHE_RETENTION_RATIO", "0.125")),
        residual_cache_force_skip_steps=os.getenv("LTX_WORKER_RESIDUAL_CACHE_FORCE_SKIP_STEPS", ""),
        residual_cache_mag_ratios=os.getenv("LTX_WORKER_RESIDUAL_CACHE_MAG_RATIOS", ""),
        residual_cache_metric_element_stride=int(os.getenv("LTX_WORKER_RESIDUAL_CACHE_METRIC_ELEMENT_STRIDE", "1")),
        device=os.getenv("LTX_WORKER_DEVICE", "cuda"),
    )


def _apply_quality_preserving_patches(args: argparse.Namespace) -> None:
    _patch_sampler_progress(args)
    _patch_compile_for_static_shapes(args)
    _patch_compile_for_fp8_linear(args)
    _patch_triton_adazero(args)
    _patch_triton_fp8_quantize(args)
    _patch_triton_ffn_gelu_fp8_quant(args)
    _patch_triton_cross_attn_adaln(args)
    _patch_triton_video_preattention(args)
    _patch_torch_addcmul_residuals(args)
    _patch_triton_residual_gate(args)
    _patch_triton_simple_residual_gate(args)
    _patch_triton_video_ada_values(args)
    _patch_triton_audio_ada_values(args)
    _patch_triton_video_text_adaln(args)
    _patch_triton_video_text_context_adaln(args)
    _patch_rope_embedding_cache(args)
    _patch_uniform_timestep_adaln(args)
    _patch_triton_video_out_bias_residual(args)
    _patch_triton_video_ffn_out_bias_residual(args)
    _patch_triton_video_qk_bias_preattention(args)
    _patch_triton_video_qkv_quant_reuse(args)
    _patch_triton_video_qkv_grouped_mm(args)
    _patch_triton_video_qk_grouped_mm(args)
    _patch_triton_video_msa_branch(args)
    _patch_flash_attention3_options(args)
    _patch_attention_shape_profile(args)


def get_engine() -> OfficialEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            args = build_engine_args()
            if args.bucket not in BUCKETS:
                raise RuntimeError(f"unknown official worker bucket: {args.bucket}")
            if args.steps != 8:
                raise RuntimeError("official distilled-1.1 worker must stay pinned to 8 steps")
            if args.encode_container != "frag-mp4":
                raise RuntimeError("official worker must emit browser-playable fragmented MP4")
            if args.decode_audio and not args.encode_mux_audio:
                raise RuntimeError("official audio worker must mux generated audio into the fragmented MP4")
            if args.encode_audio_pipe and not (args.encode_prestart and args.decode_audio and args.encode_mux_audio):
                raise RuntimeError("audio FIFO prestart requires prestart, generated audio decode, and AV muxing")
            if args.triton_video_msa_branch_mode == "direct_bf16_out" and not args.allow_quality_risk_msa_bf16_out:
                raise RuntimeError("direct_bf16_out MSA branch mode is blocked because it changes same-seed latents")
            _engine = OfficialEngine(args)
        return _engine


@app.on_event("startup")
def startup() -> None:
    if LOAD_ON_STARTUP:
        get_engine().load()


@app.get("/healthz")
def healthz() -> dict[str, object]:
    engine_loaded = _engine is not None and _engine.pipeline is not None
    args = _engine.args if _engine is not None else build_engine_args()
    video_msa_branch_stats = None
    if args.triton_video_msa_branch:
        from ltx_serve.triton_ltx_ops import collect_video_msa_branch_stats

        video_msa_branch_stats = collect_video_msa_branch_stats()
    return {
        "ok": True,
        "loaded": engine_loaded,
        "bucket": args.bucket,
        "steps": args.steps,
        "attention": args.attention,
        "quantization": args.quantization,
        "encode_container": args.encode_container,
        "encode_feed_mode": args.encode_feed_mode,
        "encode_frame_batch": args.encode_frame_batch,
        "encode_x264_preset": args.encode_x264_preset,
        "encode_x264_crf": args.encode_x264_crf,
        "encode_x264_params": args.encode_x264_params,
        "encode_threads": args.encode_threads,
        "encode_low_latency_mux": args.encode_low_latency_mux,
        "encode_movflags": args.encode_movflags,
        "encode_prestart": args.encode_prestart,
        "encode_detach_after_first_byte": args.encode_detach_after_first_byte,
        "encode_cpu_materialize_before_detach": args.encode_cpu_materialize_before_detach,
        "encode_first_byte_timeout_s": args.encode_first_byte_timeout_s,
        "encode_audio_pipe": args.encode_audio_pipe,
        "encode_mux_audio": args.encode_mux_audio,
        "emit_overlap_frames": args.emit_overlap_frames,
        "latent_continuation_emit_to_tail": args.latent_continuation_emit_to_tail,
        "sampler_progress": args.sampler_progress,
        "triton_audio_ada_values": args.triton_audio_ada_values,
        "triton_video_preattention_mode": args.triton_video_preattention_mode,
        "triton_video_ffn_out_bias_residual": args.triton_video_ffn_out_bias_residual,
        "triton_ffn_gelu_fp8_quant": args.triton_ffn_gelu_fp8_quant,
        "triton_video_qkv_quant_reuse": args.triton_video_qkv_quant_reuse,
        "triton_video_qkv_grouped_mm": args.triton_video_qkv_grouped_mm,
        "triton_video_qk_grouped_mm": args.triton_video_qk_grouped_mm,
        "triton_video_qkv_packed_linear": args.triton_video_qkv_packed_linear,
        "triton_video_qkv_packed_requant": args.triton_video_qkv_packed_requant,
        "triton_video_msa_branch": args.triton_video_msa_branch,
        "triton_video_msa_branch_tokens": args.triton_video_msa_branch_tokens,
        "triton_video_msa_branch_mode": args.triton_video_msa_branch_mode,
        "allow_quality_risk_msa_bf16_out": args.allow_quality_risk_msa_bf16_out,
        "triton_video_msa_branch_profile": args.triton_video_msa_branch_profile,
        "triton_video_msa_branch_stats": video_msa_branch_stats,
        "uniform_timestep_adaln": args.uniform_timestep_adaln,
        "cache_rope_embeddings": args.cache_rope_embeddings,
        "fa3_video_prealloc_output": args.fa3_video_prealloc_output,
        "checkpoint_path": args.checkpoint_path,
        "gemma_root": args.gemma_root,
        "spd_enabled": args.spd,
        "spd_scale": args.spd_scale,
        "spd_transition_step": args.spd_transition_step,
        "spd_mid_scale": args.spd_mid_scale,
        "spd_mid_transition_step": args.spd_mid_transition_step,
        "spd_transform": args.spd_transform,
        "spd_taper": args.spd_taper,
        "spd_initial_dct_downscale": args.spd_initial_dct_downscale,
        "spd_highfreq_noise": args.spd_highfreq_noise,
        "spd_highfreq_source": args.spd_highfreq_source,
        "residual_cache_mode": args.residual_cache_mode,
        "residual_cache_threshold": args.residual_cache_threshold,
        "residual_cache_max_skips": args.residual_cache_max_skips,
        "residual_cache_retention_ratio": args.residual_cache_retention_ratio,
        "residual_cache_metric_element_stride": args.residual_cache_metric_element_stride,
        "serving_shape": {
            "width": BUCKETS[args.bucket].final_width,
            "height": BUCKETS[args.bucket].final_height,
            "frames": BUCKETS[args.bucket].final_frames,
            "fps": BUCKETS[args.bucket].final_fps,
        },
        "live_shape": {
            "width": LIVE_BUCKET.final_width,
            "height": LIVE_BUCKET.final_height,
            "frames": LIVE_BUCKET.final_frames,
            "fps": LIVE_BUCKET.final_fps,
            "internal_frames": LIVE_BUCKET.frames,
        },
        "live_base_shape": {
            "width": LIVE_BASE_BUCKET.final_width,
            "height": LIVE_BASE_BUCKET.final_height,
            "frames": LIVE_BASE_BUCKET.final_frames,
            "fps": LIVE_BASE_BUCKET.final_fps,
            "internal_frames": LIVE_BASE_BUCKET.frames,
        },
        "live_overlap_shape": {
            "width": LIVE_OVERLAP_BUCKET.final_width,
            "height": LIVE_OVERLAP_BUCKET.final_height,
            "frames": LIVE_OVERLAP_BUCKET.final_frames,
            "fps": LIVE_OVERLAP_BUCKET.final_fps,
            "internal_frames": LIVE_OVERLAP_BUCKET.frames,
        },
        "live_overlap3_shape": {
            "width": LIVE_OVERLAP3_BUCKET.final_width,
            "height": LIVE_OVERLAP3_BUCKET.final_height,
            "frames": LIVE_OVERLAP3_BUCKET.final_frames,
            "fps": LIVE_OVERLAP3_BUCKET.final_fps,
            "internal_frames": LIVE_OVERLAP3_BUCKET.frames,
        },
        "rolling_2s_shape": {
            "width": ROLLING_2S_BUCKET.final_width,
            "height": ROLLING_2S_BUCKET.final_height,
            "frames": ROLLING_2S_BUCKET.final_frames,
            "fps": ROLLING_2S_BUCKET.final_fps,
        },
        "rolling_4s_shape": {
            "width": ROLLING_4S_BUCKET.final_width,
            "height": ROLLING_4S_BUCKET.final_height,
            "frames": ROLLING_4S_BUCKET.final_frames,
            "fps": ROLLING_4S_BUCKET.final_fps,
        },
        "premium_shape": {
            "width": PRODUCTION_BUCKET.final_width,
            "height": PRODUCTION_BUCKET.final_height,
            "frames": PRODUCTION_BUCKET.final_frames,
            "fps": PRODUCTION_BUCKET.final_fps,
        },
        "load_s": _engine.load_s if _engine is not None else None,
    }


@app.post("/generate", response_model=WorkerGenerateResponse)
def generate(req: GenerateRequest) -> WorkerGenerateResponse:
    job_id = req.job_id or uuid.uuid4().hex[:12]
    job_dir = (OUTPUT_DIR / job_id).resolve()
    with _request_lock:
        return get_engine().generate(req, job_dir)


@app.get("/streams/{job_id}")
def stream_detached_encode(job_id: str) -> StreamingResponse:
    state = _wait_for_background_encode(job_id, timeout_s=120.0)
    return StreamingResponse(_follow_background_file(state), media_type="video/mp4")


def _close_engine() -> None:
    if _engine is not None:
        _engine.close()


atexit.register(_close_engine)


def main() -> None:
    port = int(os.getenv("PORT", "9000"))
    uvicorn.run("ltx_serve.remote_official_worker:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
