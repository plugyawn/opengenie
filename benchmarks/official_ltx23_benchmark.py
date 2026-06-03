from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
import io
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks.diffusers_ltx23_benchmark import DEFAULT_PROMPT
from ltx_serve.buckets import BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark official LTX-2.3 source components.")
    parser.add_argument("--bucket", default="premium_25fps_5s_v1", choices=sorted(BUCKETS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--image-conditioning-path",
        default=None,
        help="Optional image path to condition the generated video on, typically the last frame of a prior clip.",
    )
    parser.add_argument(
        "--image-conditioning-frame-idx",
        type=int,
        default=0,
        help="Output frame index to bind the conditioning image to. Use 0 for a clean continuation start.",
    )
    parser.add_argument("--image-conditioning-strength", type=float, default=1.0)
    parser.add_argument("--image-conditioning-crf", type=int, default=33)
    parser.add_argument(
        "--image-conditioning-spec",
        action="append",
        default=[],
        metavar="PATH:FRAME:STRENGTH[:CRF]",
        help="Repeatable image conditioning spec for overlap continuation.",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--output-dir", default="outputs/official_ltx23")
    parser.add_argument("--quantization", choices=["none", "fp8-cast", "fp8-scaled-mm"], default="none")
    parser.add_argument(
        "--attention",
        choices=[
            "h100-fa3",
            "h100-fa4",
            "automatic",
            "flash-attn-3",
            "flash-attn-4",
            "sdpa-flash",
            "sdpa-cudnn",
            "sdpa-efficient",
            "sdpa-math",
        ],
        default="h100-fa3",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--spd",
        action="store_true",
        help="Lab-only Spectral Progressive Diffusion path: early steps at lower spatial latent resolution, then DCT-expand to the target bucket.",
    )
    parser.add_argument(
        "--spd-scale",
        type=float,
        default=0.5,
        help="Spatial pixel scale for early SPD denoise. The resulting low-res dimensions are rounded to multiples of 32.",
    )
    parser.add_argument(
        "--spd-transition-step",
        type=int,
        default=4,
        help="Number of initial denoise steps to run at --spd-scale before expanding to the target resolution.",
    )
    parser.add_argument(
        "--spd-mid-scale",
        type=float,
        default=None,
        help="Optional intermediate SPD spatial scale before full resolution, e.g. 0.75 for 0.5 -> 0.75 -> 1.0.",
    )
    parser.add_argument(
        "--spd-mid-transition-step",
        type=int,
        default=None,
        help="Denoise step at which --spd-mid-scale expands to full resolution. Defaults near 70% of the schedule.",
    )
    parser.add_argument("--spd-transform", choices=["dct"], default="dct")
    parser.add_argument(
        "--spd-taper",
        type=int,
        default=8,
        help="DCT-bin cosine taper width at preserved/new coefficient boundaries for SPD transitions.",
    )
    parser.add_argument(
        "--spd-initial-dct-downscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build full-resolution initial noise and DCT-downscale it to the first SPD stage.",
    )
    parser.add_argument(
        "--spd-highfreq-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill newly exposed high-frequency DCT coefficients with sigma-scaled Gaussian noise at the transition.",
    )
    parser.add_argument(
        "--spd-highfreq-source",
        choices=["fresh", "initial"],
        default="fresh",
        help=(
            "Source for newly exposed SPD high-frequency DCT coefficients. "
            "'fresh' matches the current paper-style Gaussian fill; 'initial' reuses the same full-resolution "
            "initial-noise carrier that was DCT-downscaled into the first SPD stage."
        ),
    )
    parser.add_argument(
        "--residual-cache-mode",
        choices=["off", "calibrate", "teacache", "magcache", "force-skip"],
        default="off",
        help=(
            "Lab-only whole-denoiser residual cache. 'calibrate' records metrics without skipping; "
            "'teacache' uses accumulated input-change distance; 'magcache' uses calibrated magnitude ratios; "
            "'force-skip' reuses the previous residual at explicit global step indices."
        ),
    )
    parser.add_argument(
        "--allow-quality-risk-residual-cache",
        action="store_true",
        help="Allow residual-cache modes that intentionally change same-seed latents.",
    )
    parser.add_argument(
        "--residual-cache-threshold",
        type=float,
        default=0.06,
        help="Accumulated TeaCache/MagCache error threshold. Higher values skip more and risk more drift.",
    )
    parser.add_argument(
        "--residual-cache-max-skips",
        type=int,
        default=1,
        help="Maximum consecutive denoiser calls that may reuse the cached residual.",
    )
    parser.add_argument(
        "--residual-cache-retention-ratio",
        type=float,
        default=0.125,
        help="Initial fraction of denoise calls that must compute before adaptive cache skipping is allowed.",
    )
    parser.add_argument(
        "--residual-cache-force-skip-steps",
        default="",
        help="Comma-separated global denoise step indices to skip for --residual-cache-mode=force-skip.",
    )
    parser.add_argument(
        "--residual-cache-mag-ratios",
        default="",
        help="Comma-separated MagCache ratios or a JSON file containing a list or {mag_ratios: [...]} for magcache mode.",
    )
    parser.add_argument(
        "--residual-cache-metric-element-stride",
        type=int,
        default=1,
        help="Stride over flattened latent elements for adaptive cache metrics. 1 is exact over the latent input.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Maximum batch forwarded through the transformer. Defaults to --batch-size for true microbatch timing.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", choices=["none", "true", "false"], default="none")
    parser.add_argument("--compile-dynamic-marking", action="store_true")
    parser.add_argument("--compile-trace-fp8-linear", action="store_true")
    parser.add_argument("--triton-adazero", action="store_true")
    parser.add_argument("--triton-fp8-quant", action="store_true")
    parser.add_argument(
        "--fp8-scaled-mm-bias-epilogue",
        action="store_true",
        help="Pass FP8Linear bias into torch._scaled_mm instead of launching a separate BF16 add kernel.",
    )
    parser.add_argument(
        "--triton-fp8-bias-add",
        action="store_true",
        help="Use a separate exact Triton BF16 broadcast add for FP8Linear bias; does not change scaled-mm accumulation.",
    )
    parser.add_argument(
        "--triton-ffn-gelu-fp8-quant",
        action="store_true",
        help="Fuse FFN GELU(tanh) hidden activation with the following FP8Linear input quantization.",
    )
    parser.add_argument(
        "--triton-cross-attn-adaln",
        action="store_true",
        help="Use the fused AdaZero kernel for text cross-attention AdaLN query normalization.",
    )
    parser.add_argument("--triton-video-preattention", action="store_true")
    parser.add_argument("--triton-video-preattention-checks", type=int, default=0)
    parser.add_argument(
        "--triton-video-preattention-mode",
        choices=["dual", "separate"],
        default="dual",
        help="Use the dual-output Q/K Triton preattention kernel or the old two-launch separate path.",
    )
    parser.add_argument(
        "--triton-video-ada-values",
        action="store_true",
        help="Fuse video MSA/MLP Ada table+timestep materialization into exact Triton kernels.",
    )
    parser.add_argument(
        "--triton-audio-ada-values",
        action="store_true",
        help="Fuse audio MSA/MLP Ada table+timestep materialization into exact Triton kernels.",
    )
    parser.add_argument(
        "--triton-video-text-adaln",
        action="store_true",
        help="Fuse video text-cross-attention AdaLN affine/gate materialization without changing RMSNorm.",
    )
    parser.add_argument(
        "--triton-video-text-context-adaln",
        action="store_true",
        help="Fuse prompt-context Ada affine inside video text cross-attention.",
    )
    parser.add_argument(
        "--uniform-timestep-adaln",
        action="store_true",
        help="For pure T2V full-denoise requests, embed the uniform timestep once and expand it across tokens.",
    )
    parser.add_argument(
        "--cache-rope-embeddings",
        action="store_true",
        help="Cache deterministic RoPE positional embeddings across denoise steps for fixed-shape pure T2V runs.",
    )
    parser.add_argument(
        "--triton-video-out-bias-residual",
        action="store_true",
        help="Bypass video attention residual-output FP8Linear bias adds and fuse bias+gate+residual exactly.",
    )
    parser.add_argument(
        "--triton-video-ffn-out-bias-residual",
        action="store_true",
        help="Experimental: extend video output-bias residual fusion to FFN output projections.",
    )
    parser.add_argument(
        "--triton-video-qk-bias-preattention",
        action="store_true",
        help="Bypass video self-attention Q/K bias adds and fold bias into exact Q/K RMSNorm+RoPE.",
    )
    parser.add_argument(
        "--triton-video-qkv-quant-reuse",
        action="store_true",
        help="Reuse one exact FP8 input quantization for video self-attention Q/K/V projections when scales match.",
    )
    parser.add_argument(
        "--triton-video-qkv-packed-linear",
        action="store_true",
        help="Experimental exact path: run video self-attention Q/K/V projections as one packed FP8 scaled-mm when scales match.",
    )
    parser.add_argument(
        "--triton-video-qkv-grouped-mm",
        action="store_true",
        help="Experimental exact path: run video self-attention Q/K/V projections with grouped FP8 scaled-mm and separate weight scales.",
    )
    parser.add_argument(
        "--triton-video-qkv-grouped-mm-checks",
        type=int,
        default=0,
        help="For the first N grouped-mm video QKV calls, compare Q/K/V projection outputs against the original linears.",
    )
    parser.add_argument(
        "--triton-video-qk-grouped-mm",
        action="store_true",
        help="Experimental exact path: run video self-attention Q/K projections with grouped FP8 scaled-mm and separate weight scales; V stays on the ordinary FP8Linear path.",
    )
    parser.add_argument(
        "--triton-video-qk-grouped-mm-checks",
        type=int,
        default=0,
        help="For the first N grouped-mm video Q/K calls, compare Q/K projection outputs against the original linears.",
    )
    parser.add_argument(
        "--triton-video-qkv-packed-requant",
        action="store_true",
        help="Experimental quality-risk path: repack Q/K/V weights to one common FP8 weight scale and run one packed scaled-mm.",
    )
    parser.add_argument(
        "--triton-video-qkv-packed-requant-checks",
        type=int,
        default=0,
        help="For the first N packed-requant video QKV calls, compare Q/K/V projection outputs against the original linears.",
    )
    parser.add_argument(
        "--allow-quality-risk-packed-qkv-requant",
        action="store_true",
        help="Allow the packed-requant QKV path despite changed FP8 weight quantization.",
    )
    parser.add_argument(
        "--triton-video-msa-branch",
        action="store_true",
        help="Route video self-attention through one exact shape-gated MSA branch and report branch/fallback counts.",
    )
    parser.add_argument(
        "--triton-video-msa-branch-tokens",
        default="",
        help="Optional comma-separated token counts allowed for --triton-video-msa-branch, e.g. 4032,4368.",
    )
    parser.add_argument(
        "--triton-video-msa-branch-mode",
        choices=("generic", "direct", "direct_bf16_out", "direct_qkvpacked"),
        default="generic",
        help=(
            "MSA branch implementation. 'direct' inlines the exact video self-attention subpath; "
            "'direct_qkvpacked' packs Q/K/V inside the preattention kernel and calls FA3 qkvpacked; "
            "'direct_bf16_out' also splits BF16 edge-block output-projection bias inside the branch."
        ),
    )
    parser.add_argument(
        "--triton-video-msa-branch-profile",
        action="store_true",
        help="When the exact video-MSA branch is enabled, collect CUDA-event timings for its internal phases.",
    )
    parser.add_argument(
        "--triton-video-gate-mul",
        action="store_true",
        help=(
            "Experimental exact candidate for video-MSA branch: keep PyTorch gate logits/sigmoid, "
            "then apply per-head gates with a Triton BF16 multiply kernel."
        ),
    )
    parser.add_argument(
        "--allow-quality-risk-msa-bf16-out",
        action="store_true",
        help="Allow the direct_bf16_out MSA branch mode despite known same-seed latent drift.",
    )
    parser.add_argument("--torch-addcmul-residuals", action="store_true")
    parser.add_argument("--triton-residual-gate", action="store_true")
    parser.add_argument(
        "--triton-simple-residual-gate",
        action="store_true",
        help="Fuse only simple self-attn/FFN x + y * gate residuals, leaving cross-attn multiply order unchanged.",
    )
    parser.add_argument(
        "--allow-quality-risk-pointwise-fusions",
        action="store_true",
        help="Allow pointwise fusion experiments that failed same-seed latent parity gates.",
    )
    parser.add_argument("--hot-prompt-encoder", action="store_true")
    parser.add_argument("--hot-decoders", action="store_true")
    parser.add_argument(
        "--prompt-inside-runs",
        action="store_true",
        help="Encode the prompt inside each measured request and include it in runtime_s.",
    )
    parser.add_argument("--fa3-num-splits", type=int, default=1)
    parser.add_argument("--fa3-attention-chunk", type=int, default=0)
    parser.add_argument("--fa3-sm-margin", type=int, default=0)
    parser.add_argument(
        "--fa3-video-window-left",
        type=int,
        default=-1,
        help="Experimental quality-risk path: local left window for video self-attention only.",
    )
    parser.add_argument(
        "--fa3-video-window-right",
        type=int,
        default=-1,
        help="Experimental quality-risk path: local right window for video self-attention only.",
    )
    parser.add_argument(
        "--fa3-video-window-checks",
        type=int,
        default=0,
        help="For the first N exact video self-attention calls, compare windowed FA3 output against dense BF16 FA3.",
    )
    parser.add_argument(
        "--allow-quality-risk-windowed-attention",
        action="store_true",
        help="Allow experimental local-window video self-attention despite changing dense attention semantics.",
    )
    parser.add_argument("--fa3-video-fp8-attention", action="store_true")
    parser.add_argument(
        "--fa3-video-prealloc-output",
        action="store_true",
        help="Experimental exact path: call the low-level FA3 op with a reusable output tensor for video self-attention.",
    )
    parser.add_argument(
        "--fa3-video-fp8-components",
        choices=["q", "k", "v", "qk", "qv", "kv", "qkv"],
        default="qkv",
        help="Experimental subset of exact video self-attention Q/K/V tensors to quantize before FA3.",
    )
    parser.add_argument(
        "--allow-quality-risk-fp8-video-attention",
        action="store_true",
        help="Allow experimental FP8 video self-attention despite known real-tensor error versus BF16 FA3.",
    )
    parser.add_argument(
        "--fa3-video-fp8-checks",
        type=int,
        default=0,
        help="For the first N exact video self-attention calls, compare FP8 FA3 output against BF16 FA3.",
    )
    parser.add_argument("--attention-shape-profile", action="store_true")
    parser.add_argument(
        "--block-profile",
        action="store_true",
        help="Record CUDA event timings for each BasicAVTransformerBlock during measured denoise runs.",
    )
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--decode-video-tiling", choices=["none", "default"], default="none")
    parser.add_argument("--decode-chunk-profile", action="store_true")
    parser.add_argument("--encode-video-stream", action="store_true")
    parser.add_argument("--encode-codec", choices=["h264_nvenc", "libx264"], default="h264_nvenc")
    parser.add_argument(
        "--encode-container",
        choices=["mpegts", "frag-mp4"],
        default="mpegts",
        help="Streaming container for --encode-video-stream. frag-mp4 is browser-playable through <video>.",
    )
    parser.add_argument(
        "--encode-feed-mode",
        choices=["bulk", "frame-stream"],
        default="bulk",
        help="How to feed raw frames into ffmpeg. bulk preserves the original all-frames materialization probe.",
    )
    parser.add_argument(
        "--encode-frame-batch",
        type=int,
        default=1,
        help="Number of frames per write when --encode-feed-mode=frame-stream.",
    )
    parser.add_argument(
        "--encode-start-frame",
        type=int,
        default=0,
        help="Drop this many decoded frames from the start before feeding ffmpeg.",
    )
    parser.add_argument(
        "--encode-audio-trim-start-s",
        type=float,
        default=None,
        help="Drop this many generated-audio seconds before muxing; defaults to --encode-start-frame / fps.",
    )
    parser.add_argument("--encode-x264-preset", default="ultrafast")
    parser.add_argument("--encode-x264-crf", default="19")
    parser.add_argument(
        "--encode-x264-params",
        default="",
        help="Optional libx264 -x264-params string for latency sweeps.",
    )
    parser.add_argument(
        "--encode-threads",
        type=int,
        default=0,
        help="ffmpeg encoder thread count. 0 keeps ffmpeg/libx264 defaults.",
    )
    parser.add_argument(
        "--encode-low-latency-mux",
        action="store_true",
        help="Add low-latency muxer flushing flags to the stream probe.",
    )
    parser.add_argument(
        "--encode-movflags",
        default="frag_keyframe+empty_moov+default_base_moof",
        help="movflags used for --encode-container frag-mp4.",
    )
    parser.add_argument(
        "--encode-prestart",
        action="store_true",
        help="Start ffmpeg before video decode so stream header/startup work overlaps VAE decode.",
    )
    parser.add_argument(
        "--encode-audio-pipe",
        action="store_true",
        help="When prestarting AV encode, attach generated audio through a FIFO so ffmpeg can start before audio decode completes.",
    )
    parser.add_argument(
        "--decode-audio-overlap",
        action="store_true",
        help="Decode generated audio on a background CUDA stream while video decode runs.",
    )
    parser.add_argument(
        "--allow-decode-audio-overlap-oom-risk",
        action="store_true",
        help="Allow the rejected overlapping audio/video decode experiment despite observed H100 OOM risk.",
    )
    parser.add_argument(
        "--sampler-progress",
        action="store_true",
        help="Keep tqdm progress bars inside the denoise sampler. Disabled by default for serving/benchmark latency.",
    )
    parser.add_argument(
        "--encode-mux-audio",
        action="store_true",
        help="Mux the generated audio into the stream probe. Requires --decode-audio.",
    )
    parser.add_argument("--decode-audio", action="store_true")
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-row-limit", type=int, default=40)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    with torch.inference_mode():
        _run(args, torch)


def _run(args: argparse.Namespace, torch: object) -> None:
    if (
        args.fa3_video_window_left != -1
        or args.fa3_video_window_right != -1
    ) and not args.allow_quality_risk_windowed_attention:
        raise RuntimeError(
            "Windowed video self-attention is blocked by default because it changes exact dense attention semantics. "
            "Re-run only for isolated quality-risk experiments with --allow-quality-risk-windowed-attention."
        )
    if args.triton_video_qkv_packed_requant and not args.allow_quality_risk_packed_qkv_requant:
        raise RuntimeError(
            "--triton-video-qkv-packed-requant is blocked by default because it requantizes Q/K/V weights "
            "to one common FP8 scale. Re-run only for isolated experiments with "
            "--allow-quality-risk-packed-qkv-requant."
        )
    if args.fa3_video_fp8_attention and not args.allow_quality_risk_fp8_video_attention:
        raise RuntimeError(
            "--fa3-video-fp8-attention is blocked by default. The real LTX tensor probe showed large "
            "deviation from BF16 FA3, so it is not quality-safe. Re-run only for isolated experiments with "
            "--allow-quality-risk-fp8-video-attention."
        )
    if args.triton_video_msa_branch_mode == "direct_bf16_out" and not args.allow_quality_risk_msa_bf16_out:
        raise RuntimeError(
            "--triton-video-msa-branch-mode=direct_bf16_out is blocked by default because the 1080p15 "
            "probe changed same-seed final latents. Re-run only for isolated experiments with "
            "--allow-quality-risk-msa-bf16-out."
        )
    if (
        args.fp8_scaled_mm_bias_epilogue
        or args.torch_addcmul_residuals
        or args.triton_residual_gate
        or args.triton_cross_attn_adaln
    ) and not args.allow_quality_risk_pointwise_fusions:
        raise RuntimeError(
            "Pointwise fusion experiments are blocked by default. Bias epilogue, addcmul residuals, "
            "the broad Triton residual-gate fusion, and cross-attention AdaLN fusion showed large "
            "same-seed latent drift versus the quality-preserving baseline. Re-run only for isolated "
            "experiments with --allow-quality-risk-pointwise-fusions."
        )
    if args.encode_mux_audio and not args.decode_audio:
        raise RuntimeError("--encode-mux-audio requires --decode-audio")
    if args.encode_mux_audio and not args.encode_video_stream:
        raise RuntimeError("--encode-mux-audio requires --encode-video-stream")
    if args.encode_prestart and not args.encode_video_stream:
        raise RuntimeError("--encode-prestart requires --encode-video-stream")
    if args.encode_prestart and not args.decode_video:
        raise RuntimeError("--encode-prestart requires --decode-video")
    if args.encode_audio_pipe and not (args.encode_prestart and args.encode_mux_audio and args.decode_audio):
        raise RuntimeError("--encode-audio-pipe requires --encode-prestart, --encode-mux-audio, and --decode-audio")
    if args.encode_audio_pipe and args.decode_audio_overlap:
        raise RuntimeError("--encode-audio-pipe is incompatible with --decode-audio-overlap")
    if args.decode_audio_overlap and not args.decode_audio:
        raise RuntimeError("--decode-audio-overlap requires --decode-audio")
    if args.decode_audio_overlap and not args.decode_video:
        raise RuntimeError("--decode-audio-overlap requires --decode-video")
    if args.decode_audio_overlap and not args.allow_decode_audio_overlap_oom_risk:
        raise RuntimeError(
            "--decode-audio-overlap is blocked by default. The 576p/15fps AV probe OOMed at about "
            "79.15GiB used when the audio vocoder and video VAE decoder ran concurrently. Re-run only "
            "for isolated experiments with --allow-decode-audio-overlap-oom-risk."
        )
    if args.image_conditioning_path is not None and args.batch_size != 1:
        raise RuntimeError("--image-conditioning-path currently supports --batch-size 1 only")
    if args.image_conditioning_spec and args.batch_size != 1:
        raise RuntimeError("--image-conditioning-spec currently supports --batch-size 1 only")
    if args.encode_start_frame < 0:
        raise RuntimeError("--encode-start-frame must be >= 0")
    if args.spd:
        if args.steps != 8:
            raise RuntimeError("--spd is currently implemented for the official 8-step distilled schedule only")
        if not (0.0 < args.spd_scale < 1.0):
            raise RuntimeError("--spd-scale must be in (0, 1)")
        if not (0 < args.spd_transition_step < args.steps):
            raise RuntimeError("--spd-transition-step must be between 1 and --steps - 1")
        if args.spd_mid_scale is not None:
            if not (args.spd_scale < args.spd_mid_scale < 1.0):
                raise RuntimeError("--spd-mid-scale must be greater than --spd-scale and less than 1")
            mid_step = args.spd_mid_transition_step
            if mid_step is None:
                mid_step = max(args.spd_transition_step + 1, int(round(args.steps * 0.7)))
            if not (args.spd_transition_step < mid_step < args.steps):
                raise RuntimeError("--spd-mid-transition-step must be greater than --spd-transition-step and less than --steps")
        if args.image_conditioning_path is not None or args.image_conditioning_spec:
            raise RuntimeError("--spd is lab-gated to pure T2V first; image conditioning adds extra tokens to resize")
        if args.spd_highfreq_source == "initial" and not args.spd_initial_dct_downscale:
            raise RuntimeError("--spd-highfreq-source initial requires --spd-initial-dct-downscale")
    if args.residual_cache_mode != "off":
        if not args.allow_quality_risk_residual_cache:
            raise RuntimeError(
                "Residual-cache modes intentionally change same-seed latents and are blocked by default. "
                "Re-run lab probes with --allow-quality-risk-residual-cache."
            )
        if args.steps < 2:
            raise RuntimeError("--residual-cache-mode requires at least two denoise steps")
        if args.residual_cache_threshold < 0.0:
            raise RuntimeError("--residual-cache-threshold must be >= 0")
        if args.residual_cache_max_skips < 0:
            raise RuntimeError("--residual-cache-max-skips must be >= 0")
        if not (0.0 <= args.residual_cache_retention_ratio < 1.0):
            raise RuntimeError("--residual-cache-retention-ratio must be in [0, 1)")
        if args.residual_cache_metric_element_stride < 1:
            raise RuntimeError("--residual-cache-metric-element-stride must be >= 1")
        if args.residual_cache_mode == "force-skip" and not args.residual_cache_force_skip_steps.strip():
            raise RuntimeError("--residual-cache-mode=force-skip requires --residual-cache-force-skip-steps")
        if args.residual_cache_mode == "magcache" and not args.residual_cache_mag_ratios.strip():
            raise RuntimeError("--residual-cache-mode=magcache requires --residual-cache-mag-ratios from calibration")

    bucket = BUCKETS[args.bucket]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    from ltx_pipelines.utils.types import ModalitySpec

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
    _patch_uniform_timestep_adaln(args)
    _patch_rope_embedding_cache(args)
    _patch_triton_video_out_bias_residual(args)
    _patch_triton_video_ffn_out_bias_residual(args)
    _patch_triton_video_qk_bias_preattention(args)
    _patch_triton_video_qkv_quant_reuse(args)
    _patch_triton_video_qkv_grouped_mm(args)
    _patch_triton_video_qk_grouped_mm(args)
    _patch_triton_video_qkv_packed_linear(args)
    _patch_triton_video_qkv_packed_requant(args)
    _patch_triton_video_msa_branch(args)
    _patch_flash_attention3_options(args)
    _patch_attention_shape_profile(args)
    quantization_policy = _build_quantization_policy(args.quantization, args.checkpoint_path)
    compilation_config = _build_compilation_config(args)
    device = torch.device(args.device)

    load_start = time.perf_counter()
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=[],
        quantization=quantization_policy,
        compilation_config=compilation_config,
        device=device,
    )
    _patch_hot_pipeline_blocks(args, pipeline)
    resolved_attention = _resolve_attention(args.attention)
    pipeline.stage = _pin_attention(pipeline.stage, resolved_attention)
    builder_load_s = time.perf_counter() - load_start

    prompt_s = 0.0
    video_context = None
    audio_context = None
    if not args.prompt_inside_runs:
        prompt_s, video_context, audio_context = _encode_prompt_contexts(args, pipeline, torch)
    image_conditioning_s, image_conditionings = _build_image_conditionings(args, pipeline, bucket, torch)

    sigmas = _sigmas(args.steps, DISTILLED_SIGMAS).to(dtype=torch.float32, device=device)
    measured: list[dict[str, Any]] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_start = time.perf_counter()
    with pipeline.stage.model_context() as transformer:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_load_s = time.perf_counter() - model_start
        block_profiler = _install_block_profiler(args, transformer, torch)

        total_runs = args.warmup + args.runs
        profile_outputs: list[dict[str, str]] = []
        for idx in range(total_runs):
            if idx == args.warmup and args.triton_video_msa_branch_profile:
                from ltx_serve.triton_ltx_ops import reset_video_msa_branch_profile_events

                reset_video_msa_branch_profile_events()
            if block_profiler is not None:
                block_profiler.set_enabled(idx >= args.warmup)
            run_prompt_s = 0.0
            run_video_context = video_context
            run_audio_context = audio_context
            if args.prompt_inside_runs:
                run_prompt_s, run_video_context, run_audio_context = _encode_prompt_contexts(args, pipeline, torch)

            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
            noiser = GaussianNoiser(generator=generator)
            spd_stats: dict[str, Any] = {}
            residual_cache_stats: dict[str, Any] = {}

            profiler_context = _profiler_context(args, torch, idx)
            capture_cuda_range = args.cuda_profiler_range and idx >= args.warmup and torch.cuda.is_available()
            if capture_cuda_range:
                torch.cuda.cudart().cudaProfilerStart()
            denoise_start = time.perf_counter()
            try:
                with profiler_context as profiler:
                    video_state, audio_state = _run_stage_batch(
                        stage=pipeline.stage,
                        transformer=transformer,
                        denoiser=_BatchSimpleDenoiser(
                            run_video_context,
                            run_audio_context,
                            args.batch_size,
                            residual_cache_config=_residual_cache_config_from_args(args),
                            residual_cache_stats=residual_cache_stats,
                        ),
                        sigmas=sigmas,
                        noiser=noiser,
                        width=bucket.width,
                        height=bucket.height,
                        frames=bucket.frames,
                        fps=float(bucket.fps),
                        video=ModalitySpec(context=run_video_context, conditionings=image_conditionings),
                        audio=ModalitySpec(context=run_audio_context),
                        batch_size=args.batch_size,
                        max_batch_size=args.max_batch_size or args.batch_size,
                        uniform_timestep_adaln=args.uniform_timestep_adaln,
                        spd_config=_spd_config_from_args(args, bucket) if args.spd else None,
                        spd_generator=generator,
                        spd_stats=spd_stats,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                if capture_cuda_range:
                    torch.cuda.cudart().cudaProfilerStop()
            denoise_s = time.perf_counter() - denoise_start
            if profiler is not None:
                profile_outputs.append(_write_profile(args, torch, profiler, output_dir, bucket.name, idx))

            latent_outputs = _save_latents(args, torch, output_dir, idx, video_state, audio_state) if idx >= args.warmup else {}

            decode_video_s = 0.0
            decode_audio_s = 0.0
            encode_video_s = 0.0
            encode_first_byte_s = None
            encode_bytes = None
            video_decode_chunks: list[dict[str, Any]] = []
            first_video_chunk = None
            decoded_audio = None
            post_denoise_start = time.perf_counter()
            post_denoise_s = None
            first_byte_after_denoise_s = None
            prestarted_encoder = None
            audio_future = None
            audio_executor = None
            audio_future_collected = False

            def _decode_audio_for_run() -> tuple[object, float, _EncoderSession | None]:
                assert audio_state is not None
                decode_audio_start = time.perf_counter()
                if torch.cuda.is_available():
                    stream = torch.cuda.Stream(device=device)
                    with torch.cuda.stream(stream):
                        decoded = pipeline.audio_decoder(audio_state.latent)
                        _ = decoded.waveform.shape if hasattr(decoded, "waveform") else decoded
                    stream.synchronize()
                else:
                    decoded = pipeline.audio_decoder(audio_state.latent)
                    _ = decoded.waveform.shape if hasattr(decoded, "waveform") else decoded
                elapsed = time.perf_counter() - decode_audio_start
                session = None
                if args.encode_prestart and args.encode_video_stream and args.encode_mux_audio:
                    session = _start_video_stream_probe(
                        bucket,
                        output_dir,
                        idx,
                        args.encode_codec,
                        args.encode_container,
                        x264_preset=args.encode_x264_preset,
                        x264_crf=args.encode_x264_crf,
                        x264_params=args.encode_x264_params,
                        encode_threads=args.encode_threads,
                        low_latency_mux=args.encode_low_latency_mux,
                        movflags=args.encode_movflags,
                        audio=decoded,
                        audio_pipe=args.encode_audio_pipe,
                        audio_trim_start_s=_audio_trim_start_s(args, bucket),
                    )
                return decoded, elapsed, session

            def _collect_audio_future() -> None:
                nonlocal decoded_audio, decode_audio_s, prestarted_encoder, audio_future_collected
                if audio_future is None or audio_future_collected:
                    return
                decoded_audio, decode_audio_s, overlap_encoder = audio_future.result()
                audio_future_collected = True
                if overlap_encoder is not None:
                    prestarted_encoder = overlap_encoder

            if args.decode_audio and args.decode_audio_overlap:
                assert audio_state is not None
                audio_executor = ThreadPoolExecutor(max_workers=1)
                audio_future = audio_executor.submit(_decode_audio_for_run)
            if args.encode_prestart and args.encode_video_stream:
                if args.encode_mux_audio and args.encode_audio_pipe:
                    prestarted_encoder = _start_video_stream_probe(
                        bucket,
                        output_dir,
                        idx,
                        args.encode_codec,
                        args.encode_container,
                        x264_preset=args.encode_x264_preset,
                        x264_crf=args.encode_x264_crf,
                        x264_params=args.encode_x264_params,
                        encode_threads=args.encode_threads,
                        low_latency_mux=args.encode_low_latency_mux,
                        movflags=args.encode_movflags,
                        audio_pipe=True,
                        audio_trim_start_s=_audio_trim_start_s(args, bucket),
                    )
                    assert audio_state is not None
                    decode_audio_start = time.perf_counter()
                    decoded_audio = pipeline.audio_decoder(audio_state.latent)
                    _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    decode_audio_s = time.perf_counter() - decode_audio_start
                    _attach_audio_to_encoder(
                        prestarted_encoder,
                        decoded_audio,
                        trim_start_s=_audio_trim_start_s(args, bucket),
                    )
                elif args.encode_mux_audio:
                    if not args.decode_audio_overlap:
                        assert audio_state is not None
                        decode_audio_start = time.perf_counter()
                        decoded_audio = pipeline.audio_decoder(audio_state.latent)
                        _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        decode_audio_s = time.perf_counter() - decode_audio_start
                if prestarted_encoder is None and (not args.encode_mux_audio or not args.decode_audio_overlap):
                    prestarted_encoder = _start_video_stream_probe(
                        bucket,
                        output_dir,
                        idx,
                        args.encode_codec,
                        args.encode_container,
                        x264_preset=args.encode_x264_preset,
                        x264_crf=args.encode_x264_crf,
                        x264_params=args.encode_x264_params,
                        encode_threads=args.encode_threads,
                        low_latency_mux=args.encode_low_latency_mux,
                        movflags=args.encode_movflags,
                        audio=decoded_audio,
                        audio_pipe=args.encode_audio_pipe,
                        audio_trim_start_s=_audio_trim_start_s(args, bucket),
                    )
            if args.decode_video:
                assert video_state is not None
                decode_video_start = time.perf_counter()
                tiling_config = _video_decode_tiling_config(args)
                for chunk_index, chunk in enumerate(
                    pipeline.video_decoder(video_state.latent, tiling_config=tiling_config, generator=generator)
                ):
                    # Force materialization without converting to PIL/CPU video.
                    shape = list(chunk.shape)
                    dtype = str(chunk.dtype)
                    device_name = str(chunk.device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - decode_video_start
                    if args.decode_chunk_profile:
                        video_decode_chunks.append(
                            {
                                "chunk_index": chunk_index,
                                "elapsed_s": elapsed,
                                "shape": shape,
                                "dtype": dtype,
                                "device": device_name,
                            }
                        )
                    if first_video_chunk is None:
                        first_video_chunk = chunk
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_video_s = time.perf_counter() - decode_video_start
            if args.decode_audio and args.encode_mux_audio and args.decode_audio_overlap:
                _collect_audio_future()
            if args.decode_audio and args.encode_mux_audio and decoded_audio is None:
                assert audio_state is not None
                decode_audio_start = time.perf_counter()
                decoded_audio = pipeline.audio_decoder(audio_state.latent)
                _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_audio_s = time.perf_counter() - decode_audio_start
            if args.encode_video_stream:
                if first_video_chunk is None:
                    raise RuntimeError("--encode-video-stream requires --decode-video")
                if prestarted_encoder is not None:
                    encode_stats = _finish_video_stream_probe(
                        prestarted_encoder,
                        first_video_chunk,
                        bucket,
                        feed_mode=args.encode_feed_mode,
                        frame_batch=args.encode_frame_batch,
                        start_frame=args.encode_start_frame,
                    )
                else:
                    encode_stats = _encode_video_stream_probe(
                        first_video_chunk,
                        bucket,
                        output_dir,
                        idx,
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
                        audio_pipe=args.encode_audio_pipe,
                        audio_trim_start_s=_audio_trim_start_s(args, bucket),
                        start_frame=args.encode_start_frame,
                    )
                encode_video_s = encode_stats["encode_video_s"]
                encode_first_byte_s = encode_stats["encode_first_byte_s"]
                if encode_stats.get("encode_first_byte_abs_s") is not None:
                    first_byte_after_denoise_s = max(0.0, encode_stats["encode_first_byte_abs_s"] - post_denoise_start)
                encode_bytes = encode_stats["encode_bytes"]
                encode_output_path = encode_stats["output_path"]
            else:
                encode_output_path = None
            if args.decode_audio_overlap and audio_future is not None:
                _collect_audio_future()
                if audio_executor is not None:
                    audio_executor.shutdown(wait=True)
            if args.decode_audio and decoded_audio is None:
                assert audio_state is not None
                decode_audio_start = time.perf_counter()
                decoded_audio = pipeline.audio_decoder(audio_state.latent)
                _ = decoded_audio.waveform.shape if hasattr(decoded_audio, "waveform") else decoded_audio
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_audio_s = time.perf_counter() - decode_audio_start
            if args.encode_prestart or args.decode_audio_overlap:
                post_denoise_s = time.perf_counter() - post_denoise_start

            serial_first_byte_s = (
                run_prompt_s
                + denoise_s
                + (video_decode_chunks[0]["elapsed_s"] if video_decode_chunks else decode_video_s)
                + (decode_audio_s if args.encode_mux_audio else 0.0)
                + (encode_first_byte_s if encode_first_byte_s is not None else 0.0)
                if args.decode_video
                else None
            )
            serial_runtime_s = run_prompt_s + denoise_s + decode_video_s + decode_audio_s + encode_video_s
            if post_denoise_s is not None and first_byte_after_denoise_s is not None:
                runtime_s = run_prompt_s + denoise_s + post_denoise_s
                time_to_first_video_byte_s = (
                    run_prompt_s + denoise_s + first_byte_after_denoise_s
                    if first_byte_after_denoise_s is not None
                    else serial_first_byte_s
                )
            else:
                runtime_s = serial_runtime_s
                time_to_first_video_byte_s = serial_first_byte_s

            run = {
                "run_index": idx,
                "measured": idx >= args.warmup,
                "batch_size": args.batch_size,
                "prompt_s": run_prompt_s,
                "denoise_s": denoise_s,
                "decode_video_s": decode_video_s,
                "first_video_chunk_s": video_decode_chunks[0]["elapsed_s"] if video_decode_chunks else None,
                "video_decode_chunks": video_decode_chunks if args.decode_chunk_profile else None,
                "encode_video_s": encode_video_s,
                "encode_first_byte_s": encode_first_byte_s,
                "encode_prestart": args.encode_prestart,
                "decode_audio_overlap": args.decode_audio_overlap,
                "post_denoise_s": post_denoise_s,
                "first_byte_after_denoise_s": first_byte_after_denoise_s,
                "encode_bytes": encode_bytes,
                "encode_output_path": encode_output_path,
                "latent_outputs": latent_outputs,
                "spd": spd_stats if args.spd else None,
                "residual_cache": residual_cache_stats if args.residual_cache_mode != "off" else None,
                "decode_audio_s": decode_audio_s,
                "time_to_first_video_byte_s": time_to_first_video_byte_s,
                "runtime_s": runtime_s,
                "realtime_factor": runtime_s / bucket.output_duration_s,
                "effective_runtime_per_clip_s": runtime_s / args.batch_size,
                "clip_realtime_factor": runtime_s / (bucket.output_duration_s * args.batch_size),
                "throughput_clips_per_hour": 3600.0 * args.batch_size / runtime_s,
            }
            print(json.dumps(run), flush=True)
            if run["measured"]:
                measured.append(run)

    attention_shape_profile = _collect_attention_shape_profile(torch) if args.attention_shape_profile else None
    fp8_video_attention_checks = _collect_fp8_video_attention_checks()
    window_video_attention_checks = _collect_window_video_attention_checks()
    video_preattention_checks = _collect_video_preattention_checks(args)
    video_ada_values_calls = _collect_video_ada_values_calls(args)
    audio_ada_values_calls = _collect_audio_ada_values_calls(args)
    video_text_adaln_calls = _collect_video_text_adaln_calls(args)
    uniform_timestep_adaln_calls = _collect_uniform_timestep_adaln_calls(args)
    rope_cache_stats = _collect_rope_embedding_cache_stats(args)
    video_qkv_quant_reuse_calls = _collect_video_qkv_quant_reuse_calls(args)
    video_qkv_grouped_mm_stats = _collect_video_qkv_grouped_mm_stats(args)
    video_qk_grouped_mm_stats = _collect_video_qk_grouped_mm_stats(args)
    video_qkv_packed_linear_stats = _collect_video_qkv_packed_linear_stats(args)
    video_qkv_packed_requant_stats = _collect_video_qkv_packed_requant_stats(args)
    video_msa_branch_stats = _collect_video_msa_branch_stats(args)
    ffn_gelu_fp8_quant_calls = _collect_ffn_gelu_fp8_quant_calls(args)
    cross_attn_adaln_calls = _collect_cross_attn_adaln_calls(args)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    summary = {
        "backend": f"official-ltx2.3:{args.quantization}:steps={args.steps}:compile={args.compile}",
        "attention": args.attention,
        "resolved_attention": resolved_attention,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "compile_backend": args.compile_backend if args.compile else None,
        "compile_fullgraph": args.compile_fullgraph if args.compile else None,
        "compile_dynamic": args.compile_dynamic if args.compile else None,
        "compile_dynamic_marking": args.compile_dynamic_marking if args.compile else None,
        "compile_trace_fp8_linear": args.compile_trace_fp8_linear if args.compile else None,
        "triton_adazero": args.triton_adazero,
        "triton_fp8_quant": args.triton_fp8_quant,
        "fp8_scaled_mm_bias_epilogue": args.fp8_scaled_mm_bias_epilogue,
        "triton_fp8_bias_add": args.triton_fp8_bias_add,
        "triton_ffn_gelu_fp8_quant": args.triton_ffn_gelu_fp8_quant,
        "triton_ffn_gelu_fp8_quant_calls": ffn_gelu_fp8_quant_calls,
        "triton_cross_attn_adaln": args.triton_cross_attn_adaln,
        "triton_cross_attn_adaln_calls": cross_attn_adaln_calls,
        "triton_video_preattention": args.triton_video_preattention,
        "triton_video_preattention_mode": args.triton_video_preattention_mode,
        "triton_video_preattention_checks_requested": args.triton_video_preattention_checks,
        "triton_video_preattention_checks": video_preattention_checks,
        "triton_video_ada_values": args.triton_video_ada_values,
        "triton_video_ada_values_calls": video_ada_values_calls,
        "triton_audio_ada_values": args.triton_audio_ada_values,
        "triton_audio_ada_values_calls": audio_ada_values_calls,
        "triton_video_text_adaln": args.triton_video_text_adaln,
        "triton_video_text_context_adaln": args.triton_video_text_context_adaln,
        "triton_video_text_adaln_calls": video_text_adaln_calls,
        "uniform_timestep_adaln": args.uniform_timestep_adaln,
        "uniform_timestep_adaln_calls": uniform_timestep_adaln_calls,
        "cache_rope_embeddings": args.cache_rope_embeddings,
        "rope_embedding_cache": rope_cache_stats,
        "triton_video_out_bias_residual": args.triton_video_out_bias_residual,
        "triton_video_ffn_out_bias_residual": args.triton_video_ffn_out_bias_residual,
        "triton_video_qk_bias_preattention": args.triton_video_qk_bias_preattention,
        "triton_video_qkv_quant_reuse": args.triton_video_qkv_quant_reuse,
        "triton_video_qkv_quant_reuse_calls": video_qkv_quant_reuse_calls,
        "triton_video_qkv_grouped_mm": args.triton_video_qkv_grouped_mm,
        "triton_video_qkv_grouped_mm_stats": video_qkv_grouped_mm_stats,
        "triton_video_qk_grouped_mm": args.triton_video_qk_grouped_mm,
        "triton_video_qk_grouped_mm_stats": video_qk_grouped_mm_stats,
        "triton_video_qkv_packed_linear": args.triton_video_qkv_packed_linear,
        "triton_video_qkv_packed_linear_stats": video_qkv_packed_linear_stats,
        "triton_video_qkv_packed_requant": args.triton_video_qkv_packed_requant,
        "triton_video_qkv_packed_requant_stats": video_qkv_packed_requant_stats,
        "allow_quality_risk_packed_qkv_requant": args.allow_quality_risk_packed_qkv_requant,
        "triton_video_msa_branch": args.triton_video_msa_branch,
        "triton_video_msa_branch_tokens": _parse_token_counts(args.triton_video_msa_branch_tokens),
        "triton_video_msa_branch_mode": args.triton_video_msa_branch_mode,
        "triton_video_msa_branch_profile": args.triton_video_msa_branch_profile,
        "triton_video_gate_mul": args.triton_video_gate_mul,
        "triton_video_msa_branch_stats": video_msa_branch_stats,
        "torch_addcmul_residuals": args.torch_addcmul_residuals,
        "triton_residual_gate": args.triton_residual_gate,
        "triton_simple_residual_gate": args.triton_simple_residual_gate,
        "allow_quality_risk_pointwise_fusions": args.allow_quality_risk_pointwise_fusions,
        "hot_prompt_encoder": args.hot_prompt_encoder,
        "hot_decoders": args.hot_decoders,
        "prompt_inside_runs": args.prompt_inside_runs,
        "image_conditioning_path": args.image_conditioning_path,
        "image_conditioning_frame_idx": args.image_conditioning_frame_idx if args.image_conditioning_path else None,
        "image_conditioning_strength": args.image_conditioning_strength if args.image_conditioning_path else None,
        "image_conditioning_spec": args.image_conditioning_spec,
        "image_conditioning_s": image_conditioning_s,
        "fa3_num_splits": args.fa3_num_splits,
        "fa3_attention_chunk": args.fa3_attention_chunk,
        "fa3_sm_margin": args.fa3_sm_margin,
        "fa3_video_window_left": args.fa3_video_window_left,
        "fa3_video_window_right": args.fa3_video_window_right,
        "fa3_video_window_checks_requested": args.fa3_video_window_checks,
        "fa3_video_window_checks": window_video_attention_checks,
        "allow_quality_risk_windowed_attention": args.allow_quality_risk_windowed_attention,
        "fa3_video_fp8_attention": args.fa3_video_fp8_attention,
        "fa3_video_prealloc_output": args.fa3_video_prealloc_output,
        "fa3_video_fp8_components": args.fa3_video_fp8_components,
        "allow_quality_risk_fp8_video_attention": args.allow_quality_risk_fp8_video_attention,
        "fa3_video_fp8_checks_requested": args.fa3_video_fp8_checks,
        "fa3_video_fp8_checks": fp8_video_attention_checks,
        "attention_shape_profile_enabled": args.attention_shape_profile,
        "attention_shape_profile": attention_shape_profile,
        "block_profile_enabled": args.block_profile,
        "block_profile": block_profiler.summary() if block_profiler is not None else None,
        "decode_video_tiling": args.decode_video_tiling,
        "decode_chunk_profile_enabled": args.decode_chunk_profile,
        "encode_video_stream": args.encode_video_stream,
        "encode_codec": args.encode_codec if args.encode_video_stream else None,
        "encode_container": args.encode_container if args.encode_video_stream else None,
        "encode_feed_mode": args.encode_feed_mode if args.encode_video_stream else None,
        "encode_frame_batch": args.encode_frame_batch if args.encode_video_stream else None,
        "encode_start_frame": args.encode_start_frame if args.encode_video_stream else None,
        "encode_audio_trim_start_s": _audio_trim_start_s(args, bucket) if args.encode_video_stream else None,
        "encode_x264_preset": args.encode_x264_preset if args.encode_video_stream else None,
        "encode_x264_crf": args.encode_x264_crf if args.encode_video_stream else None,
        "encode_x264_params": args.encode_x264_params if args.encode_video_stream else None,
        "encode_threads": args.encode_threads if args.encode_video_stream else None,
        "encode_low_latency_mux": args.encode_low_latency_mux if args.encode_video_stream else None,
        "encode_movflags": args.encode_movflags if args.encode_video_stream else None,
        "encode_prestart": args.encode_prestart if args.encode_video_stream else None,
        "encode_audio_pipe": args.encode_audio_pipe if args.encode_video_stream else None,
        "decode_audio_overlap": args.decode_audio_overlap,
        "allow_decode_audio_overlap_oom_risk": args.allow_decode_audio_overlap_oom_risk,
        "sampler_progress": args.sampler_progress,
        "encode_mux_audio": args.encode_mux_audio if args.encode_video_stream else None,
        "save_latents": args.save_latents,
        "spd_enabled": args.spd,
        "spd_scale": args.spd_scale if args.spd else None,
        "spd_transition_step": args.spd_transition_step if args.spd else None,
        "spd_mid_scale": args.spd_mid_scale if args.spd else None,
        "spd_mid_transition_step": (
            _spd_config_from_args(args, bucket).transition_steps[1]
            if args.spd and args.spd_mid_scale is not None
            else None
        ),
        "spd_transform": args.spd_transform if args.spd else None,
        "spd_taper": args.spd_taper if args.spd else None,
        "spd_initial_dct_downscale": args.spd_initial_dct_downscale if args.spd else None,
        "spd_highfreq_noise": args.spd_highfreq_noise if args.spd else None,
        "spd_highfreq_source": args.spd_highfreq_source if args.spd else None,
        "spd_low_shape": _spd_config_from_args(args, bucket).low_shape if args.spd else None,
        "residual_cache_mode": args.residual_cache_mode,
        "allow_quality_risk_residual_cache": args.allow_quality_risk_residual_cache,
        "residual_cache_threshold": args.residual_cache_threshold if args.residual_cache_mode != "off" else None,
        "residual_cache_max_skips": args.residual_cache_max_skips if args.residual_cache_mode != "off" else None,
        "residual_cache_retention_ratio": (
            args.residual_cache_retention_ratio if args.residual_cache_mode != "off" else None
        ),
        "residual_cache_force_skip_steps": (
            _parse_int_set(args.residual_cache_force_skip_steps) if args.residual_cache_mode == "force-skip" else None
        ),
        "residual_cache_mag_ratios": (
            _parse_mag_ratios(args.residual_cache_mag_ratios) if args.residual_cache_mode == "magcache" else None
        ),
        "residual_cache_metric_element_stride": (
            args.residual_cache_metric_element_stride if args.residual_cache_mode != "off" else None
        ),
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size or args.batch_size,
        "bucket": bucket.name,
        "checkpoint_path": args.checkpoint_path,
        "gemma_root": args.gemma_root,
        "target_output_duration_s": bucket.output_duration_s,
        "builder_load_s": builder_load_s,
        "external_prompt_s": prompt_s,
        "model_load_s": model_load_s,
        "trtllm_scaled_mm_usable": trtllm_scaled_mm_usable(),
        "peak_vram_gb": peak_vram_gb,
        "profile_outputs": profile_outputs,
        "runs": measured,
        "best_runtime_s": min((r["runtime_s"] for r in measured), default=None),
        "best_realtime_factor": min((r["realtime_factor"] for r in measured), default=None),
        "best_effective_runtime_per_clip_s": min(
            (r["effective_runtime_per_clip_s"] for r in measured), default=None
        ),
        "best_clip_realtime_factor": min((r["clip_realtime_factor"] for r in measured), default=None),
        "best_throughput_clips_per_hour": max((r["throughput_clips_per_hour"] for r in measured), default=None),
    }
    summary_path = output_dir / f"{bucket.name}_{args.quantization}_steps{args.steps}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2), flush=True)


@dataclass(frozen=True)
class _SPDStage:
    scale: float
    height: int
    width: int
    shape: dict[str, int | float | str | bool]


@dataclass(frozen=True)
class _SPDConfig:
    transition_step: int
    scale: float
    transition_steps: tuple[int, ...]
    stages: tuple[_SPDStage, ...]
    transform: str
    taper: int
    initial_dct_downscale: bool
    highfreq_noise: bool
    highfreq_source: str
    low_height: int
    low_width: int
    low_shape: dict[str, int | float | str | bool]


_DCT_MATRIX_CACHE: dict[tuple[int, str, str], object] = {}


def _spd_stage_from_scale(bucket: object, scale: float) -> _SPDStage:
    height = max(32, int(round(float(bucket.height) * float(scale) / 32.0)) * 32)
    width = max(32, int(round(float(bucket.width) * float(scale) / 32.0)) * 32)
    if height >= int(bucket.height) or width >= int(bucket.width):
        raise RuntimeError(
            f"SPD scale {scale} produced non-progressive shape {width}x{height} "
            f"for target {bucket.width}x{bucket.height}"
        )
    return _SPDStage(
        scale=float(scale),
        height=height,
        width=width,
        shape={
            "scale": float(scale),
            "width": width,
            "height": height,
            "frames": int(bucket.frames),
            "fps": float(bucket.fps),
            "latent_width": width // 32,
            "latent_height": height // 32,
            "latent_frames": 1 + (int(bucket.frames) - 1) // 8,
        },
    )


def _spd_config_from_args(args: argparse.Namespace, bucket: object) -> _SPDConfig:
    highfreq_source = str(getattr(args, "spd_highfreq_source", "fresh"))
    if highfreq_source not in {"fresh", "initial"}:
        raise RuntimeError(f"unsupported SPD high-frequency source: {highfreq_source}")
    if highfreq_source == "initial" and not bool(args.spd_initial_dct_downscale):
        raise RuntimeError("SPD high-frequency source 'initial' requires initial DCT downscale")
    stages = [_spd_stage_from_scale(bucket, float(args.spd_scale))]
    transition_steps = [int(args.spd_transition_step)]
    if args.spd_mid_scale is not None:
        stages.append(_spd_stage_from_scale(bucket, float(args.spd_mid_scale)))
        mid_step = args.spd_mid_transition_step
        if mid_step is None:
            mid_step = max(int(args.spd_transition_step) + 1, int(round(int(args.steps) * 0.7)))
        transition_steps.append(int(mid_step))
    if len(set(transition_steps)) != len(transition_steps) or list(transition_steps) != sorted(transition_steps):
        raise RuntimeError(f"SPD transition steps must be strictly increasing, got {transition_steps}")
    if len(stages) == 2 and not (stages[0].height < stages[1].height <= int(bucket.height) and stages[0].width < stages[1].width <= int(bucket.width)):
        raise RuntimeError("SPD intermediate stage must have strictly larger spatial dimensions than the initial stage")
    low_stage = stages[0]
    return _SPDConfig(
        transition_step=int(args.spd_transition_step),
        scale=float(args.spd_scale),
        transition_steps=tuple(transition_steps),
        stages=tuple(stages),
        transform=str(args.spd_transform),
        taper=max(0, int(args.spd_taper)),
        initial_dct_downscale=bool(args.spd_initial_dct_downscale),
        highfreq_noise=bool(args.spd_highfreq_noise),
        highfreq_source=highfreq_source,
        low_height=low_stage.height,
        low_width=low_stage.width,
        low_shape=dict(low_stage.shape),
    )


def _dct_matrix(torch: object, n: int, *, device: object, dtype: object) -> object:
    key = (n, str(device), str(dtype))
    cached = _DCT_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    arange = torch.arange(n, device=device, dtype=dtype)
    k = arange[:, None]
    x = arange[None, :]
    matrix = torch.cos(torch.pi / float(n) * (x + 0.5) * k)
    matrix[0, :] *= (1.0 / float(n)) ** 0.5
    if n > 1:
        matrix[1:, :] *= (2.0 / float(n)) ** 0.5
    _DCT_MATRIX_CACHE[key] = matrix
    return matrix


def _dct2(torch: object, x: object) -> object:
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    dtype = torch.float32
    device = x.device
    c_h = _dct_matrix(torch, h, device=device, dtype=dtype)
    c_w = _dct_matrix(torch, w, device=device, dtype=dtype)
    flat = x.to(dtype).reshape(-1, h, w)
    return torch.matmul(torch.matmul(c_h, flat), c_w.transpose(0, 1)).reshape(*x.shape[:-2], h, w)


def _dct2_partial(torch: object, x: object, h_keep: int, w_keep: int) -> object:
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    dtype = torch.float32
    device = x.device
    c_h = _dct_matrix(torch, h, device=device, dtype=dtype)[:h_keep]
    c_w = _dct_matrix(torch, w, device=device, dtype=dtype)[:w_keep]
    flat = x.to(dtype).reshape(-1, h, w)
    return torch.matmul(torch.matmul(c_h, flat), c_w.transpose(0, 1)).reshape(*x.shape[:-2], h_keep, w_keep)


def _idct2(torch: object, coeff: object) -> object:
    h = int(coeff.shape[-2])
    w = int(coeff.shape[-1])
    dtype = torch.float32
    device = coeff.device
    c_h = _dct_matrix(torch, h, device=device, dtype=dtype)
    c_w = _dct_matrix(torch, w, device=device, dtype=dtype)
    flat = coeff.to(dtype).reshape(-1, h, w)
    return torch.matmul(torch.matmul(c_h.transpose(0, 1), flat), c_w).reshape(*coeff.shape[:-2], h, w)


def _spd_cos_taper_1d(torch: object, n: int, kept: int, taper: int, *, device: object, dtype: object) -> object:
    weights = torch.ones(n, device=device, dtype=dtype)
    taper = max(0, min(int(taper), int(kept)))
    if taper > 0:
        ramp_start = int(kept) - taper
        idx = torch.arange(taper, device=device, dtype=dtype)
        weights[ramp_start:kept] = 0.5 * (1.0 + torch.cos(torch.pi * (idx + 1.0) / float(taper)))
    if kept < n:
        weights[kept:] = 0.0
    return weights


def _spd_preserve_mask_2d(
    torch: object,
    h_hi: int,
    w_hi: int,
    h_lo: int,
    w_lo: int,
    taper: int,
    *,
    device: object,
    dtype: object,
) -> object:
    mask_h = _spd_cos_taper_1d(torch, h_hi, h_lo, taper, device=device, dtype=dtype)
    mask_w = _spd_cos_taper_1d(torch, w_hi, w_lo, taper, device=device, dtype=dtype)
    return mask_h[:, None] * mask_w[None, :]


def _spd_resolution_ratio(low_height: int, low_width: int, target_height: int, target_width: int) -> float:
    return ((float(target_height) / float(low_height)) * (float(target_width) / float(low_width))) ** 0.5


def _spd_aligned_sigma_and_kappa(
    torch: object,
    sigma: object,
    *,
    low_height: int,
    low_width: int,
    target_height: int,
    target_width: int,
) -> tuple[object, object, float]:
    ratio = _spd_resolution_ratio(low_height, low_width, target_height, target_width)
    timestep = torch.as_tensor(sigma, device=sigma.device if hasattr(sigma, "device") else None, dtype=torch.float32).reshape(())
    denom = 1.0 + (ratio - 1.0) * timestep
    sigma_aligned = (ratio * timestep) / denom
    kappa = ratio / denom
    return sigma_aligned, kappa, ratio


def _spd_rewrite_suffix_sigmas(torch: object, suffix_sigmas: object, sigma_aligned: object) -> object:
    rewritten = suffix_sigmas.clone()
    original = rewritten[0].clone()
    rewritten[0] = sigma_aligned.to(device=rewritten.device, dtype=rewritten.dtype)
    if float(original.detach().cpu()) > 0.0:
        rewritten[1:] = rewritten[0] * (rewritten[1:] / original)
    return rewritten


def _spd_dct_downscale_video_latent(
    torch: object,
    latent: object,
    *,
    target_height: int,
    target_width: int,
    taper: int,
) -> object:
    """DCT-downscale `[B,C,F,H,W]` video latents by preserving low spatial coefficients."""
    height = int(latent.shape[-2])
    width = int(latent.shape[-1])
    if target_height == height and target_width == width:
        return latent
    if target_height > height or target_width > width:
        raise RuntimeError(
            f"SPD downscale requires target shape <= source shape, got {width}x{height} "
            f"to {target_width}x{target_height}"
        )
    coeff = _dct2_partial(torch, latent, target_height, target_width)
    if taper > 0:
        mask = _spd_preserve_mask_2d(
            torch,
            target_height,
            target_width,
            target_height,
            target_width,
            taper,
            device=latent.device,
            dtype=torch.float32,
        )
        coeff = coeff * mask.view(*([1] * (coeff.dim() - 2)), target_height, target_width)
    downscaled = _idct2(torch, coeff)
    return downscaled.to(dtype=latent.dtype)


def _spd_dct_expand_video_latent(
    torch: object,
    latent: object,
    *,
    target_height: int,
    target_width: int,
    sigma: object,
    generator: object | None,
    highfreq_noise: bool,
    highfreq_reference_latent: object | None = None,
    highfreq_reference_sigma: object | None = None,
    taper: int = 8,
) -> tuple[object, object, float]:
    """Expand `[B,C,F,H,W]` video latents with paper SPD timestep alignment."""
    low_height = int(latent.shape[-2])
    low_width = int(latent.shape[-1])
    if low_height == target_height and low_width == target_width:
        sigma_aligned = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32).reshape(())
        return latent, sigma_aligned, 1.0
    if low_height > target_height or low_width > target_width:
        raise RuntimeError(
            f"SPD expansion requires low spatial shape <= target shape, got {low_width}x{low_height} "
            f"to {target_width}x{target_height}"
        )
    sigma_aligned, kappa, ratio = _spd_aligned_sigma_and_kappa(
        torch,
        sigma,
        low_height=low_height,
        low_width=low_width,
        target_height=target_height,
        target_width=target_width,
    )
    coeff_low = _dct2(torch, latent)
    coeff_high = torch.zeros(
        *coeff_low.shape[:-2],
        target_height,
        target_width,
        device=latent.device,
        dtype=torch.float32,
    )
    coeff_high[..., :low_height, :low_width] = coeff_low
    preserve_mask = _spd_preserve_mask_2d(
        torch,
        target_height,
        target_width,
        low_height,
        low_width,
        taper,
        device=latent.device,
        dtype=torch.float32,
    )
    preserve_mask_view = preserve_mask.view(*([1] * (coeff_high.dim() - 2)), target_height, target_width)
    if highfreq_noise:
        if highfreq_reference_latent is not None:
            reference = highfreq_reference_latent
            ref_height = int(reference.shape[-2])
            ref_width = int(reference.shape[-1])
            if ref_height != target_height or ref_width != target_width:
                reference = _spd_dct_downscale_video_latent(
                    torch,
                    reference,
                    target_height=target_height,
                    target_width=target_width,
                    taper=0,
                )
            ref_coeff = _dct2(torch, reference)
            ref_sigma = (
                torch.as_tensor(highfreq_reference_sigma, device=latent.device, dtype=torch.float32).reshape(())
                if highfreq_reference_sigma is not None
                else torch.ones((), device=latent.device, dtype=torch.float32)
            )
            sigma_scale = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32).reshape(()) / ref_sigma.clamp_min(1e-6)
            highfreq_fill = ref_coeff * sigma_scale
        else:
            highfreq_fill = torch.randn(
                coeff_high.shape,
                device=latent.device,
                dtype=torch.float32,
                generator=generator,
            ) * torch.as_tensor(sigma, device=latent.device, dtype=torch.float32).reshape(())
        coeff_high = coeff_high * preserve_mask_view + highfreq_fill * (1.0 - preserve_mask_view)
    else:
        coeff_high = coeff_high * preserve_mask_view
    expanded = _idct2(torch, coeff_high) * kappa
    return expanded.to(dtype=latent.dtype), sigma_aligned, ratio


def _spd_expand_video_state(
    *,
    torch: object,
    low_video_tools: object,
    high_video_tools: object,
    video_state: object,
    sigma: object,
    generator: object | None,
    highfreq_noise: bool,
    highfreq_reference_latent: object | None,
    highfreq_reference_sigma: object | None,
    taper: int,
) -> tuple[object, object, float]:
    low_unpatched = low_video_tools.clear_conditioning(video_state)
    low_unpatched = low_video_tools.unpatchify(low_unpatched)
    high_shape = high_video_tools.target_shape
    expanded_latent, sigma_aligned, ratio = _spd_dct_expand_video_latent(
        torch,
        low_unpatched.latent,
        target_height=int(high_shape.height),
        target_width=int(high_shape.width),
        sigma=sigma,
        generator=generator,
        highfreq_noise=highfreq_noise,
        highfreq_reference_latent=highfreq_reference_latent,
        highfreq_reference_sigma=highfreq_reference_sigma,
        taper=taper,
    )
    return _spd_video_state_from_unpatched_latent(torch=torch, video_tools=high_video_tools, latent=expanded_latent), sigma_aligned, ratio


def _spd_video_state_from_unpatched_latent(*, torch: object, video_tools: object, latent: object) -> object:
    zero_latent = torch.zeros_like(latent)
    state = video_tools.create_initial_state(latent.device, latent.dtype, zero_latent)
    return replace(state, latent=video_tools.patchifier.patchify(latent))


def _spd_build_initial_low_video_state(
    *,
    torch: object,
    blocks: object,
    video: object,
    noiser: object,
    dtype: object,
    device: object,
    high_pixel_shape: object,
    low_pixel_shape: object,
    fps: float,
    taper: int,
) -> tuple[object, object, object]:
    high_v_shape = blocks.VideoLatentShape.from_pixel_shape(high_pixel_shape)
    high_video_tools = blocks.VideoLatentTools(blocks.VideoLatentPatchifier(patch_size=1), high_v_shape, fps)
    high_video_state = blocks._build_state(video, high_video_tools, noiser, dtype, device)
    high_unpatched = high_video_tools.clear_conditioning(high_video_state)
    high_unpatched = high_video_tools.unpatchify(high_unpatched)

    low_v_shape = blocks.VideoLatentShape.from_pixel_shape(low_pixel_shape)
    low_video_tools = blocks.VideoLatentTools(blocks.VideoLatentPatchifier(patch_size=1), low_v_shape, fps)
    low_latent = _spd_dct_downscale_video_latent(
        torch,
        high_unpatched.latent,
        target_height=int(low_v_shape.height),
        target_width=int(low_v_shape.width),
        taper=taper,
    )
    return (
        _spd_video_state_from_unpatched_latent(torch=torch, video_tools=low_video_tools, latent=low_latent),
        low_video_tools,
        high_unpatched.latent,
    )


def _run_stage_batch(
    *,
    stage: object,
    transformer: object,
    denoiser: object,
    sigmas: object,
    noiser: object,
    width: int,
    height: int,
    frames: int,
    fps: float,
    video: object | None = None,
    audio: object | None = None,
    batch_size: int = 1,
    max_batch_size: int = 1,
    uniform_timestep_adaln: bool = False,
    spd_config: _SPDConfig | None = None,
    spd_generator: object | None = None,
    spd_stats: dict[str, Any] | None = None,
) -> tuple[object | None, object | None]:
    """DiffusionStage.run with a fixed benchmark batch dimension.

    Upstream currently hard-codes ``VideoPixelShape(batch=1)``. The serving
    benchmark needs to test true same-shape microbatching, so this mirrors the
    official run path but lets the latent tools allocate B>1 states.
    """
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if max_batch_size < 1:
        raise ValueError("--max-batch-size must be >= 1")
    if video is None and audio is None:
        raise ValueError("At least one of `video` or `audio` must be provided")

    import ltx_pipelines.utils.blocks as blocks
    import torch

    high_pixel_shape = blocks.VideoPixelShape(batch=batch_size, frames=frames, height=height, width=width, fps=fps)
    build_pixel_shape = (
        blocks.VideoPixelShape(
            batch=batch_size,
            frames=frames,
            height=spd_config.low_height,
            width=spd_config.low_width,
            fps=fps,
        )
        if spd_config is not None
        else high_pixel_shape
    )

    video_state = None
    video_tools = None
    spd_highfreq_reference_latent = None
    if video is not None:
        v_shape = blocks.VideoLatentShape.from_pixel_shape(build_pixel_shape)
        video_tools = blocks.VideoLatentTools(blocks.VideoLatentPatchifier(patch_size=1), v_shape, fps)
        if spd_config is not None and spd_config.initial_dct_downscale:
            video_state, video_tools, spd_highfreq_reference_latent = _spd_build_initial_low_video_state(
                torch=torch,
                blocks=blocks,
                video=video,
                noiser=noiser,
                dtype=stage._dtype,
                device=stage._device,
                high_pixel_shape=high_pixel_shape,
                low_pixel_shape=build_pixel_shape,
                fps=fps,
                taper=spd_config.taper,
            )
            if spd_config.highfreq_source != "initial":
                spd_highfreq_reference_latent = None
        else:
            video_state = blocks._build_state(video, video_tools, noiser, stage._dtype, stage._device)
        if uniform_timestep_adaln:
            _assert_uniform_full_denoise_state(video_state, "video")

    audio_state = None
    audio_tools = None
    if audio is not None:
        a_shape = blocks.AudioLatentShape.from_video_pixel_shape(build_pixel_shape)
        audio_tools = blocks.AudioLatentTools(blocks.AudioPatchifier(patch_size=1), a_shape)
        audio_state = blocks._build_state(audio, audio_tools, noiser, stage._dtype, stage._device)
        if uniform_timestep_adaln:
            _assert_uniform_full_denoise_state(audio_state, "audio")

    wrapped = blocks.BatchSplitAdapter(transformer, max_batch_size=max_batch_size)
    if spd_config is None:
        loop_start = time.perf_counter()
        video_state, audio_state = blocks.euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=blocks.EulerDiffusionStep(),
            transformer=wrapped,
            denoiser=denoiser,
        )
        if spd_stats is not None:
            spd_stats["full_loop_s"] = time.perf_counter() - loop_start
    else:
        if video is None or video_state is None or video_tools is None:
            raise RuntimeError("--spd requires video generation")
        if spd_config.transform != "dct":
            raise RuntimeError(f"unsupported SPD transform: {spd_config.transform}")
        if any(step <= 0 or step >= len(sigmas) - 1 for step in spd_config.transition_steps):
            raise RuntimeError(f"SPD transition steps are outside the denoise schedule: {spd_config.transition_steps}")

        working_sigmas = sigmas.clone()
        segment_start_idx = 0
        segment_records: list[dict[str, Any]] = []
        transition_records: list[dict[str, Any]] = []
        current_scale = spd_config.stages[0].scale
        expand_s_total = 0.0

        for transition_idx, transition_step in enumerate(spd_config.transition_steps):
            segment_sigmas = working_sigmas[segment_start_idx : transition_step + 1]
            if len(segment_sigmas) < 2:
                raise RuntimeError("SPD split produced an empty denoise segment")

            segment_start = time.perf_counter()
            video_state, audio_state = blocks.euler_denoising_loop(
                sigmas=segment_sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=blocks.EulerDiffusionStep(),
                transformer=wrapped,
                denoiser=denoiser,
            )
            segment_loop_s = time.perf_counter() - segment_start
            segment_records.append(
                {
                    "kind": "denoise",
                    "start_step": segment_start_idx,
                    "end_step": transition_step,
                    "scale": float(current_scale),
                    "loop_s": segment_loop_s,
                    "sigmas": [float(x) for x in segment_sigmas.detach().cpu().tolist()],
                }
            )

            expand_start = time.perf_counter()
            if transition_idx + 1 < len(spd_config.stages):
                next_stage = spd_config.stages[transition_idx + 1]
                next_pixel_shape = blocks.VideoPixelShape(
                    batch=batch_size,
                    frames=frames,
                    height=next_stage.height,
                    width=next_stage.width,
                    fps=fps,
                )
                next_scale = next_stage.scale
                target_shape_meta = dict(next_stage.shape)
            else:
                next_pixel_shape = high_pixel_shape
                next_scale = 1.0
                target_shape_meta = {
                    "scale": 1.0,
                    "width": int(width),
                    "height": int(height),
                    "frames": int(frames),
                    "fps": float(fps),
                    "latent_width": int(width) // 32,
                    "latent_height": int(height) // 32,
                    "latent_frames": 1 + (int(frames) - 1) // 8,
                }
            next_v_shape = blocks.VideoLatentShape.from_pixel_shape(next_pixel_shape)
            next_video_tools = blocks.VideoLatentTools(blocks.VideoLatentPatchifier(patch_size=1), next_v_shape, fps)
            raw_suffix_sigmas = working_sigmas[transition_step:].clone()
            video_state, sigma_aligned, scale_ratio = _spd_expand_video_state(
                torch=torch,
                low_video_tools=video_tools,
                high_video_tools=next_video_tools,
                video_state=video_state,
                sigma=working_sigmas[transition_step],
                generator=spd_generator,
                highfreq_noise=spd_config.highfreq_noise,
                highfreq_reference_latent=spd_highfreq_reference_latent,
                highfreq_reference_sigma=working_sigmas[0],
                taper=spd_config.taper,
            )
            suffix_sigmas = _spd_rewrite_suffix_sigmas(torch, raw_suffix_sigmas, sigma_aligned)
            working_sigmas[transition_step:] = suffix_sigmas
            video_tools = next_video_tools
            expand_s = time.perf_counter() - expand_start
            expand_s_total += expand_s
            transition_records.append(
                {
                    "kind": "expand",
                    "step": transition_step,
                    "from_scale": float(current_scale),
                    "to_scale": float(next_scale),
                    "target_shape": target_shape_meta,
                    "expand_s": expand_s,
                    "scale_ratio": float(scale_ratio),
                    "sigma_original": float(raw_suffix_sigmas[0].detach().cpu()),
                    "sigma_aligned": float(sigma_aligned.detach().cpu()),
                    "raw_suffix_sigmas": [float(x) for x in raw_suffix_sigmas.detach().cpu().tolist()],
                    "suffix_sigmas": [float(x) for x in suffix_sigmas.detach().cpu().tolist()],
                }
            )
            current_scale = next_scale
            segment_start_idx = transition_step

        final_sigmas = working_sigmas[segment_start_idx:]
        if len(final_sigmas) < 2:
            raise RuntimeError("SPD split produced an empty final denoise segment")
        final_start = time.perf_counter()
        video_state, audio_state = blocks.euler_denoising_loop(
            sigmas=final_sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=blocks.EulerDiffusionStep(),
            transformer=wrapped,
            denoiser=denoiser,
        )
        final_loop_s = time.perf_counter() - final_start
        segment_records.append(
            {
                "kind": "denoise",
                "start_step": segment_start_idx,
                "end_step": len(working_sigmas) - 1,
                "scale": float(current_scale),
                "loop_s": final_loop_s,
                "sigmas": [float(x) for x in final_sigmas.detach().cpu().tolist()],
            }
        )
        low_loop_s = segment_records[0]["loop_s"]
        high_loop_s = segment_records[-1]["loop_s"]
        if spd_stats is not None:
            spd_stats.update(
                {
                    "enabled": True,
                    "scale": spd_config.scale,
                    "transition_step": spd_config.transition_step,
                    "transition_steps": list(spd_config.transition_steps),
                    "stages": [dict(stage.shape) for stage in spd_config.stages]
                    + [
                        {
                            "scale": 1.0,
                            "width": int(width),
                            "height": int(height),
                            "frames": int(frames),
                            "fps": float(fps),
                            "latent_width": int(width) // 32,
                            "latent_height": int(height) // 32,
                            "latent_frames": 1 + (int(frames) - 1) // 8,
                        }
                    ],
                    "transform": spd_config.transform,
                    "taper": spd_config.taper,
                    "initial_dct_downscale": spd_config.initial_dct_downscale,
                    "highfreq_noise": spd_config.highfreq_noise,
                    "highfreq_source": spd_config.highfreq_source,
                    "scale_ratio": transition_records[-1]["scale_ratio"],
                    "sigma_original": transition_records[-1]["sigma_original"],
                    "sigma_aligned": transition_records[-1]["sigma_aligned"],
                    "low_shape": dict(spd_config.low_shape),
                    "low_loop_s": low_loop_s,
                    "expand_s": expand_s_total,
                    "high_loop_s": high_loop_s,
                    "segments": segment_records,
                    "transitions": transition_records,
                    "prefix_sigmas": segment_records[0]["sigmas"],
                    "raw_suffix_sigmas": transition_records[-1]["raw_suffix_sigmas"],
                    "suffix_sigmas": transition_records[-1]["suffix_sigmas"],
                }
            )

    if video_state is not None and video_tools is not None:
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
    if audio_state is not None and audio_tools is not None:
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


def _assert_uniform_full_denoise_state(state: object, name: str) -> None:
    import torch

    denoise_mask = getattr(state, "denoise_mask", None)
    if denoise_mask is None:
        raise RuntimeError(f"{name} state has no denoise_mask")
    if not bool(torch.all(denoise_mask == 1).item()):
        raise RuntimeError(
            "--uniform-timestep-adaln is valid only for full-denoise T2V states with an all-ones denoise mask"
        )


def _save_latents(
    args: argparse.Namespace,
    torch: object,
    output_dir: Path,
    run_index: int,
    video_state: object | None,
    audio_state: object | None,
) -> dict[str, str]:
    if not args.save_latents:
        return {}

    outputs: dict[str, str] = {}
    if video_state is not None:
        path = output_dir / f"video_latent_run_{run_index}.pt"
        torch.save(video_state.latent.detach().cpu(), path)
        outputs["video_latent"] = str(path)
    if audio_state is not None:
        path = output_dir / f"audio_latent_run_{run_index}.pt"
        torch.save(audio_state.latent.detach().cpu(), path)
        outputs["audio_latent"] = str(path)
    return outputs


def _encode_prompt_contexts(args: argparse.Namespace, pipeline: object, torch: object) -> tuple[float, object, object]:
    prompt_start = time.perf_counter()
    prompt_contexts = pipeline.prompt_encoder([args.prompt] * args.batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prompt_s = time.perf_counter() - prompt_start
    video_context = torch.cat([ctx.video_encoding for ctx in prompt_contexts], dim=0)
    audio_context = torch.cat([ctx.audio_encoding for ctx in prompt_contexts], dim=0)
    return prompt_s, video_context, audio_context


def _build_image_conditionings(
    args: argparse.Namespace,
    pipeline: object,
    bucket: object,
    torch: object,
) -> tuple[float, list[object]]:
    if args.image_conditioning_path is None and not args.image_conditioning_spec:
        return 0.0, []

    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.helpers import combined_image_conditionings

    images: list[object] = []
    if args.image_conditioning_path is not None:
        conditioning_path = Path(args.image_conditioning_path)
        if not conditioning_path.exists():
            raise FileNotFoundError(f"image conditioning path does not exist: {conditioning_path}")
        images.append(
            ImageConditioningInput(
                path=str(conditioning_path),
                frame_idx=args.image_conditioning_frame_idx,
                strength=args.image_conditioning_strength,
                crf=args.image_conditioning_crf,
            )
        )
    for spec in args.image_conditioning_spec:
        images.append(_parse_image_conditioning_spec(spec, ImageConditioningInput))

    device = torch.device(args.device)
    dtype = getattr(pipeline, "dtype", getattr(pipeline, "_dtype", torch.bfloat16))

    conditioning_start = time.perf_counter()
    conditionings = pipeline.image_conditioner(
        lambda encoder: combined_image_conditionings(
            images=images,
            height=bucket.height,
            width=bucket.width,
            video_encoder=encoder,
            dtype=dtype,
            device=device,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - conditioning_start, conditionings


def _parse_image_conditioning_spec(spec: str, cls: object) -> object:
    parts = spec.rsplit(":", 3)
    if len(parts) not in {3, 4}:
        raise ValueError("image conditioning spec must be PATH:FRAME:STRENGTH[:CRF]")
    path = Path(parts[0])
    if not path.exists():
        raise FileNotFoundError(f"image conditioning path does not exist: {path}")
    frame_idx = int(parts[1])
    strength = float(parts[2])
    crf = int(parts[3]) if len(parts) == 4 else 33
    return cls(path=str(path), frame_idx=frame_idx, strength=strength, crf=crf)


def _video_decode_tiling_config(args: argparse.Namespace) -> object | None:
    if args.decode_video_tiling == "none":
        return None
    if args.decode_video_tiling == "default":
        from ltx_core.model.video_vae.tiling import TilingConfig

        return TilingConfig.default()
    raise ValueError(f"unknown decode video tiling mode: {args.decode_video_tiling}")


def _audio_trim_start_s(args: argparse.Namespace, bucket: object) -> float:
    if args.encode_audio_trim_start_s is not None:
        return float(args.encode_audio_trim_start_s)
    if args.encode_start_frame <= 0:
        return 0.0
    return float(args.encode_start_frame) / float(bucket.final_fps)


@dataclass
class _EncoderSession:
    proc: subprocess.Popen[Any]
    reader: threading.Thread
    first_byte_queue: queue.Queue[float]
    output_path: Path
    encode_start: float
    byte_count: list[int]
    audio_fifo_path: Path | None = None
    audio_writer: threading.Thread | None = None
    audio_errors: list[str] | None = None


def _build_ffmpeg_stream_cmd(
    bucket: object,
    codec: str,
    container: str = "mpegts",
    *,
    audio_path: Path | None = None,
    x264_preset: str = "ultrafast",
    x264_crf: str = "19",
    x264_params: str = "",
    encode_threads: int = 0,
    low_latency_mux: bool = False,
    movflags: str = "frag_keyframe+empty_moov+default_base_moof",
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{int(bucket.final_width)}x{int(bucket.final_height)}",
        "-r",
        str(int(bucket.final_fps)),
        "-i",
        "pipe:0",
    ]
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if audio_path is not None:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    if codec == "h264_nvenc":
        cmd += ["-preset", "p1", "-tune", "ull", "-rc", "constqp", "-qp", "23", "-g", str(int(bucket.final_fps)), "-bf", "0"]
    else:
        cmd += ["-preset", x264_preset, "-tune", "zerolatency", "-crf", x264_crf, "-g", str(int(bucket.final_fps)), "-bf", "0"]
        if x264_params:
            cmd += ["-x264-params", x264_params]
    if encode_threads > 0:
        cmd += ["-threads", str(encode_threads)]
    if low_latency_mux:
        cmd += ["-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0"]
    if container == "mpegts":
        cmd += ["-f", "mpegts", "pipe:1"]
    elif container == "frag-mp4":
        cmd += ["-movflags", movflags, "-f", "mp4", "pipe:1"]
    else:
        raise ValueError(f"unknown encode container: {container}")
    return cmd


def _start_video_stream_probe(
    bucket: object,
    output_dir: Path,
    run_index: int,
    codec: str,
    container: str = "mpegts",
    *,
    x264_preset: str = "ultrafast",
    x264_crf: str = "19",
    x264_params: str = "",
    encode_threads: int = 0,
    low_latency_mux: bool = False,
    movflags: str = "frag_keyframe+empty_moov+default_base_moof",
    audio: object | None = None,
    audio_pipe: bool = False,
    audio_trim_start_s: float = 0.0,
) -> _EncoderSession:
    suffix = ".mp4" if container == "frag-mp4" else ".ts"
    mux_suffix = "_av" if audio is not None or audio_pipe else ""
    output_path = output_dir / f"stream_probe_run_{run_index}{mux_suffix}{suffix}"
    encode_start = time.perf_counter()
    audio_fifo_path = None
    if audio_pipe:
        audio_fifo_path = output_dir / f"audio_run_{run_index}.wavpipe"
        try:
            audio_fifo_path.unlink()
        except FileNotFoundError:
            pass
        os.mkfifo(audio_fifo_path)
        audio_path = audio_fifo_path
    else:
        audio_path = _write_audio_wav(audio, output_dir, run_index, trim_start_s=audio_trim_start_s) if audio is not None else None
    cmd = _build_ffmpeg_stream_cmd(
        bucket,
        codec,
        container,
        audio_path=audio_path,
        x264_preset=x264_preset,
        x264_crf=x264_crf,
        x264_params=x264_params,
        encode_threads=encode_threads,
        low_latency_mux=low_latency_mux,
        movflags=movflags,
    )

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_byte_queue: queue.Queue[float] = queue.Queue(maxsize=1)
    byte_count = [0]

    def _reader() -> None:
        assert proc.stdout is not None
        with output_path.open("wb") as output_file:
            while True:
                data = proc.stdout.read(65536)
                if not data:
                    break
                output_file.write(data)
                byte_count[0] += len(data)
                if first_byte_queue.empty():
                    first_byte_queue.put(time.perf_counter())

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    session = _EncoderSession(
        proc,
        reader,
        first_byte_queue,
        output_path,
        encode_start,
        byte_count,
        audio_fifo_path=audio_fifo_path,
        audio_errors=[],
    )
    if audio_pipe and audio is not None:
        _attach_audio_to_encoder(session, audio, trim_start_s=audio_trim_start_s)
    return session


def _finish_video_stream_probe(
    session: _EncoderSession,
    chunk: object,
    bucket: object,
    *,
    feed_mode: str = "bulk",
    frame_batch: int = 1,
    start_frame: int = 0,
    cpu_materialize: bool = False,
) -> dict[str, float | int | str]:
    import torch

    if session.audio_fifo_path is not None and session.audio_writer is None:
        raise RuntimeError("audio FIFO encoder was started but no generated audio writer was attached")

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    frames = int(min(max(int(chunk.shape[0]) - start_frame, 0), bucket.final_frames))
    if frames <= 0:
        raise ValueError("start_frame drops all decoded frames")
    crop_top = max((int(chunk.shape[1]) - int(bucket.final_height)) // 2, 0)
    crop_left = max((int(chunk.shape[2]) - int(bucket.final_width)) // 2, 0)
    cropped = chunk[
        start_frame : start_frame + frames,
        crop_top : crop_top + int(bucket.final_height),
        crop_left : crop_left + int(bucket.final_width),
        :,
    ]
    if cpu_materialize and hasattr(cropped, "detach"):
        cropped = _frame_block_to_uint8_numpy(cropped, torch)
    if frame_batch < 1:
        raise ValueError("--encode-frame-batch must be >= 1")
    proc = session.proc
    assert proc.stdin is not None
    try:
        if feed_mode == "bulk":
            proc.stdin.write(_frame_block_to_bytes(cropped, torch))
        elif feed_mode == "frame-stream":
            for start_frame in range(0, frames, frame_batch):
                frame_block = cropped[start_frame : start_frame + frame_batch]
                proc.stdin.write(_frame_block_to_bytes(frame_block, torch))
                proc.stdin.flush()
        else:
            raise ValueError(f"unknown encode feed mode: {feed_mode}")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    try:
        if session.audio_writer is not None:
            session.audio_writer.join(timeout=10)
            if session.audio_writer.is_alive():
                raise RuntimeError("timed out waiting for audio FIFO writer")
        if session.audio_errors:
            raise RuntimeError(f"audio FIFO writer failed: {'; '.join(session.audio_errors)}")
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        return_code = proc.wait()
        session.reader.join(timeout=5)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg stream probe failed with code {return_code}: {stderr[-2000:]}")
        first_byte_abs_s = (
            session.first_byte_queue.get_nowait() if not session.first_byte_queue.empty() else time.perf_counter()
        )
        return {
            "encode_video_s": time.perf_counter() - session.encode_start,
            "encode_first_byte_s": first_byte_abs_s - session.encode_start,
            "encode_first_byte_abs_s": first_byte_abs_s,
            "encode_bytes": session.byte_count[0],
            "output_path": str(session.output_path),
        }
    finally:
        if session.audio_fifo_path is not None:
            try:
                session.audio_fifo_path.unlink()
            except FileNotFoundError:
                pass


def _frame_block_to_uint8_numpy(frame_block: object, torch: object) -> object:
    if hasattr(frame_block, "detach"):
        return frame_block.float().mul(255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().contiguous().numpy()
    return frame_block


def _frame_block_to_bytes(frame_block: object, torch: object) -> bytes:
    if hasattr(frame_block, "detach"):
        frame_block = _frame_block_to_uint8_numpy(frame_block, torch)
    if not hasattr(frame_block, "tobytes"):
        raise TypeError(f"unsupported frame block type: {type(frame_block)!r}")
    return frame_block.tobytes()


def _encode_video_stream_probe(
    chunk: object,
    bucket: object,
    output_dir: Path,
    run_index: int,
    codec: str,
    container: str = "mpegts",
    *,
    feed_mode: str = "bulk",
    frame_batch: int = 1,
    x264_preset: str = "ultrafast",
    x264_crf: str = "19",
    x264_params: str = "",
    encode_threads: int = 0,
    low_latency_mux: bool = False,
    movflags: str = "frag_keyframe+empty_moov+default_base_moof",
    audio: object | None = None,
    audio_pipe: bool = False,
    audio_trim_start_s: float = 0.0,
    start_frame: int = 0,
) -> dict[str, float | int | str]:
    session = _start_video_stream_probe(
        bucket,
        output_dir,
        run_index,
        codec,
        container,
        x264_preset=x264_preset,
        x264_crf=x264_crf,
        x264_params=x264_params,
        encode_threads=encode_threads,
        low_latency_mux=low_latency_mux,
        movflags=movflags,
        audio=audio,
        audio_pipe=audio_pipe,
        audio_trim_start_s=audio_trim_start_s,
    )
    return _finish_video_stream_probe(
        session,
        chunk,
        bucket,
        feed_mode=feed_mode,
        frame_batch=frame_batch,
        start_frame=start_frame,
    )


def _audio_wav_bytes(audio: object, trim_start_s: float = 0.0) -> bytes:
    import wave

    pcm, sampling_rate = _audio_pcm_int16_stereo(audio, trim_start_s=trim_start_s)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sampling_rate))
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _attach_audio_to_encoder(session: _EncoderSession, audio: object, trim_start_s: float = 0.0) -> None:
    if session.audio_fifo_path is None:
        raise RuntimeError("cannot attach audio to an encoder without an audio FIFO")
    if session.audio_writer is not None:
        raise RuntimeError("audio writer already attached")
    audio_bytes = _audio_wav_bytes(audio, trim_start_s=trim_start_s)
    audio_errors = session.audio_errors if session.audio_errors is not None else []
    session.audio_errors = audio_errors

    def _write_audio() -> None:
        try:
            with session.audio_fifo_path.open("wb") as fifo:
                fifo.write(audio_bytes)
        except Exception as exc:  # pragma: no cover - exercised on remote ffmpeg failures.
            audio_errors.append(str(exc))

    writer = threading.Thread(target=_write_audio, daemon=True)
    writer.start()
    session.audio_writer = writer


def _write_audio_wav(
    audio: object | None,
    output_dir: Path,
    run_index: int,
    *,
    trim_start_s: float = 0.0,
) -> Path | None:
    if audio is None:
        return None
    audio_path = output_dir / f"audio_run_{run_index}.wav"
    audio_path.write_bytes(_audio_wav_bytes(audio, trim_start_s=trim_start_s))
    return audio_path

def _audio_pcm_int16_stereo(audio: object, trim_start_s: float = 0.0) -> tuple[object, int]:
    import torch

    waveform = getattr(audio, "waveform", None)
    sampling_rate = getattr(audio, "sampling_rate", None)
    if waveform is None or sampling_rate is None:
        raise TypeError("audio muxing requires an object with waveform and sampling_rate")

    samples = waveform.detach().float().cpu()
    if samples.ndim == 3 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim == 1:
        samples = samples.unsqueeze(0).repeat(2, 1)
    elif samples.ndim == 2:
        if samples.shape[0] == 1:
            samples = samples.repeat(2, 1)
        elif samples.shape[0] == 2:
            pass
        elif samples.shape[1] == 1:
            samples = samples.T.repeat(2, 1)
        elif samples.shape[1] == 2:
            samples = samples.T
        else:
            raise ValueError(f"cannot infer audio channel layout from waveform shape {tuple(samples.shape)}")
    else:
        raise ValueError(f"cannot mux waveform with shape {tuple(samples.shape)}")

    if trim_start_s < 0:
        raise ValueError("audio trim must be >= 0")
    trim_samples = int(round(float(trim_start_s) * int(sampling_rate)))
    if trim_samples:
        samples = samples[:, min(trim_samples, samples.shape[-1]) :]

    pcm = torch.clip(samples, -1.0, 1.0).mul(32767.0).to(torch.int16).T.contiguous().numpy()
    return pcm, int(sampling_rate)


def _patch_hot_pipeline_blocks(args: argparse.Namespace, pipeline: object) -> None:
    if args.hot_prompt_encoder:
        pipeline.prompt_encoder = _HotPromptEncoder(pipeline.prompt_encoder)
    if args.hot_decoders:
        pipeline.video_decoder = _HotVideoDecoder(pipeline.video_decoder)
        pipeline.audio_decoder = _HotAudioDecoder(pipeline.audio_decoder)


class _HotPromptEncoder:
    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.text_encoder = None
        self.embeddings_processor = None

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> list[object]:
        if self.text_encoder is None:
            self.text_encoder = self.inner._build_text_encoder()
        if self.embeddings_processor is None:
            self.embeddings_processor = self.inner._build_embeddings_processor()

        if enhance_first_prompt:
            from ltx_pipelines.utils.helpers import generate_enhanced_prompt

            prompts = list(prompts)
            prompts[0] = generate_enhanced_prompt(
                self.text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
            )
        raw_outputs = [self.text_encoder.encode(p) for p in prompts]
        return [self.embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]


class _HotVideoDecoder:
    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.decoder = None

    def __call__(
        self,
        latent: object,
        tiling_config: object | None = None,
        generator: object | None = None,
    ) -> object:
        if self.decoder is None:
            self.decoder = self.inner._decoder_builder.build(
                device=self.inner._device,
                dtype=self.inner._dtype,
            ).eval()
        return self.decoder.decode_video(latent, tiling_config, generator)


class _HotAudioDecoder:
    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.decoder = None
        self.vocoder = None

    def __call__(self, latent: object) -> object:
        if self.decoder is None:
            self.decoder = self.inner._decoder_builder.build(device=self.inner._device, dtype=self.inner._dtype).eval()
        if self.vocoder is None:
            self.vocoder = self.inner._vocoder_builder.build(device=self.inner._device, dtype=self.inner._dtype).eval()

        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

        return vae_decode_audio(latent, self.decoder, self.vocoder)


@dataclass(frozen=True)
class _ResidualCacheConfig:
    mode: str
    threshold: float
    max_skips: int
    retention_ratio: float
    force_skip_steps: frozenset[int]
    mag_ratios: tuple[float, ...]
    metric_element_stride: int
    total_steps: int


@dataclass
class _ResidualCacheDecision:
    skip: bool
    video: object | None
    audio: object | None
    record: dict[str, Any]


def _parse_int_set(value: str) -> list[int]:
    if not value.strip():
        return []
    parsed: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        parsed.append(int(token))
    return sorted(set(parsed))


def _parse_mag_ratios(value: str) -> list[float]:
    if not value.strip():
        return []
    path = Path(value)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("mag_ratios", payload.get("ratios"))
        if not isinstance(payload, list):
            raise ValueError("MagCache ratio JSON must be a list or an object with mag_ratios/ratios")
        return [float(x) for x in payload]
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _nearest_interp(values: tuple[float, ...], target_length: int) -> tuple[float, ...]:
    if not values:
        return ()
    if target_length <= 1:
        return (float(values[-1]),)
    if len(values) == target_length:
        return tuple(float(x) for x in values)
    scale = (len(values) - 1) / float(target_length - 1)
    return tuple(float(values[round(i * scale)]) for i in range(target_length))


def _residual_cache_config_from_args(args: argparse.Namespace) -> _ResidualCacheConfig | None:
    if args.residual_cache_mode == "off":
        return None
    total_steps = int(args.steps)
    mag_ratios = tuple(_parse_mag_ratios(args.residual_cache_mag_ratios))
    if args.residual_cache_mode == "magcache":
        mag_ratios = _nearest_interp(mag_ratios, total_steps)
    return _ResidualCacheConfig(
        mode=str(args.residual_cache_mode),
        threshold=float(args.residual_cache_threshold),
        max_skips=int(args.residual_cache_max_skips),
        retention_ratio=float(args.residual_cache_retention_ratio),
        force_skip_steps=frozenset(_parse_int_set(args.residual_cache_force_skip_steps)),
        mag_ratios=mag_ratios,
        metric_element_stride=int(args.residual_cache_metric_element_stride),
        total_steps=total_steps,
    )


class _ResidualDenoiserCache:
    """Whole-denoiser residual cache for lab probes.

    This intentionally sits outside the model internals. It stores the Euler
    velocity field `(latent - denoised) / sigma` for the full video/audio
    denoiser output and reuses it only while the modality shape key is
    unchanged. SPD resolution transitions therefore force a full compute before
    a new stage-local velocity can be cached.
    """

    def __init__(self, config: _ResidualCacheConfig, stats: dict[str, Any] | None = None) -> None:
        self.config = config
        self.stats = stats if stats is not None else {}
        self.step_index = 0
        self.consecutive_skips = 0
        self.accumulated_metric = 0.0
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_mag_steps = 0
        self.shape_key: tuple[Any, ...] | None = None
        self.cached_video_velocity = None
        self.cached_audio_velocity = None
        self.previous_video_sample = None
        self.previous_audio_sample = None
        self.previous_video_velocity_mag: float | None = None
        self.previous_audio_velocity_mag: float | None = None
        self._init_stats()

    def _init_stats(self) -> None:
        self.stats.clear()
        self.stats.update(
            {
                "mode": self.config.mode,
                "threshold": self.config.threshold,
                "max_skips": self.config.max_skips,
                "retention_ratio": self.config.retention_ratio,
                "metric_element_stride": self.config.metric_element_stride,
                "force_skip_steps": sorted(self.config.force_skip_steps),
                "mag_ratios": list(self.config.mag_ratios) if self.config.mode == "magcache" else None,
                "computed_steps": [],
                "skipped_steps": [],
                "records": [],
                "mag_calibration_ratios": [],
                "video_mag_calibration_ratios": [],
                "audio_mag_calibration_ratios": [],
                "invalidations": [],
            }
        )

    def begin_step(
        self,
        video_latent: object | None,
        audio_latent: object | None,
        sigma: object,
    ) -> _ResidualCacheDecision:
        shape_key = self._shape_key(video_latent, audio_latent)
        shape_changed = self.shape_key is not None and shape_key != self.shape_key
        if shape_changed:
            self._invalidate("shape_changed", shape_key)
        if self.shape_key is None:
            self.shape_key = shape_key

        teacache_metric = None
        if self.config.mode in {"teacache", "calibrate"}:
            teacache_metric = self._teacache_metric(video_latent, audio_latent)
            if self.config.mode == "teacache" and teacache_metric is not None:
                self.accumulated_metric += teacache_metric

        mag_scale = None
        if self.config.mode == "magcache":
            mag_scale = self.config.mag_ratios[min(self.step_index, len(self.config.mag_ratios) - 1)]
            if self._past_retention():
                self.accumulated_ratio *= float(mag_scale)
                self.accumulated_mag_steps += 1
                self.accumulated_err += abs(1.0 - self.accumulated_ratio)

        has_required_residuals = self._has_required_residuals(video_latent, audio_latent)
        may_skip_common = (
            has_required_residuals
            and self._past_retention()
            and self.step_index < self.config.total_steps - 1
            and self.consecutive_skips < self.config.max_skips
            and len(self.stats["skipped_steps"]) < self.config.max_skips
        )
        skip = False
        reason = "compute"
        if shape_changed:
            reason = "shape_changed"
        elif self.config.mode == "force-skip":
            skip = may_skip_common and self.step_index in self.config.force_skip_steps
            reason = "force_skip" if skip else "not_forced"
        elif self.config.mode == "teacache":
            skip = may_skip_common and teacache_metric is not None and self.accumulated_metric <= self.config.threshold
            reason = "teacache_threshold" if skip else "teacache_compute"
        elif self.config.mode == "magcache":
            skip = may_skip_common and self.accumulated_err <= self.config.threshold and self.accumulated_mag_steps <= self.config.max_skips
            reason = "magcache_threshold" if skip else "magcache_compute"
        elif self.config.mode == "calibrate":
            reason = "calibrate"

        if self.config.mode in {"teacache", "calibrate"}:
            self._store_metric_samples(video_latent, audio_latent)
        record = {
            "step": self.step_index,
            "action": "skip" if skip else "compute",
            "reason": reason,
            "shape_changed": shape_changed,
            "video_shape": list(video_latent.shape) if video_latent is not None and hasattr(video_latent, "shape") else None,
            "audio_shape": list(audio_latent.shape) if audio_latent is not None and hasattr(audio_latent, "shape") else None,
            "teacache_metric": teacache_metric,
            "accumulated_metric": self.accumulated_metric if self.config.mode == "teacache" else None,
            "mag_scale": mag_scale,
            "accumulated_mag_err": self.accumulated_err if self.config.mode == "magcache" else None,
            "consecutive_skips_before": self.consecutive_skips,
            "total_skips_before": len(self.stats["skipped_steps"]),
        }
        if skip:
            video_out = self._apply_velocity(video_latent, self.cached_video_velocity, sigma)
            audio_out = self._apply_velocity(audio_latent, self.cached_audio_velocity, sigma)
            self._finish_skip(record)
            return _ResidualCacheDecision(True, video_out, audio_out, record)
        return _ResidualCacheDecision(False, None, None, record)

    def finish_compute(
        self,
        *,
        video_latent: object | None,
        audio_latent: object | None,
        sigma: object,
        denoised_video: object | None,
        denoised_audio: object | None,
        record: dict[str, Any],
    ) -> None:
        video_velocity = self._velocity(denoised_video, video_latent, sigma)
        audio_velocity = self._velocity(denoised_audio, audio_latent, sigma)
        video_mag, video_ratio = self._velocity_mag_and_ratio(video_velocity, self.previous_video_velocity_mag)
        audio_mag, audio_ratio = self._velocity_mag_and_ratio(audio_velocity, self.previous_audio_velocity_mag)
        self.cached_video_velocity = video_velocity
        self.cached_audio_velocity = audio_velocity
        self.previous_video_velocity_mag = video_mag
        self.previous_audio_velocity_mag = audio_mag
        if self.config.mode == "calibrate":
            ratio_candidates = [r for r in (video_ratio, audio_ratio) if r is not None]
            if ratio_candidates:
                ratio = max(ratio_candidates)
                self.stats["mag_calibration_ratios"].append(ratio)
                record["mag_calibration_ratio"] = ratio
            if video_ratio is not None:
                self.stats["video_mag_calibration_ratios"].append(video_ratio)
                record["video_mag_calibration_ratio"] = video_ratio
            if audio_ratio is not None:
                self.stats["audio_mag_calibration_ratios"].append(audio_ratio)
                record["audio_mag_calibration_ratio"] = audio_ratio
        self.consecutive_skips = 0
        self.accumulated_metric = 0.0
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_mag_steps = 0
        self.stats["computed_steps"].append(self.step_index)
        self._finish_record(record)

    def _finish_skip(self, record: dict[str, Any]) -> None:
        self.consecutive_skips += 1
        record["consecutive_skips_after"] = self.consecutive_skips
        self.stats["skipped_steps"].append(self.step_index)
        self._finish_record(record)

    def _finish_record(self, record: dict[str, Any]) -> None:
        self.stats["records"].append(record)
        self.step_index += 1

    def _invalidate(self, reason: str, new_shape_key: tuple[Any, ...]) -> None:
        self.stats["invalidations"].append({"step": self.step_index, "reason": reason})
        self.shape_key = new_shape_key
        self.cached_video_velocity = None
        self.cached_audio_velocity = None
        self.previous_video_sample = None
        self.previous_audio_sample = None
        self.previous_video_velocity_mag = None
        self.previous_audio_velocity_mag = None
        self.consecutive_skips = 0
        self.accumulated_metric = 0.0
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_mag_steps = 0

    def _past_retention(self) -> bool:
        retention_steps = int(self.config.retention_ratio * self.config.total_steps + 0.5)
        return self.step_index >= retention_steps

    def _has_required_residuals(self, video_latent: object | None, audio_latent: object | None) -> bool:
        if video_latent is not None and self.cached_video_velocity is None:
            return False
        if audio_latent is not None and self.cached_audio_velocity is None:
            return False
        return video_latent is not None or audio_latent is not None

    def _teacache_metric(self, video_latent: object | None, audio_latent: object | None) -> float | None:
        metrics = []
        video_sample = self._sample(video_latent)
        audio_sample = self._sample(audio_latent)
        if video_sample is not None and self.previous_video_sample is not None:
            metrics.append(self._relative_l1(video_sample, self.previous_video_sample))
        if audio_sample is not None and self.previous_audio_sample is not None:
            metrics.append(self._relative_l1(audio_sample, self.previous_audio_sample))
        return max(metrics) if metrics else None

    def _store_metric_samples(self, video_latent: object | None, audio_latent: object | None) -> None:
        video_sample = self._sample(video_latent)
        audio_sample = self._sample(audio_latent)
        self.previous_video_sample = video_sample.detach().clone() if video_sample is not None else None
        self.previous_audio_sample = audio_sample.detach().clone() if audio_sample is not None else None

    def _sample(self, tensor: object | None) -> object | None:
        if tensor is None or not hasattr(tensor, "detach"):
            return None
        sample = tensor.detach().flatten()
        stride = self.config.metric_element_stride
        if stride > 1:
            sample = sample[::stride]
        return sample

    @staticmethod
    def _relative_l1(current: object, previous: object) -> float:
        # This scalar sync is the adaptive-cache cost; force-skip and MagCache avoid it.
        diff = (current.float() - previous.float()).abs().mean()
        denom = previous.float().abs().mean().clamp_min(1e-6)
        return float((diff / denom).detach().cpu())

    @staticmethod
    def _shape_key(video_latent: object | None, audio_latent: object | None) -> tuple[Any, ...]:
        return (
            tuple(video_latent.shape) if video_latent is not None and hasattr(video_latent, "shape") else None,
            str(video_latent.dtype) if video_latent is not None and hasattr(video_latent, "dtype") else None,
            str(video_latent.device) if video_latent is not None and hasattr(video_latent, "device") else None,
            tuple(audio_latent.shape) if audio_latent is not None and hasattr(audio_latent, "shape") else None,
            str(audio_latent.dtype) if audio_latent is not None and hasattr(audio_latent, "dtype") else None,
            str(audio_latent.device) if audio_latent is not None and hasattr(audio_latent, "device") else None,
        )

    @staticmethod
    def _velocity(denoised: object | None, latent: object | None, sigma: object) -> object | None:
        if denoised is None or latent is None:
            return None
        if not hasattr(denoised, "shape") or not hasattr(latent, "shape") or denoised.shape != latent.shape:
            return None
        sigma_view = _ResidualDenoiserCache._sigma_view(sigma, latent)
        return ((latent - denoised).to(dtype=sigma_view.dtype) / sigma_view.clamp_min(1e-6)).to(dtype=latent.dtype).detach()

    @staticmethod
    def _apply_velocity(latent: object | None, velocity: object | None, sigma: object) -> object | None:
        if latent is None:
            return None
        if velocity is None:
            raise RuntimeError("attempted residual-cache skip without a cached velocity")
        if velocity.shape != latent.shape:
            raise RuntimeError(f"cached velocity shape {tuple(velocity.shape)} does not match latent {tuple(latent.shape)}")
        if velocity.device != latent.device:
            velocity = velocity.to(device=latent.device)
        sigma_view = _ResidualDenoiserCache._sigma_view(sigma, latent)
        return (latent.to(dtype=sigma_view.dtype) - velocity.to(dtype=sigma_view.dtype) * sigma_view).to(dtype=latent.dtype)

    @staticmethod
    def _sigma_view(sigma: object, latent: object) -> object:
        import torch

        sigma_tensor = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32)
        if sigma_tensor.dim() == 0:
            return sigma_tensor.reshape(*([1] * latent.dim()))
        if sigma_tensor.dim() == 1 and latent.dim() >= 1 and int(sigma_tensor.shape[0]) == int(latent.shape[0]):
            return sigma_tensor.reshape(int(sigma_tensor.shape[0]), *([1] * (latent.dim() - 1)))
        return sigma_tensor

    @staticmethod
    def _velocity_magnitude(velocity: object | None) -> float | None:
        if velocity is None:
            return None
        value = velocity.float()
        if value.ndim >= 2:
            mag = value.norm(dim=-1).mean()
        else:
            mag = value.abs().mean()
        return float(mag.detach().cpu())

    def _velocity_mag_and_ratio(
        self,
        velocity: object | None,
        previous_mag: float | None,
    ) -> tuple[float | None, float | None]:
        mag = self._velocity_magnitude(velocity)
        if mag is None:
            return None, None
        if previous_mag is None:
            return mag, 1.0
        return mag, mag / max(previous_mag, 1e-8)


class _BatchSimpleDenoiser:
    """SimpleDenoiser variant that expands scalar sigma for B>1 benchmarking."""

    def __init__(
        self,
        v_context: object | None,
        a_context: object | None,
        batch_size: int,
        *,
        residual_cache_config: _ResidualCacheConfig | None = None,
        residual_cache_stats: dict[str, Any] | None = None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.batch_size = batch_size
        self.residual_cache = (
            _ResidualDenoiserCache(residual_cache_config, residual_cache_stats)
            if residual_cache_config is not None
            else None
        )

    def __call__(
        self,
        transformer: object,
        video_state: object | None,
        audio_state: object | None,
        sigmas: object,
        step_index: int,
    ) -> tuple[object | None, object | None]:
        import torch
        from ltx_pipelines.utils.helpers import modality_from_latent_state
        from ltx_pipelines.utils.types import DenoisedLatentResult

        sigma = sigmas[step_index]
        if isinstance(sigma, torch.Tensor) and sigma.dim() == 0:
            sigma = sigma.expand(self.batch_size)
        pos_video = (
            modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        )
        pos_audio = (
            modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        )
        cache_decision = None
        if self.residual_cache is not None:
            cache_decision = self.residual_cache.begin_step(
                getattr(pos_video, "latent", None),
                getattr(pos_audio, "latent", None),
                sigma,
            )
            if cache_decision.skip:
                return (
                    DenoisedLatentResult.result_or_none(denoised=cache_decision.video),
                    DenoisedLatentResult.result_or_none(denoised=cache_decision.audio),
                )
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        if self.residual_cache is not None and cache_decision is not None:
            self.residual_cache.finish_compute(
                video_latent=getattr(pos_video, "latent", None),
                audio_latent=getattr(pos_audio, "latent", None),
                sigma=sigma,
                denoised_video=denoised_video,
                denoised_audio=denoised_audio,
                record=cache_decision.record,
            )
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )


def _build_quantization_policy(name: str, checkpoint_path: str) -> object | None:
    if name == "none":
        return None
    if name == "fp8-cast":
        from ltx_core.quantization.fp8_cast import build_policy

        return build_policy(checkpoint_path)
    if name == "fp8-scaled-mm":
        from ltx_core.quantization.fp8_scaled_mm import build_policy

        return build_policy(checkpoint_path)
    raise ValueError(f"unknown quantization policy: {name}")


def _build_compilation_config(args: argparse.Namespace) -> object | None:
    if not args.compile:
        return None
    from ltx_core.model.transformer.compiling import CompilationConfig

    dynamic = {
        "none": None,
        "true": True,
        "false": False,
    }[args.compile_dynamic]
    mode = None if args.compile_mode in {"", "none"} else args.compile_mode
    return CompilationConfig(
        mode=mode,
        backend=args.compile_backend,
        fullgraph=args.compile_fullgraph,
        dynamic=dynamic,
    )


def _patch_compile_for_static_shapes(args: argparse.Namespace) -> None:
    if not args.compile or args.compile_dynamic_marking:
        return
    import ltx_core.model.transformer.compiling as compiling

    class _StaticShapeProcessor:
        def __init__(self, inner: object) -> None:
            self.inner = inner

        def __call__(self, *processor_args: object, **processor_kwargs: object) -> object:
            return self.inner(*processor_args, **processor_kwargs)

    compiling._SeqDynamicMarkingProcessor = _StaticShapeProcessor


def _patch_sampler_progress(args: argparse.Namespace) -> None:
    if args.sampler_progress:
        return
    import ltx_pipelines.utils.samplers as samplers

    def _no_progress(iterable: object, *unused_args: object, **unused_kwargs: object) -> object:
        return iterable

    samplers.tqdm = _no_progress


def _patch_compile_for_fp8_linear(args: argparse.Namespace) -> None:
    if not args.compile or args.compile_trace_fp8_linear or args.quantization != "fp8-scaled-mm":
        return
    import torch
    from ltx_core.quantization.fp8_scaled_mm import FP8Linear

    FP8Linear.forward = torch.compiler.disable(FP8Linear.forward)


def _patch_triton_adazero(args: argparse.Namespace) -> None:
    if not args.triton_adazero:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_adazero

    patch_ltx_adazero()


def _patch_triton_fp8_quantize(args: argparse.Namespace) -> None:
    if not args.triton_fp8_quant:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_fp8_quantize

    patch_ltx_fp8_quantize(
        bias_epilogue=args.fp8_scaled_mm_bias_epilogue,
        triton_bias_add=args.triton_fp8_bias_add,
    )


def _patch_triton_ffn_gelu_fp8_quant(args: argparse.Namespace) -> None:
    if not args.triton_ffn_gelu_fp8_quant:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_ffn_gelu_fp8_quant

    patch_ltx_ffn_gelu_fp8_quant()


def _collect_ffn_gelu_fp8_quant_calls(args: argparse.Namespace) -> int | None:
    if not args.triton_ffn_gelu_fp8_quant:
        return None
    from ltx_serve.triton_ltx_ops import collect_ffn_gelu_fp8_quant_calls

    return collect_ffn_gelu_fp8_quant_calls()


def _patch_triton_cross_attn_adaln(args: argparse.Namespace) -> None:
    if not args.triton_cross_attn_adaln:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_cross_attention_adaln

    patch_ltx_cross_attention_adaln()


def _collect_cross_attn_adaln_calls(args: argparse.Namespace) -> int | None:
    if not args.triton_cross_attn_adaln:
        return None
    from ltx_serve.triton_ltx_ops import collect_cross_attention_adaln_calls

    return collect_cross_attention_adaln_calls()


def _patch_triton_video_preattention(args: argparse.Namespace) -> None:
    if not args.triton_video_preattention:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_preattention

    patch_ltx_video_preattention(checks=args.triton_video_preattention_checks, mode=args.triton_video_preattention_mode)


def _patch_torch_addcmul_residuals(args: argparse.Namespace) -> None:
    if not args.torch_addcmul_residuals:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_block_residual_addcmul

    patch_ltx_block_residual_addcmul()


def _patch_triton_residual_gate(args: argparse.Namespace) -> None:
    if not args.triton_residual_gate:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_block_residual_triton

    patch_ltx_block_residual_triton()


def _patch_triton_simple_residual_gate(args: argparse.Namespace) -> None:
    if not args.triton_simple_residual_gate:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_block_simple_residual_triton

    patch_ltx_block_simple_residual_triton()


def _patch_triton_video_ada_values(args: argparse.Namespace) -> None:
    if not args.triton_video_ada_values:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_ada_values

    patch_ltx_video_ada_values()


def _patch_triton_audio_ada_values(args: argparse.Namespace) -> None:
    if not args.triton_audio_ada_values:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_audio_ada_values

    patch_ltx_audio_ada_values()


def _patch_triton_video_text_adaln(args: argparse.Namespace) -> None:
    if not args.triton_video_text_adaln:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_text_adaln

    patch_ltx_video_text_adaln()


def _patch_triton_video_text_context_adaln(args: argparse.Namespace) -> None:
    if not args.triton_video_text_context_adaln:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_text_context_adaln

    patch_ltx_video_text_context_adaln()


def _patch_uniform_timestep_adaln(args: argparse.Namespace) -> None:
    if not args.uniform_timestep_adaln:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_uniform_timestep_adaln

    patch_ltx_uniform_timestep_adaln()


def _collect_uniform_timestep_adaln_calls(args: argparse.Namespace) -> int | None:
    if not args.uniform_timestep_adaln:
        return None
    from ltx_serve.triton_ltx_ops import collect_uniform_timestep_adaln_calls

    return collect_uniform_timestep_adaln_calls()


def _patch_rope_embedding_cache(args: argparse.Namespace) -> None:
    if not args.cache_rope_embeddings:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_rope_embedding_cache

    patch_ltx_rope_embedding_cache()


def _collect_rope_embedding_cache_stats(args: argparse.Namespace) -> dict[str, int] | None:
    if not args.cache_rope_embeddings:
        return None
    from ltx_serve.triton_ltx_ops import collect_rope_embedding_cache_stats

    return collect_rope_embedding_cache_stats()


def _patch_triton_video_out_bias_residual(args: argparse.Namespace) -> None:
    if not args.triton_video_out_bias_residual:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_out_bias_residual

    patch_ltx_video_out_bias_residual()


def _patch_triton_video_ffn_out_bias_residual(args: argparse.Namespace) -> None:
    if not args.triton_video_ffn_out_bias_residual:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_ffn_out_bias_residual

    patch_ltx_video_ffn_out_bias_residual()


def _patch_triton_video_qk_bias_preattention(args: argparse.Namespace) -> None:
    if not args.triton_video_qk_bias_preattention:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_qk_bias_preattention

    patch_ltx_video_qk_bias_preattention()


def _patch_triton_video_qkv_quant_reuse(args: argparse.Namespace) -> None:
    if not args.triton_video_qkv_quant_reuse:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_qkv_quant_reuse

    patch_ltx_video_qkv_quant_reuse()


def _patch_triton_video_qkv_grouped_mm(args: argparse.Namespace) -> None:
    if not args.triton_video_qkv_grouped_mm:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_qkv_grouped_mm

    patch_ltx_video_qkv_grouped_mm(checks=args.triton_video_qkv_grouped_mm_checks)


def _patch_triton_video_qk_grouped_mm(args: argparse.Namespace) -> None:
    if not args.triton_video_qk_grouped_mm:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_qk_grouped_mm

    patch_ltx_video_qk_grouped_mm(checks=args.triton_video_qk_grouped_mm_checks)


def _patch_triton_video_qkv_packed_linear(args: argparse.Namespace) -> None:
    if not args.triton_video_qkv_packed_linear:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_qkv_packed_linear

    patch_ltx_video_qkv_packed_linear()


def _patch_triton_video_qkv_packed_requant(args: argparse.Namespace) -> None:
    if not args.triton_video_qkv_packed_requant:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_qkv_packed_requant

    patch_ltx_video_qkv_packed_requant(checks=args.triton_video_qkv_packed_requant_checks)


def _collect_video_qkv_packed_linear_stats(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.triton_video_qkv_packed_linear:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_qkv_packed_linear_stats

    return collect_video_qkv_packed_linear_stats()


def _collect_video_qkv_packed_requant_stats(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.triton_video_qkv_packed_requant:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_qkv_packed_requant_stats

    return collect_video_qkv_packed_requant_stats()


def _collect_video_qkv_grouped_mm_stats(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.triton_video_qkv_grouped_mm:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_qkv_grouped_mm_stats

    return collect_video_qkv_grouped_mm_stats()


def _collect_video_qk_grouped_mm_stats(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.triton_video_qk_grouped_mm:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_qk_grouped_mm_stats

    return collect_video_qk_grouped_mm_stats()


def _parse_token_counts(raw: str) -> tuple[int, ...] | None:
    raw = raw.strip()
    if not raw:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _patch_triton_video_msa_branch(args: argparse.Namespace) -> None:
    if not args.triton_video_msa_branch:
        return
    from ltx_serve.triton_ltx_ops import patch_ltx_video_msa_branch

    patch_ltx_video_msa_branch(
        token_counts=_parse_token_counts(args.triton_video_msa_branch_tokens),
        profile=args.triton_video_msa_branch_profile,
        mode=args.triton_video_msa_branch_mode,
        gate_mul=args.triton_video_gate_mul,
    )


def _collect_video_msa_branch_stats(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.triton_video_msa_branch:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_msa_branch_stats

    return collect_video_msa_branch_stats()


def _collect_video_preattention_checks(args: argparse.Namespace) -> list[dict[str, float]]:
    if not args.triton_video_preattention:
        return []
    from ltx_serve.triton_ltx_ops import collect_video_preattention_checks

    return collect_video_preattention_checks()


def _collect_video_ada_values_calls(args: argparse.Namespace) -> int | None:
    if not args.triton_video_ada_values:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_ada_values_calls

    return collect_video_ada_values_calls()


def _collect_audio_ada_values_calls(args: argparse.Namespace) -> int | None:
    if not args.triton_audio_ada_values:
        return None
    from ltx_serve.triton_ltx_ops import collect_audio_ada_values_calls

    return collect_audio_ada_values_calls()


def _collect_video_qkv_quant_reuse_calls(args: argparse.Namespace) -> int | None:
    if not args.triton_video_qkv_quant_reuse:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_qkv_quant_reuse_calls

    return collect_video_qkv_quant_reuse_calls()


def _collect_video_text_adaln_calls(args: argparse.Namespace) -> int | None:
    if not args.triton_video_text_adaln:
        return None
    from ltx_serve.triton_ltx_ops import collect_video_text_adaln_calls

    return collect_video_text_adaln_calls()


def _patch_flash_attention3_options(args: argparse.Namespace) -> None:
    if (
        args.fa3_num_splits == 1
        and args.fa3_attention_chunk == 0
        and args.fa3_sm_margin == 0
        and args.fa3_video_window_left == -1
        and args.fa3_video_window_right == -1
        and not args.fa3_video_fp8_attention
        and not args.fa3_video_prealloc_output
    ):
        return

    from ltx_core.model.transformer import attention as attention_mod

    def _call(
        self: object,
        q: object,
        k: object,
        v: object,
        heads: int,
    ) -> object:
        if attention_mod.flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        ref_out = None
        check_fp8_video = (
            args.fa3_video_fp8_attention
            and args.fa3_video_fp8_checks > len(_FP8_VIDEO_ATTENTION_CHECKS)
            and _is_ltx_video_self_attention(q, k, v)
        )
        check_window_video = (
            (args.fa3_video_window_left != -1 or args.fa3_video_window_right != -1)
            and args.fa3_video_window_checks > len(_WINDOW_VIDEO_ATTENTION_CHECKS)
            and _is_ltx_video_self_attention(q, k, v)
        )
        if check_fp8_video or check_window_video:
            ref_out = attention_mod.flash_attn_interface.flash_attn_func(
                q,
                k,
                v,
                attention_chunk=args.fa3_attention_chunk,
                num_splits=args.fa3_num_splits,
                sm_margin=args.fa3_sm_margin,
            )

        q_descale = k_descale = v_descale = None
        window_size = (-1, -1)
        if _is_ltx_video_self_attention(q, k, v):
            window_size = (args.fa3_video_window_left, args.fa3_video_window_right)
        if args.fa3_video_fp8_attention and _is_ltx_video_self_attention(q, k, v):
            components = set(args.fa3_video_fp8_components)
            if "q" in components:
                q, q_descale = _to_fp8_per_head(q)
            if "k" in components:
                k, k_descale = _to_fp8_per_head(k)
            if "v" in components:
                v, v_descale = _to_fp8_per_head(v)

        if (
            args.fa3_video_prealloc_output
            and q_descale is None
            and k_descale is None
            and v_descale is None
            and _is_ltx_video_self_attention(q, k, v)
        ):
            out = _fa3_preallocated_video_attention(attention_mod.flash_attn_interface, q, k, v, args)
        else:
            out = attention_mod.flash_attn_interface.flash_attn_func(
                q,
                k,
                v,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                window_size=window_size,
                attention_chunk=args.fa3_attention_chunk,
                num_splits=args.fa3_num_splits,
                sm_margin=args.fa3_sm_margin,
            )
        if ref_out is not None:
            if check_fp8_video:
                _record_fp8_video_attention_check(ref_out, out)
            if check_window_video:
                _record_window_video_attention_check(ref_out, out)
        out = out.reshape(b, -1, heads * dim_head)
        return out

    attention_mod.FlashAttention3.__call__ = _call


_FA3_VIDEO_OUT_CACHE: dict[tuple[object, ...], object] = {}


def _fa3_preallocated_video_attention(flash_attn_interface: object, q: object, k: object, v: object, args: argparse.Namespace) -> object:
    import torch

    key = (q.device, q.dtype, tuple(q.shape))
    out = _FA3_VIDEO_OUT_CACHE.get(key)
    if out is None:
        out = torch.empty_like(q)
        _FA3_VIDEO_OUT_CACHE[key] = out
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
        -1,
        -1,
        args.fa3_attention_chunk,
        0.0,
        True,
        None,
        args.fa3_num_splits,
        None,
        args.fa3_sm_margin,
    )
    return y


def _is_ltx_video_self_attention(q: object, k: object, v: object) -> bool:
    import torch

    return (
        q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.shape[0] == 1
        and q.shape[1] == k.shape[1] == v.shape[1]
        and q.shape[1] >= 1024
        and q.shape[2] == 32
        and k.shape[2] == 32
        and v.shape[2] == 32
        and q.shape[3] == 128
        and k.shape[3] == 128
        and v.shape[3] == 128
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
    )


def _to_fp8_per_head(x: object) -> tuple[object, object]:
    from ltx_serve.triton_ltx_ops import fused_fp8_quantize_e4m3_per_head_4d

    return fused_fp8_quantize_e4m3_per_head_4d(x)


_FP8_VIDEO_ATTENTION_CHECKS: list[dict[str, float]] = []


def _record_fp8_video_attention_check(ref: object, out: object) -> None:
    import torch

    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    _FP8_VIDEO_ATTENTION_CHECKS.append(
        {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
            "mean_rel": float((diff / denom).mean().item()),
        }
    )


def _collect_fp8_video_attention_checks() -> list[dict[str, float]]:
    return list(_FP8_VIDEO_ATTENTION_CHECKS)


_WINDOW_VIDEO_ATTENTION_CHECKS: list[dict[str, float]] = []


def _record_window_video_attention_check(ref: object, out: object) -> None:
    import torch

    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    _WINDOW_VIDEO_ATTENTION_CHECKS.append(
        {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
            "mean_rel": float((diff / denom).mean().item()),
        }
    )


def _collect_window_video_attention_checks() -> list[dict[str, float]]:
    return list(_WINDOW_VIDEO_ATTENTION_CHECKS)


_ATTENTION_SHAPE_EVENTS: list[dict[str, Any]] = []


def _patch_attention_shape_profile(args: argparse.Namespace) -> None:
    if not args.attention_shape_profile:
        return

    import torch
    from ltx_core.model.transformer import attention as attention_mod

    def _wrap(cls: type, backend: str) -> None:
        original = cls.__call__

        def _profiled_call(self: object, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
            if not q.is_cuda:
                return original(self, q, k, v, heads)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = original(self, q, k, v, heads)
            end.record()
            _ATTENTION_SHAPE_EVENTS.append(
                {
                    "backend": backend,
                    "start": start,
                    "end": end,
                    "q_tokens": int(q.shape[1]),
                    "k_tokens": int(k.shape[1]),
                    "v_tokens": int(v.shape[1]),
                    "heads": int(heads),
                    "head_dim": int(q.shape[-1] // heads),
                    "q_dtype": str(q.dtype),
                    "v_dtype": str(v.dtype),
                }
            )
            return out

        cls.__call__ = _profiled_call

    if hasattr(attention_mod, "FlashAttention3"):
        _wrap(attention_mod.FlashAttention3, "flash-attn-3")
    if hasattr(attention_mod, "FlashAttention4"):
        _wrap(attention_mod.FlashAttention4, "flash-attn-4")


def _collect_attention_shape_profile(torch: object) -> list[dict[str, Any]]:
    if not _ATTENTION_SHAPE_EVENTS:
        return []
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in _ATTENTION_SHAPE_EVENTS:
        key = (
            event["backend"],
            event["q_tokens"],
            event["k_tokens"],
            event["v_tokens"],
            event["heads"],
            event["head_dim"],
            event["q_dtype"],
            event["v_dtype"],
        )
        elapsed_ms = float(event["start"].elapsed_time(event["end"]))
        row = grouped.setdefault(
            key,
            {
                "backend": event["backend"],
                "q_tokens": event["q_tokens"],
                "k_tokens": event["k_tokens"],
                "v_tokens": event["v_tokens"],
                "heads": event["heads"],
                "head_dim": event["head_dim"],
                "q_dtype": event["q_dtype"],
                "v_dtype": event["v_dtype"],
                "instances": 0,
                "total_ms": 0.0,
                "min_ms": None,
                "max_ms": 0.0,
            },
        )
        row["instances"] += 1
        row["total_ms"] += elapsed_ms
        row["min_ms"] = elapsed_ms if row["min_ms"] is None else min(row["min_ms"], elapsed_ms)
        row["max_ms"] = max(row["max_ms"], elapsed_ms)

    rows = list(grouped.values())
    for row in rows:
        row["avg_ms"] = row["total_ms"] / row["instances"]
    rows.sort(key=lambda item: item["total_ms"], reverse=True)
    return rows


class _BlockProfiler:
    def __init__(self, torch: object) -> None:
        self._torch = torch
        self.enabled = False
        self.events: list[tuple[int, str, Any, Any]] = []

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def record(self, index: int, module: object, call: object, video: object, audio: object) -> object:
        if not self.enabled or not self._torch.cuda.is_available():
            return call(video, audio)
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        result = call(video, audio)
        end.record()
        q = getattr(getattr(module, "attn1", None), "to_q", None)
        q_class = q.__class__.__name__ if q is not None else "unknown"
        self.events.append((index, q_class, start, end))
        return result

    def summary(self) -> list[dict[str, Any]]:
        if not self.events:
            return []
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        by_index: dict[int, dict[str, Any]] = {}
        for index, q_class, start, end in self.events:
            row = by_index.setdefault(
                index,
                {
                    "block_index": index,
                    "q_projection_class": q_class,
                    "calls": 0,
                    "total_ms": 0.0,
                    "min_ms": None,
                    "max_ms": 0.0,
                },
            )
            elapsed_ms = float(start.elapsed_time(end))
            row["calls"] += 1
            row["total_ms"] += elapsed_ms
            row["min_ms"] = elapsed_ms if row["min_ms"] is None else min(row["min_ms"], elapsed_ms)
            row["max_ms"] = max(row["max_ms"], elapsed_ms)
        rows = list(by_index.values())
        for row in rows:
            row["avg_ms"] = row["total_ms"] / row["calls"] if row["calls"] else 0.0
        rows.sort(key=lambda item: item["block_index"])
        return rows


def _install_block_profiler(args: argparse.Namespace, transformer: object, torch: object) -> _BlockProfiler | None:
    if not args.block_profile:
        return None
    from types import MethodType

    profiler = _BlockProfiler(torch)
    blocks = [m for m in transformer.modules() if m.__class__.__name__ == "BasicAVTransformerBlock"]
    for index, block in enumerate(blocks):
        original = block.forward

        def _forward(
            self: object,
            video: object | None,
            audio: object | None,
            *,
            _index: int = index,
            _call: object = original,
        ) -> object:
            return profiler.record(_index, self, _call, video, audio)

        block.forward = MethodType(_forward, block)
    return profiler


def _pin_attention(stage: object, attention: str) -> object:
    if attention == "automatic":
        return stage

    import copy
    import dataclasses

    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction

    attention_map = {
        "flash-attn-3": AttentionFunction.FLASH_ATTENTION_3,
        "flash-attn-4": AttentionFunction.FLASH_ATTENTION_4,
        "sdpa-flash": AttentionFunction.SDPA_FLASH,
        "sdpa-cudnn": AttentionFunction.SDPA_CUDNN,
        "sdpa-efficient": AttentionFunction.SDPA_EFFICIENT,
        "sdpa-math": AttentionFunction.SDPA_MATH,
    }
    masked_map = {
        "flash-attn-3": MaskedAttentionFunction.AUTOMATIC,
        "flash-attn-4": MaskedAttentionFunction.AUTOMATIC,
        "sdpa-flash": MaskedAttentionFunction.AUTOMATIC,
        "sdpa-cudnn": MaskedAttentionFunction.SDPA_CUDNN,
        "sdpa-efficient": MaskedAttentionFunction.SDPA_EFFICIENT,
        "sdpa-math": MaskedAttentionFunction.SDPA_MATH,
    }
    op = set_attention_module_op(attention_map[attention], masked_map[attention])
    new_stage = copy.copy(stage)
    new_stage._transformer_builder = stage._transformer_builder.with_module_ops(
        (*stage._transformer_builder.module_ops, op)
    )
    if getattr(stage, "_offload_mode", None) is not None and getattr(stage, "_offload_mode").value != "none":
        new_stage._streaming_builder = dataclasses.replace(
            stage._streaming_builder,
            module_ops=(*stage._streaming_builder.module_ops, op),
    )
    return new_stage


def _profiler_context(args: argparse.Namespace, torch: object, run_index: int) -> object:
    if not args.torch_profile or run_index < args.warmup:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def _write_profile(
    args: argparse.Namespace,
    torch: object,
    profiler: object,
    output_dir: Path,
    bucket_name: str,
    run_index: int,
) -> dict[str, str]:
    profile_dir = Path(args.profile_dir) if args.profile_dir else output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{bucket_name}_{args.quantization}_{args.attention}_steps{args.steps}_run{run_index}"
    trace_path = profile_dir / f"{stem}.chrome_trace.json"
    table_path = profile_dir / f"{stem}.cuda_table.txt"
    shape_table_path = profile_dir / f"{stem}.cuda_shapes_table.txt"
    profiler.export_chrome_trace(str(trace_path))
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    table = profiler.key_averages().table(sort_by=sort_key, row_limit=args.profile_row_limit)
    table_path.write_text(table, encoding="utf-8")
    shape_table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by=sort_key,
        row_limit=args.profile_row_limit,
    )
    shape_table_path.write_text(shape_table, encoding="utf-8")
    return {"trace": str(trace_path), "table": str(table_path), "shape_table": str(shape_table_path)}


def _resolve_attention(attention: str) -> str:
    if attention == "h100-fa3":
        try:
            import flash_attn_interface as _flash_attn_interface  # noqa: F401
        except Exception:
            return _resolve_attention("h100-fa4")
        return "flash-attn-3"
    if attention != "h100-fa4":
        return attention
    try:
        from flash_attn.cute import flash_attn_func as _flash_attn_func  # noqa: F401
    except Exception:
        return "sdpa-cudnn"
    return "flash-attn-4"


def _sigmas(steps: int, distilled_sigmas: object) -> object:
    import torch

    if steps == 8:
        return distilled_sigmas
    if steps == 1:
        return torch.tensor([1.0, 0.0])
    if steps < 1 or steps > 8:
        raise ValueError("--steps must be between 1 and 8 for the distilled benchmark")
    indices = torch.linspace(0, len(distilled_sigmas) - 1, steps + 1).round().long()
    sigmas = distilled_sigmas[indices]
    sigmas[-1] = 0.0
    return sigmas


if __name__ == "__main__":
    main()
