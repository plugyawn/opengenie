from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from ltx_serve.buckets import LIVE_BUCKET_NAME
from ltx_serve.remote_official_worker import (
    OfficialEngine,
    _apply_request_residual_cache_overrides,
    _apply_request_spd_overrides,
    _build_latent_continuation_conditionings,
    _decoded_frames_for_video_latent_frames,
    _prepare_continuation_specs,
    _trim_crop_video_chunk,
    _video_latent_frames_for_requested_overlap,
    build_engine_args,
)
from ltx_serve import remote_official_worker
from ltx_serve.schemas import GenerateRequest


def test_official_worker_pins_distilled_quality_contract() -> None:
    args = build_engine_args()

    assert args.bucket == LIVE_BUCKET_NAME
    assert args.steps == 8
    assert args.quantization == "fp8-scaled-mm"
    assert args.attention == "h100-fa3"
    assert args.hot_prompt_encoder is True
    assert args.hot_decoders is True
    assert args.sampler_progress is False


def test_official_worker_pins_known_good_triton_flags() -> None:
    args = build_engine_args()

    assert args.triton_adazero is True
    assert args.triton_fp8_quant is True
    assert args.triton_fp8_bias_add is True
    assert args.triton_video_preattention is True
    assert args.triton_video_preattention_mode == "dual"
    assert args.triton_simple_residual_gate is True
    assert args.triton_video_ada_values is True
    assert args.triton_video_text_adaln is True
    assert args.uniform_timestep_adaln is False
    assert args.cache_rope_embeddings is False
    assert args.triton_video_out_bias_residual is True

    assert args.triton_audio_ada_values is True
    assert args.triton_video_ffn_out_bias_residual is False
    assert args.triton_video_qk_bias_preattention is False
    assert args.triton_video_text_context_adaln is False
    assert args.triton_video_qkv_quant_reuse is False
    assert args.triton_video_msa_branch is False
    assert args.triton_video_qk_grouped_mm is False
    assert args.triton_video_msa_branch_tokens == ""
    assert args.triton_video_msa_branch_mode == "generic"
    assert args.allow_quality_risk_msa_bf16_out is False
    assert args.fa3_video_prealloc_output is False
    assert args.fa3_video_fp8_attention is False
    assert args.allow_quality_risk_fp8_video_attention is False
    assert args.allow_quality_risk_pointwise_fusions is False


def test_official_worker_allows_video_msa_branch_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX_WORKER_TRITON_VIDEO_MSA_BRANCH", "1")
    monkeypatch.setenv("LTX_WORKER_TRITON_VIDEO_MSA_BRANCH_TOKENS", "4032,4368")

    args = build_engine_args()

    assert args.triton_video_msa_branch is True
    assert args.triton_video_msa_branch_tokens == "4032,4368"
    assert args.triton_video_msa_branch_mode == "generic"


def test_official_worker_allows_audio_ada_pin_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX_WORKER_TRITON_AUDIO_ADA_VALUES", "0")

    args = build_engine_args()

    assert args.triton_audio_ada_values is False


def test_residual_cache_request_override_is_explicit() -> None:
    args = build_engine_args()
    args.residual_cache_mode = "calibrate"

    _apply_request_residual_cache_overrides(args, GenerateRequest(prompt="inherit cache args"))

    assert args.residual_cache_mode == "calibrate"


def test_prompt_context_cache_reuses_exact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    args = build_engine_args()
    args.prompt = "same prompt"
    args.batch_size = 1
    engine = OfficialEngine(args)
    engine.pipeline = object()
    calls = 0

    def fake_encode_prompt_contexts(args, pipeline, torch_module):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return 0.5, torch.tensor([[calls]], dtype=torch.float32), torch.tensor([[calls + 10]], dtype=torch.float32)

    monkeypatch.setattr(remote_official_worker, "_encode_prompt_contexts", fake_encode_prompt_contexts)

    prompt_s0, video0, audio0, hit0 = engine._encode_prompt_contexts_cached(args, torch)
    prompt_s1, video1, audio1, hit1 = engine._encode_prompt_contexts_cached(args, torch)

    assert prompt_s0 == 0.5
    assert hit0 is False
    assert hit1 is True
    assert calls == 1
    assert video1 is video0
    assert audio1 is audio0


def test_residual_cache_request_override_pins_magcache_policy() -> None:
    args = build_engine_args()
    req = GenerateRequest(
        prompt="magcache override",
        residual_cache_mode="magcache",
        allow_quality_risk_residual_cache=True,
        residual_cache_threshold=0.02,
        residual_cache_max_skips=1,
        residual_cache_retention_ratio=0.25,
        residual_cache_metric_element_stride=64,
        residual_cache_mag_ratios=[1.1, 0.9, 1.0],
    )

    _apply_request_residual_cache_overrides(args, req)

    assert args.residual_cache_mode == "magcache"
    assert args.allow_quality_risk_residual_cache is True
    assert args.residual_cache_threshold == 0.02
    assert args.residual_cache_max_skips == 1
    assert args.residual_cache_retention_ratio == 0.25
    assert args.residual_cache_metric_element_stride == 64
    assert args.residual_cache_mag_ratios == "1.1,0.9,1"


def test_magcache_request_requires_ratios() -> None:
    args = build_engine_args()
    req = GenerateRequest(
        prompt="bad magcache",
        residual_cache_mode="magcache",
        allow_quality_risk_residual_cache=True,
    )

    with pytest.raises(Exception, match="magcache mode requires"):
        _apply_request_residual_cache_overrides(args, req)


def test_spd_request_override_requires_quality_risk_opt_in() -> None:
    args = build_engine_args()
    req = GenerateRequest(prompt="spd override", spd=True)

    with pytest.raises(Exception, match="quality-risk"):
        _apply_request_spd_overrides(args, req, remote_official_worker.BUCKETS[args.bucket])


def test_spd_request_override_sets_initial_highfreq_source() -> None:
    args = build_engine_args()
    req = GenerateRequest(
        prompt="spd override",
        spd=True,
        allow_quality_risk_spd=True,
        spd_scale=0.5,
        spd_transition_step=5,
        spd_taper=8,
        spd_initial_dct_downscale=True,
        spd_highfreq_noise=True,
        spd_highfreq_source="initial",
    )

    _apply_request_spd_overrides(args, req, remote_official_worker.BUCKETS[args.bucket])

    assert args.spd is True
    assert args.spd_scale == 0.5
    assert args.spd_transition_step == 5
    assert args.spd_taper == 8
    assert args.spd_initial_dct_downscale is True
    assert args.spd_highfreq_noise is True
    assert args.spd_highfreq_source == "initial"


def test_spd_request_override_rejects_continuation() -> None:
    args = build_engine_args()
    req = GenerateRequest(
        prompt="spd continuation",
        spd=True,
        allow_quality_risk_spd=True,
        continuation_video_path="/tmp/prev.mp4",
        continuation_frames=22,
    )

    with pytest.raises(Exception, match="pure T2V"):
        _apply_request_spd_overrides(args, req, remote_official_worker.BUCKETS[args.bucket])


def test_official_worker_emits_browser_playable_stream() -> None:
    args = build_engine_args()

    assert args.decode_video is True
    assert args.encode_video_stream is True
    assert args.encode_codec == "libx264"
    assert args.encode_container == "frag-mp4"
    assert args.encode_feed_mode == "frame-stream"
    assert args.encode_frame_batch == 2
    assert args.encode_x264_preset == "ultrafast"
    assert args.encode_x264_crf == "19"
    assert args.encode_x264_params == ""
    assert args.encode_threads == 0
    assert args.encode_low_latency_mux is False
    assert args.encode_movflags == "frag_every_frame+empty_moov+default_base_moof+omit_tfhd_offset+separate_moof"
    assert args.encode_prestart is True
    assert args.encode_detach_after_first_byte is True
    assert args.encode_cpu_materialize_before_detach is True
    assert args.encode_audio_pipe is False
    assert args.decode_audio is True
    assert args.encode_mux_audio is True
    assert args.latent_continuation_emit_to_tail is True


def test_official_worker_allows_incremental_detached_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX_WORKER_ENCODE_DETACH_AFTER_FIRST_BYTE", "1")
    monkeypatch.setenv("LTX_WORKER_ENCODE_CPU_MATERIALIZE_BEFORE_DETACH", "0")
    monkeypatch.setenv("LTX_WORKER_ENCODE_AUDIO_PIPE", "1")

    args = build_engine_args()

    assert args.encode_detach_after_first_byte is True
    assert args.encode_cpu_materialize_before_detach is False
    assert args.encode_audio_pipe is True


def test_continuation_specs_extract_tail_frames(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for continuation frame extraction")

    previous = tmp_path / "previous.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=15:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(previous),
        ],
        check=True,
    )

    elapsed, specs = _prepare_continuation_specs(
        video_path=previous,
        job_dir=tmp_path / "job",
        frames=6,
        strength=0.75,
        crf=31,
    )

    assert elapsed >= 0.0
    assert len(specs) == 6
    for idx, spec in enumerate(specs):
        path, frame_idx, strength, crf = spec.rsplit(":", 3)
        assert int(frame_idx) == idx
        assert float(strength) == 0.75
        assert int(crf) == 31
        assert path.endswith(".png")
        assert Path(path).exists()


class _FakeBucket:
    final_fps = 15
    output_duration_s = 5.0


def test_latent_continuation_uses_visible_tail_latents(tmp_path) -> None:
    previous_mp4 = tmp_path / "previous_job" / "stream_probe_run_0_av.mp4"
    previous_mp4.parent.mkdir(parents=True)
    previous_mp4.write_bytes(b"placeholder")
    video_latent = torch.arange(1 * 4 * 11 * 2 * 3, dtype=torch.float32).reshape(1, 4, 11, 2, 3)
    audio_latent = torch.arange(1 * 2 * 135 * 4, dtype=torch.float32).reshape(1, 2, 135, 4)
    torch.save(video_latent, previous_mp4.parent / "video_latent.pt")
    torch.save(audio_latent, previous_mp4.parent / "audio_latent.pt")

    elapsed, video_conditionings, audio_conditionings, video_tail_frames, effective_frames = (
        _build_latent_continuation_conditionings(
            video_path=previous_mp4,
            bucket=_FakeBucket(),
            torch=torch,
            device="cpu",
            dtype=torch.float32,
            frames=6,
            strength=1.0,
        )
    )

    assert elapsed >= 0.0
    assert len(video_conditionings) == 1
    assert len(audio_conditionings) == 1
    assert torch.equal(video_conditionings[0].latent, video_latent[:, :, -1:, :, :])
    assert torch.equal(audio_conditionings[0].latent, audio_latent[:, :, 133:135, :])
    assert video_conditionings[0].start_idx == 0
    assert audio_conditionings[0].start_idx == 0
    assert video_tail_frames == 1
    assert effective_frames == 1


def test_latent_continuation_uses_effective_decoded_overlap_for_overlap_bucket(tmp_path) -> None:
    previous_mp4 = tmp_path / "previous_job" / "stream_probe_run_0_av.mp4"
    previous_mp4.parent.mkdir(parents=True)
    previous_mp4.write_bytes(b"placeholder")
    video_latent = torch.arange(1 * 4 * 12 * 2 * 3, dtype=torch.float32).reshape(1, 4, 12, 2, 3)
    audio_latent = torch.arange(1 * 2 * 135 * 4, dtype=torch.float32).reshape(1, 2, 135, 4)
    torch.save(video_latent, previous_mp4.parent / "video_latent.pt")
    torch.save(audio_latent, previous_mp4.parent / "audio_latent.pt")

    elapsed, video_conditionings, audio_conditionings, video_tail_frames, effective_frames = (
        _build_latent_continuation_conditionings(
            video_path=previous_mp4,
            bucket=_FakeBucket(),
            torch=torch,
            device="cpu",
            dtype=torch.float32,
            frames=14,
            strength=1.0,
        )
    )

    assert elapsed >= 0.0
    assert len(video_conditionings) == 1
    assert len(audio_conditionings) == 1
    assert torch.equal(video_conditionings[0].latent, video_latent[:, :, -2:, :, :])
    assert torch.equal(audio_conditionings[0].latent, audio_latent[:, :, 120:135, :])
    assert video_tail_frames == 2
    assert effective_frames == 9


def test_requested_latent_overlap_reports_decoded_frame_count() -> None:
    assert _video_latent_frames_for_requested_overlap(0) == 0
    assert _decoded_frames_for_video_latent_frames(0) == 0
    assert _video_latent_frames_for_requested_overlap(6) == 1
    assert _decoded_frames_for_video_latent_frames(1) == 1
    assert _video_latent_frames_for_requested_overlap(14) == 2
    assert _decoded_frames_for_video_latent_frames(2) == 9
    assert _video_latent_frames_for_requested_overlap(22) == 3
    assert _decoded_frames_for_video_latent_frames(3) == 17


class _CropBucket:
    final_frames = 4
    final_height = 4
    final_width = 6


def test_trim_crop_video_chunk_drops_hidden_overlap_before_cpu_materialize() -> None:
    chunk = torch.arange(10 * 6 * 8 * 1, dtype=torch.float32).reshape(10, 6, 8, 1)

    cropped = _trim_crop_video_chunk(chunk, _CropBucket(), start_frame=2)

    assert cropped.shape == (4, 4, 6, 1)
    torch.testing.assert_close(cropped, chunk[2:6, 1:5, 1:7, :])
