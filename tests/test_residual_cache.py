from __future__ import annotations

import pytest
import torch

from benchmarks.official_ltx23_benchmark import _ResidualCacheConfig, _ResidualDenoiserCache


def _config(
    *,
    mode: str = "force-skip",
    force_skip_steps: frozenset[int] = frozenset({1}),
    total_steps: int = 4,
    max_skips: int = 1,
    retention_ratio: float = 0.0,
) -> _ResidualCacheConfig:
    return _ResidualCacheConfig(
        mode=mode,
        threshold=0.01,
        max_skips=max_skips,
        retention_ratio=retention_ratio,
        force_skip_steps=force_skip_steps,
        mag_ratios=(),
        metric_element_stride=1,
        total_steps=total_steps,
    )


def test_force_skip_reuses_cached_velocity_at_current_sigma() -> None:
    stats: dict[str, object] = {}
    cache = _ResidualDenoiserCache(_config(), stats)
    latent0 = torch.full((1, 4, 3), 5.0, dtype=torch.bfloat16)
    velocity = torch.full_like(latent0, 2.0)
    sigma0 = torch.tensor([1.0])

    decision0 = cache.begin_step(latent0, None, sigma0)
    assert decision0.skip is False
    cache.finish_compute(
        video_latent=latent0,
        audio_latent=None,
        sigma=sigma0,
        denoised_video=latent0.float().sub(velocity.float() * sigma0.reshape(1, 1, 1)).to(torch.bfloat16),
        denoised_audio=None,
        record=decision0.record,
    )

    latent1 = torch.full((1, 4, 3), 8.0, dtype=torch.bfloat16)
    sigma1 = torch.tensor([0.5])
    decision1 = cache.begin_step(latent1, None, sigma1)

    assert decision1.skip is True
    expected = latent1.float().sub(velocity.float() * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(decision1.video, expected)
    assert stats["computed_steps"] == [0]
    assert stats["skipped_steps"] == [1]


def test_shape_change_invalidates_cache_before_forced_skip() -> None:
    stats: dict[str, object] = {}
    cache = _ResidualDenoiserCache(_config(), stats)
    latent0 = torch.ones((1, 4, 3), dtype=torch.bfloat16)

    decision0 = cache.begin_step(latent0, None, torch.tensor([1.0]))
    cache.finish_compute(
        video_latent=latent0,
        audio_latent=None,
        sigma=torch.tensor([1.0]),
        denoised_video=torch.zeros_like(latent0),
        denoised_audio=None,
        record=decision0.record,
    )

    latent1 = torch.ones((1, 5, 3), dtype=torch.bfloat16)
    decision1 = cache.begin_step(latent1, None, torch.tensor([0.5]))

    assert decision1.skip is False
    assert decision1.record["reason"] == "shape_changed"
    assert stats["invalidations"] == [{"step": 1, "reason": "shape_changed"}]


def test_first_and_last_steps_are_not_skipped() -> None:
    stats: dict[str, object] = {}
    cache = _ResidualDenoiserCache(
        _config(force_skip_steps=frozenset({0, 1, 2}), total_steps=3, max_skips=3),
        stats,
    )
    latent = torch.ones((1, 2, 2), dtype=torch.bfloat16)

    decision0 = cache.begin_step(latent, None, torch.tensor([1.0]))
    assert decision0.skip is False
    cache.finish_compute(
        video_latent=latent,
        audio_latent=None,
        sigma=torch.tensor([1.0]),
        denoised_video=torch.zeros_like(latent),
        denoised_audio=None,
        record=decision0.record,
    )

    decision1 = cache.begin_step(latent, None, torch.tensor([0.5]))
    assert decision1.skip is True

    decision2 = cache.begin_step(latent, None, torch.tensor([0.25]))
    assert decision2.skip is False
    assert decision2.record["reason"] == "not_forced"


def test_max_skips_caps_total_nonconsecutive_skips() -> None:
    stats: dict[str, object] = {}
    cache = _ResidualDenoiserCache(
        _config(force_skip_steps=frozenset({1, 3, 4}), total_steps=6, max_skips=2),
        stats,
    )
    latent = torch.ones((1, 2, 2), dtype=torch.bfloat16)

    decision0 = cache.begin_step(latent, None, torch.tensor([1.0]))
    cache.finish_compute(
        video_latent=latent,
        audio_latent=None,
        sigma=torch.tensor([1.0]),
        denoised_video=torch.zeros_like(latent),
        denoised_audio=None,
        record=decision0.record,
    )

    decision1 = cache.begin_step(latent, None, torch.tensor([0.8]))
    assert decision1.skip is True

    decision2 = cache.begin_step(latent, None, torch.tensor([0.6]))
    assert decision2.skip is False
    cache.finish_compute(
        video_latent=latent,
        audio_latent=None,
        sigma=torch.tensor([0.6]),
        denoised_video=torch.zeros_like(latent),
        denoised_audio=None,
        record=decision2.record,
    )

    decision3 = cache.begin_step(latent, None, torch.tensor([0.4]))
    assert decision3.skip is True

    decision4 = cache.begin_step(latent, None, torch.tensor([0.2]))
    assert decision4.skip is False
    assert decision4.record["total_skips_before"] == 2
    assert stats["skipped_steps"] == [1, 3]


def test_dual_stream_skip_requires_both_cached_streams() -> None:
    stats: dict[str, object] = {}
    cache = _ResidualDenoiserCache(_config(force_skip_steps=frozenset({1})), stats)
    video = torch.ones((1, 2, 2), dtype=torch.bfloat16)
    audio = torch.ones((1, 3, 2), dtype=torch.bfloat16)

    decision0 = cache.begin_step(video, None, torch.tensor([1.0]))
    cache.finish_compute(
        video_latent=video,
        audio_latent=None,
        sigma=torch.tensor([1.0]),
        denoised_video=torch.zeros_like(video),
        denoised_audio=None,
        record=decision0.record,
    )

    decision1 = cache.begin_step(video, audio, torch.tensor([0.5]))
    assert decision1.skip is False


def test_magcache_interpolates_existing_ratio_schedule_without_tensor_metric() -> None:
    stats: dict[str, object] = {}
    config = _ResidualCacheConfig(
        mode="magcache",
        threshold=0.02,
        max_skips=1,
        retention_ratio=0.0,
        force_skip_steps=frozenset(),
        mag_ratios=(1.0, 1.0, 1.0),
        metric_element_stride=1,
        total_steps=3,
    )
    cache = _ResidualDenoiserCache(config, stats)
    latent = torch.ones((1, 2, 2), dtype=torch.bfloat16)

    decision0 = cache.begin_step(latent, None, torch.tensor([1.0]))
    cache.finish_compute(
        video_latent=latent,
        audio_latent=None,
        sigma=torch.tensor([1.0]),
        denoised_video=torch.zeros_like(latent),
        denoised_audio=None,
        record=decision0.record,
    )
    decision1 = cache.begin_step(latent, None, torch.tensor([0.5]))

    assert decision1.skip is True
    assert decision1.record["teacache_metric"] is None
    assert decision1.record["mag_scale"] == pytest.approx(1.0)
