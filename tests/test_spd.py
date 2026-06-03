from types import SimpleNamespace

import pytest
import torch

from benchmarks.official_ltx23_benchmark import (
    _spd_aligned_sigma_and_kappa,
    _spd_config_from_args,
    _spd_dct_downscale_video_latent,
    _spd_dct_expand_video_latent,
    _spd_rewrite_suffix_sigmas,
)


def test_spd_dct_expand_preserves_constant_field() -> None:
    latent = torch.full((1, 2, 3, 4, 6), 3.25, dtype=torch.bfloat16)

    expanded, sigma_aligned, ratio = _spd_dct_expand_video_latent(
        torch,
        latent,
        target_height=8,
        target_width=12,
        sigma=torch.tensor(0.0),
        generator=torch.Generator(device="cpu").manual_seed(0),
        highfreq_noise=False,
        taper=0,
    )

    assert expanded.shape == (1, 2, 3, 8, 12)
    assert sigma_aligned.item() == 0.0
    assert ratio == 2.0
    torch.testing.assert_close(expanded.float(), torch.full_like(expanded.float(), 3.25), atol=5e-3, rtol=0)


def test_spd_dct_expand_can_fill_new_high_frequency_coefficients() -> None:
    latent = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(123)

    expanded, sigma_aligned, ratio = _spd_dct_expand_video_latent(
        torch,
        latent,
        target_height=8,
        target_width=8,
        sigma=torch.tensor(0.5),
        generator=generator,
        highfreq_noise=True,
        taper=0,
    )

    assert expanded.shape == (1, 1, 1, 8, 8)
    assert sigma_aligned.item() == pytest.approx(2.0 / 3.0)
    assert ratio == 2.0
    assert float(expanded.abs().max()) > 0.0


def test_spd_dct_expand_can_use_initial_high_frequency_carrier() -> None:
    latent = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)
    reference = torch.randn((1, 1, 1, 8, 8), generator=torch.Generator().manual_seed(7), dtype=torch.float32)

    expanded_a, _, _ = _spd_dct_expand_video_latent(
        torch,
        latent,
        target_height=8,
        target_width=8,
        sigma=torch.tensor(0.5),
        generator=torch.Generator(device="cpu").manual_seed(123),
        highfreq_noise=True,
        highfreq_reference_latent=reference,
        highfreq_reference_sigma=torch.tensor(1.0),
        taper=0,
    )
    expanded_b, _, _ = _spd_dct_expand_video_latent(
        torch,
        latent,
        target_height=8,
        target_width=8,
        sigma=torch.tensor(0.5),
        generator=torch.Generator(device="cpu").manual_seed(456),
        highfreq_noise=True,
        highfreq_reference_latent=reference,
        highfreq_reference_sigma=torch.tensor(1.0),
        taper=0,
    )

    torch.testing.assert_close(expanded_a, expanded_b)
    assert float(expanded_a.abs().max()) > 0.0


def test_spd_alignment_matches_paper_equations() -> None:
    sigma_aligned, kappa, ratio = _spd_aligned_sigma_and_kappa(
        torch,
        torch.tensor(0.5),
        low_height=4,
        low_width=4,
        target_height=8,
        target_width=8,
    )

    assert ratio == 2.0
    assert sigma_aligned.item() == pytest.approx(2.0 / 3.0)
    assert kappa.item() == pytest.approx(4.0 / 3.0)


def test_spd_rewrites_suffix_schedule_from_aligned_sigma() -> None:
    sigmas = torch.tensor([0.5, 0.25, 0.0])

    rewritten = _spd_rewrite_suffix_sigmas(torch, sigmas, torch.tensor(2.0 / 3.0))

    torch.testing.assert_close(rewritten, torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0]))


def test_spd_initial_dct_downscale_roundtrips_constant_at_zero_taper() -> None:
    latent = torch.full((1, 1, 2, 8, 12), -1.5, dtype=torch.float32)

    downscaled = _spd_dct_downscale_video_latent(
        torch,
        latent,
        target_height=4,
        target_width=6,
        taper=0,
    )

    assert downscaled.shape == (1, 1, 2, 4, 6)
    torch.testing.assert_close(downscaled, torch.full_like(downscaled, -3.0), atol=5e-5, rtol=0)


def test_spd_config_builds_optional_mid_stage() -> None:
    args = SimpleNamespace(
        spd_scale=0.5,
        spd_transition_step=4,
        spd_mid_scale=0.75,
        spd_mid_transition_step=None,
        steps=8,
        spd_transform="dct",
        spd_taper=8,
        spd_initial_dct_downscale=True,
        spd_highfreq_noise=True,
        spd_highfreq_source="fresh",
    )
    bucket = SimpleNamespace(width=1920, height=1080, frames=81, fps=15.0)

    config = _spd_config_from_args(args, bucket)

    assert config.transition_steps == (4, 6)
    assert [stage.scale for stage in config.stages] == [0.5, 0.75]
    assert config.stages[0].shape["width"] == 960
    assert config.stages[0].shape["height"] == 544
    assert config.stages[1].shape["width"] == 1440
    assert config.stages[1].shape["height"] == 800


def test_spd_initial_highfreq_source_requires_initial_dct_downscale() -> None:
    args = SimpleNamespace(
        spd_scale=0.5,
        spd_transition_step=4,
        spd_mid_scale=None,
        spd_mid_transition_step=None,
        steps=8,
        spd_transform="dct",
        spd_taper=8,
        spd_initial_dct_downscale=False,
        spd_highfreq_noise=True,
        spd_highfreq_source="initial",
    )
    bucket = SimpleNamespace(width=1920, height=1080, frames=81, fps=15.0)

    with pytest.raises(RuntimeError, match="initial DCT downscale"):
        _spd_config_from_args(args, bucket)
