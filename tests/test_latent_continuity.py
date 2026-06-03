from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ltx_serve.latent_continuity import LatentTailConditioning


class _VideoShape(NamedTuple):
    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])


class _Patchifier:
    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], -1, latent.shape[1])

    def get_token_count(self, shape: _VideoShape) -> int:
        return shape.frames * shape.height * shape.width


@dataclass
class _Tools:
    target_shape: _VideoShape
    patchifier: _Patchifier


@dataclass
class _State:
    latent: torch.Tensor
    clean_latent: torch.Tensor
    denoise_mask: torch.Tensor

    def clone(self) -> "_State":
        return _State(self.latent.clone(), self.clean_latent.clone(), self.denoise_mask.clone())


def test_latent_tail_conditioning_replaces_prefix_tokens_and_freezes_mask() -> None:
    target_shape = _VideoShape(batch=1, channels=2, frames=4, height=2, width=2)
    state = _State(
        latent=torch.zeros(1, 16, 2),
        clean_latent=torch.zeros(1, 16, 2),
        denoise_mask=torch.ones(1, 16, 1),
    )
    tail = torch.arange(1 * 2 * 1 * 2 * 2, dtype=torch.float32).reshape(1, 2, 1, 2, 2)

    conditioned = LatentTailConditioning(tail, start_idx=0, strength=1.0).apply_to(
        state,
        _Tools(target_shape=target_shape, patchifier=_Patchifier()),
    )

    expected_tokens = _Patchifier().patchify(tail)
    assert torch.equal(conditioned.latent[:, :4], expected_tokens)
    assert torch.equal(conditioned.clean_latent[:, :4], expected_tokens)
    assert torch.count_nonzero(conditioned.denoise_mask[:, :4]).item() == 0
    assert torch.all(conditioned.denoise_mask[:, 4:] == 1)
    assert torch.all(state.latent == 0)
