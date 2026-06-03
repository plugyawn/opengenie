from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LatentTailConditioning:
    """Replace a contiguous latent time prefix with clean continuation latents.

    This intentionally mirrors LTX's latent-index image conditioner, but it
    consumes already-denoised video/audio latents instead of round-tripping
    through pixels and the VAE encoder.
    """

    latent: Any
    start_idx: int = 0
    strength: float = 1.0

    def apply_to(self, latent_state: Any, latent_tools: Any) -> Any:
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")

        target_shape = latent_tools.target_shape
        target_torch_shape = target_shape.to_torch_shape()
        cond_shape = self.latent.shape
        if len(cond_shape) != len(target_torch_shape):
            raise ValueError(f"latent rank {len(cond_shape)} does not match target rank {len(target_torch_shape)}")
        if cond_shape[0] != target_torch_shape[0] or cond_shape[1] != target_torch_shape[1]:
            raise ValueError(f"latent batch/channels {cond_shape[:2]} do not match target {target_torch_shape[:2]}")
        if cond_shape[3:] != target_torch_shape[3:]:
            raise ValueError(f"latent non-temporal shape {cond_shape[3:]} does not match target {target_torch_shape[3:]}")

        frames = int(cond_shape[2])
        if frames <= 0:
            raise ValueError("conditioning latent must contain at least one frame")
        if self.start_idx + frames > int(target_torch_shape[2]):
            raise ValueError(
                f"conditioning frames [{self.start_idx}, {self.start_idx + frames}) exceed target frames "
                f"{int(target_torch_shape[2])}"
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        start_shape = target_shape._replace(frames=self.start_idx)
        start_token = latent_tools.patchifier.get_token_count(start_shape)
        stop_token = start_token + tokens.shape[1]

        next_state = latent_state.clone()
        next_state.latent[:, start_token:stop_token] = tokens
        next_state.clean_latent[:, start_token:stop_token] = tokens
        next_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength
        return next_state
