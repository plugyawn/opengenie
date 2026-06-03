from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"


class LiveActionState(str, Enum):
    queued = "queued"
    running = "running"
    ready = "ready"
    failed = "failed"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    job_id: str | None = Field(default=None, min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")
    duration_s: int = Field(default=5, ge=2, le=5)
    tier: Literal["realtime", "fast", "standard", "premium"] = "realtime"
    bucket: str | None = Field(default=None, min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")
    seed: int | None = None
    audio: bool = True
    image_conditioning_path: str | None = None
    image_conditioning_frame_idx: int = Field(default=0, ge=0)
    image_conditioning_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    image_conditioning_crf: int = Field(default=33, ge=0, le=63)
    continuation_mode: Literal["latent", "pixel"] = "latent"
    continuation_video_path: str | None = None
    continuation_frames: int = Field(default=0, ge=0, le=64)
    continuation_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    continuation_crf: int = Field(default=33, ge=0, le=63)
    residual_cache_mode: Literal["off", "calibrate", "magcache", "teacache", "force-skip"] | None = None
    allow_quality_risk_residual_cache: bool = False
    residual_cache_threshold: float | None = Field(default=None, ge=0.0)
    residual_cache_max_skips: int | None = Field(default=None, ge=0, le=8)
    residual_cache_retention_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    residual_cache_force_skip_steps: list[int] | None = None
    residual_cache_mag_ratios: list[float] | None = None
    residual_cache_metric_element_stride: int | None = Field(default=None, ge=1)
    spd: bool | None = None
    allow_quality_risk_spd: bool = False
    spd_scale: float | None = Field(default=None, gt=0.0, lt=1.0)
    spd_transition_step: int | None = Field(default=None, ge=1, le=7)
    spd_mid_scale: float | None = Field(default=None, gt=0.0, lt=1.0)
    spd_mid_transition_step: int | None = Field(default=None, ge=1, le=7)
    spd_taper: int | None = Field(default=None, ge=0)
    spd_initial_dct_downscale: bool | None = None
    spd_highfreq_noise: bool | None = None
    spd_highfreq_source: Literal["fresh", "initial"] | None = None


class GenerateResponse(BaseModel):
    job_id: str
    state: JobState
    bucket: str
    status_url: str


class LiveSessionCreateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    duration_s: int = Field(default=5, ge=2, le=5)
    tier: Literal["realtime", "fast", "standard", "premium"] = "realtime"
    bucket: str | None = Field(default=None, min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")
    seed: int | None = None
    audio: bool = True
    initial_image_path: str | None = None
    initial_image_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    continuity_frames: int = Field(default=14, ge=0, le=64)
    continuity_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    live_cache_mode: Literal["off", "magcache", "teacache"] | None = None
    live_cache_threshold: float | None = Field(default=None, ge=0.0)
    live_cache_max_skips: int | None = Field(default=None, ge=0, le=8)
    live_cache_retention_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    live_cache_metric_element_stride: int | None = Field(default=None, ge=1)
    live_cache_refresh_interval: int | None = Field(default=None, ge=0)


class LiveActionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class LiveActionStatus(BaseModel):
    action_id: str
    state: LiveActionState
    text: str
    prompt: str
    segment_index: int
    user_visible: bool = True
    created_at: float
    updated_at: float
    job_id: str | None = None
    bucket: str | None = None
    output_url: str | None = None
    media_type: str | None = None
    runtime_s: float | None = None
    time_to_first_video_byte_s: float | None = None
    realtime_factor: float | None = None
    first_byte_realtime_factor: float | None = None
    faster_than_realtime: bool | None = None
    encode_detached: bool | None = None
    encode_complete: bool | None = None
    continuation_video_path: str | None = None
    continuation_frames: int = 0
    cache_policy: str = "off"
    cache_reason: str | None = None
    cache_refresh: bool = False
    cache_ratio_version: int = 0
    residual_cache: dict[str, object] | None = None
    stage_times: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class LiveSessionStatus(BaseModel):
    session_id: str
    base_prompt: str
    tier: Literal["realtime", "fast", "standard", "premium"]
    duration_s: int
    target_duration_s: float
    audio: bool
    seed: int | None
    initial_image_path: str | None = None
    initial_image_strength: float = 1.0
    continuity_frames: int
    continuity_strength: float
    bucket: str
    created_at: float
    updated_at: float
    closed: bool = False
    processing: bool
    current_action_id: str | None = None
    current_output_url: str | None = None
    cache_enabled: bool = False
    cache_policy: str = "off"
    cache_ratio_version: int = 0
    cache_ratio_count: int = 0
    cache_chunks_since_refresh: int = 0
    cache_refresh_interval: int = 0
    actions: list[LiveActionStatus] = Field(default_factory=list)


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    prompt: str
    bucket: str
    created_at: float
    updated_at: float
    duration_s: float
    target_runtime_s: float
    runtime_s: float | None = None
    time_to_first_video_byte_s: float | None = None
    realtime_factor: float | None = None
    first_byte_realtime_factor: float | None = None
    faster_than_realtime: bool | None = None
    output_url: str | None = None
    media_type: str | None = None
    encode_detached: bool | None = None
    encode_complete: bool | None = None
    residual_cache: dict[str, object] | None = None
    stage_times: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class BenchmarkResult(BaseModel):
    backend: str
    bucket: str
    prompt: str
    output_path: str | None
    runtime_s: float
    e2e_runtime_s: float
    target_duration_s: float
    realtime_factor: float
    faster_than_realtime: bool
    gpu_name: str | None = None
    peak_vram_gb: float | None = None
    stage_times: dict[str, float] = Field(default_factory=dict)
