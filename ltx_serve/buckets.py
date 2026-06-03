from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeBucket:
    name: str
    height: int
    width: int
    frames: int
    fps: int
    tier: str
    stage_mode: str
    output_height: int | None = None
    output_width: int | None = None
    output_frames: int | None = None
    output_fps: int | None = None

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps

    @property
    def output_duration_s(self) -> float:
        return self.final_frames / self.final_fps

    @property
    def latent_frames(self) -> int:
        return 1 + (self.frames - 1) // 8

    @property
    def latent_height(self) -> int:
        return self.height // 32

    @property
    def latent_width(self) -> int:
        return self.width // 32

    @property
    def video_tokens(self) -> int:
        return self.latent_frames * self.latent_height * self.latent_width

    @property
    def final_height(self) -> int:
        return self.output_height or self.height

    @property
    def final_width(self) -> int:
        return self.output_width or self.width

    @property
    def final_frames(self) -> int:
        return self.output_frames or self.frames

    @property
    def final_fps(self) -> int:
        return self.output_fps or self.fps


BUCKETS: dict[str, ShapeBucket] = {
    "realtime_5s_v1": ShapeBucket("realtime_5s_v1", 672, 384, 121, 24, "realtime", "single"),
    "realtime_10s_v1": ShapeBucket("realtime_10s_v1", 672, 384, 241, 24, "realtime", "single"),
    "realtime_25fps_5s_v1": ShapeBucket(
        "realtime_25fps_5s_v1", 672, 384, 129, 25, "realtime", "single", 672, 384, 125, 25
    ),
    "realtime_16x9_25fps_5s_v1": ShapeBucket(
        "realtime_16x9_25fps_5s_v1", 288, 512, 129, 25, "realtime", "single", 288, 512, 125, 25
    ),
    "live_16x9_24fps_672x384_5s_v1": ShapeBucket(
        "live_16x9_24fps_672x384_5s_v1", 384, 672, 121, 24, "realtime", "single"
    ),
    "live_16x9_24fps_704x384_5s_v1": ShapeBucket(
        "live_16x9_24fps_704x384_5s_v1", 384, 704, 121, 24, "realtime", "single"
    ),
    "live_16x9_24fps_736x416_5s_v1": ShapeBucket(
        "live_16x9_24fps_736x416_5s_v1", 416, 736, 121, 24, "realtime", "single"
    ),
    "fast_5s_v1": ShapeBucket("fast_5s_v1", 1024, 576, 121, 24, "fast", "single"),
    "std_5s_v1": ShapeBucket("std_5s_v1", 1280, 736, 121, 24, "standard", "single"),
    "fast_16x9_15fps_5s_v1": ShapeBucket(
        "fast_16x9_15fps_5s_v1", 576, 1024, 81, 15, "fast", "single", 576, 1024, 75, 15
    ),
    "qhd_16x9_15fps_5s_v1": ShapeBucket(
        "qhd_16x9_15fps_5s_v1", 544, 960, 81, 15, "fast", "single", 540, 960, 75, 15
    ),
    "nearqhd_16x9_15fps_5s_v1": ShapeBucket(
        "nearqhd_16x9_15fps_5s_v1", 544, 928, 81, 15, "fast", "single", 522, 928, 75, 15
    ),
    "premium_16x9_15fps_5s_v1": ShapeBucket(
        "premium_16x9_15fps_5s_v1", 1088, 1920, 81, 15, "premium", "single", 1080, 1920, 75, 15
    ),
    "live_16x9_15fps_5s_v1": ShapeBucket(
        "live_16x9_15fps_5s_v1", 512, 896, 81, 15, "realtime", "single", 504, 896, 75, 15
    ),
    "livestream_16x9_15fps_5s_v1": ShapeBucket(
        "livestream_16x9_15fps_5s_v1", 448, 768, 81, 15, "realtime", "single", 432, 768, 75, 15
    ),
    "livestream_16x9_15fps_5s_overlap14_v1": ShapeBucket(
        "livestream_16x9_15fps_5s_overlap14_v1", 448, 768, 89, 15, "realtime", "single", 432, 768, 75, 15
    ),
    "livestream_16x9_15fps_5s_overlap22_v1": ShapeBucket(
        "livestream_16x9_15fps_5s_overlap22_v1", 448, 768, 97, 15, "realtime", "single", 432, 768, 75, 15
    ),
    "rolling_16x9_15fps_2s_v1": ShapeBucket(
        "rolling_16x9_15fps_2s_v1", 448, 768, 33, 15, "realtime", "single", 432, 768, 29, 15
    ),
    "rolling_16x9_15fps_4s_v1": ShapeBucket(
        "rolling_16x9_15fps_4s_v1", 448, 768, 65, 15, "realtime", "single", 432, 768, 60, 15
    ),
    "fast_16x9_25fps_5s_v1": ShapeBucket(
        "fast_16x9_25fps_5s_v1", 576, 1024, 129, 25, "fast", "single", 576, 1024, 125, 25
    ),
    "fast_25fps_5s_v1": ShapeBucket(
        "fast_25fps_5s_v1", 1024, 576, 129, 25, "fast", "single", 1024, 576, 125, 25
    ),
    "fast_15fps_5s_v1": ShapeBucket(
        "fast_15fps_5s_v1", 1024, 576, 81, 15, "fast", "single", 1024, 576, 75, 15
    ),
    "std_25fps_5s_v1": ShapeBucket(
        "std_25fps_5s_v1", 1280, 736, 129, 25, "standard", "single", 1280, 720, 125, 25
    ),
    "std_16x9_25fps_5s_overlap32_v1": ShapeBucket(
        "std_16x9_25fps_5s_overlap32_v1", 736, 1280, 153, 25, "standard", "single", 720, 1280, 125, 25
    ),
    "hd_16x9_20fps_5s_v1": ShapeBucket(
        "hd_16x9_20fps_5s_v1", 736, 1280, 105, 20, "standard", "single", 720, 1280, 100, 20
    ),
    "hd_16x9_24fps_5s_v1": ShapeBucket(
        "hd_16x9_24fps_5s_v1", 736, 1280, 121, 24, "standard", "single", 720, 1280, 120, 24
    ),
    "hd_16x9_24fps_5s_overlap22_v1": ShapeBucket(
        "hd_16x9_24fps_5s_overlap22_v1", 736, 1280, 137, 24, "standard", "single", 720, 1280, 120, 24
    ),
    "hd_16x9_15fps_5s_v1": ShapeBucket(
        "hd_16x9_15fps_5s_v1", 736, 1280, 81, 15, "standard", "single", 720, 1280, 75, 15
    ),
    "hd_16x9_15fps_5s_overlap22_v1": ShapeBucket(
        "hd_16x9_15fps_5s_overlap22_v1", 736, 1280, 97, 15, "standard", "single", 720, 1280, 75, 15
    ),
    "hd_16x9_10fps_5s_v1": ShapeBucket(
        "hd_16x9_10fps_5s_v1", 736, 1280, 57, 10, "standard", "single", 720, 1280, 50, 10
    ),
    "hd_16x9_10fps_5s_overlap10_v1": ShapeBucket(
        "hd_16x9_10fps_5s_overlap10_v1", 736, 1280, 65, 10, "standard", "single", 720, 1280, 50, 10
    ),
    "fast_10s_v1": ShapeBucket("fast_10s_v1", 1024, 576, 241, 24, "fast", "single"),
    "std_10s_v1": ShapeBucket("std_10s_v1", 1280, 736, 241, 24, "standard", "single"),
    "premium_5s_v1": ShapeBucket("premium_5s_v1", 1920, 1088, 121, 24, "premium", "single", 1920, 1080),
    "premium_10s_v1": ShapeBucket("premium_10s_v1", 1920, 1088, 241, 24, "premium", "single", 1920, 1080),
    "premium_lowfps_5s_v1": ShapeBucket("premium_lowfps_5s_v1", 1920, 1088, 33, 6, "premium", "single", 1920, 1080),
    "premium_ultralowfps_5s_v1": ShapeBucket("premium_ultralowfps_5s_v1", 1920, 1088, 25, 5, "premium", "single", 1920, 1080),
    "premium_25fps_5s_v1": ShapeBucket("premium_25fps_5s_v1", 1920, 1088, 129, 25, "premium", "single", 1920, 1080, 125, 25),
}

LIVE_BASE_BUCKET_NAME = "livestream_16x9_15fps_5s_v1"
LIVE_BUCKET_NAME = "livestream_16x9_15fps_5s_overlap14_v1"
LIVE_OVERLAP_BUCKET_NAME = LIVE_BUCKET_NAME
LIVE_OVERLAP3_BUCKET_NAME = "livestream_16x9_15fps_5s_overlap22_v1"
ROLLING_2S_BUCKET_NAME = "rolling_16x9_15fps_2s_v1"
ROLLING_4S_BUCKET_NAME = "rolling_16x9_15fps_4s_v1"
ROLLING_BUCKET_NAME = ROLLING_2S_BUCKET_NAME
FAST_BUCKET_NAME = "nearqhd_16x9_15fps_5s_v1"
PRODUCTION_BUCKET_NAME = "premium_25fps_5s_v1"
LIVE_BUCKET = BUCKETS[LIVE_BUCKET_NAME]
LIVE_BASE_BUCKET = BUCKETS[LIVE_BASE_BUCKET_NAME]
LIVE_OVERLAP_BUCKET = BUCKETS[LIVE_OVERLAP_BUCKET_NAME]
LIVE_OVERLAP3_BUCKET = BUCKETS[LIVE_OVERLAP3_BUCKET_NAME]
ROLLING_2S_BUCKET = BUCKETS[ROLLING_2S_BUCKET_NAME]
ROLLING_4S_BUCKET = BUCKETS[ROLLING_4S_BUCKET_NAME]
ROLLING_BUCKET = ROLLING_2S_BUCKET
FAST_BUCKET = BUCKETS[FAST_BUCKET_NAME]
PRODUCTION_BUCKET = BUCKETS[PRODUCTION_BUCKET_NAME]


def choose_bucket(duration_s: int, tier: str) -> ShapeBucket:
    """Return the pinned serving bucket for a public quality tier."""
    tier_name = tier.lower()
    if tier_name not in {"realtime", "fast", "standard", "premium"}:
        raise ValueError(f"Unsupported tier: {tier!r}")
    if duration_s == 2 and tier_name == "realtime":
        return ROLLING_2S_BUCKET
    if duration_s == 4 and tier_name == "realtime":
        return ROLLING_4S_BUCKET
    if duration_s != 5:
        raise ValueError("Production LTX-2.3 serving supports 2s/4s realtime chunks or exactly 5s reels.")
    if tier_name == "realtime":
        return LIVE_BUCKET
    if tier_name == "fast":
        return FAST_BUCKET
    return PRODUCTION_BUCKET


def resolve_bucket(duration_s: int, tier: str, bucket_name: str | None = None) -> ShapeBucket:
    """Resolve a public tier or an explicit internal bucket for controlled probes."""
    if bucket_name is None:
        return choose_bucket(duration_s, tier)
    bucket = BUCKETS.get(bucket_name)
    if bucket is None:
        raise ValueError(f"Unsupported bucket: {bucket_name!r}")
    return bucket
