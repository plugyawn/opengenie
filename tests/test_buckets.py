import pytest

from ltx_serve.buckets import (
    BUCKETS,
    FAST_BUCKET_NAME,
    LIVE_BASE_BUCKET_NAME,
    LIVE_BUCKET_NAME,
    LIVE_OVERLAP3_BUCKET_NAME,
    LIVE_OVERLAP_BUCKET_NAME,
    PRODUCTION_BUCKET_NAME,
    ROLLING_2S_BUCKET_NAME,
    ROLLING_4S_BUCKET_NAME,
    ROLLING_BUCKET_NAME,
    choose_bucket,
    resolve_bucket,
)


def test_bucket_token_counts() -> None:
    assert BUCKETS["realtime_5s_v1"].video_tokens == 4032
    assert BUCKETS["realtime_10s_v1"].video_tokens == 7812
    assert BUCKETS["realtime_25fps_5s_v1"].video_tokens == 4284
    assert BUCKETS["realtime_25fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["realtime_16x9_25fps_5s_v1"].video_tokens == 2448
    assert BUCKETS["realtime_16x9_25fps_5s_v1"].final_height == 288
    assert BUCKETS["realtime_16x9_25fps_5s_v1"].final_width == 512
    assert BUCKETS["realtime_16x9_25fps_5s_v1"].final_frames == 125
    assert BUCKETS["realtime_16x9_25fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["live_16x9_24fps_672x384_5s_v1"].video_tokens == 4032
    assert BUCKETS["live_16x9_24fps_672x384_5s_v1"].final_height == 384
    assert BUCKETS["live_16x9_24fps_672x384_5s_v1"].final_width == 672
    assert BUCKETS["live_16x9_24fps_672x384_5s_v1"].final_fps == 24
    assert BUCKETS["live_16x9_24fps_704x384_5s_v1"].video_tokens == 4224
    assert BUCKETS["live_16x9_24fps_736x416_5s_v1"].video_tokens == 4784
    assert BUCKETS["fast_5s_v1"].video_tokens == 9216
    assert BUCKETS["std_5s_v1"].video_tokens == 14720
    assert BUCKETS["fast_16x9_15fps_5s_v1"].video_tokens == 6336
    assert BUCKETS["fast_16x9_15fps_5s_v1"].final_height == 576
    assert BUCKETS["fast_16x9_15fps_5s_v1"].final_width == 1024
    assert BUCKETS["fast_16x9_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["qhd_16x9_15fps_5s_v1"].video_tokens == 5610
    assert BUCKETS["qhd_16x9_15fps_5s_v1"].final_height == 540
    assert BUCKETS["qhd_16x9_15fps_5s_v1"].final_width == 960
    assert BUCKETS["qhd_16x9_15fps_5s_v1"].final_frames == 75
    assert BUCKETS["qhd_16x9_15fps_5s_v1"].final_fps == 15
    assert BUCKETS["qhd_16x9_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["nearqhd_16x9_15fps_5s_v1"].video_tokens == 5423
    assert BUCKETS["nearqhd_16x9_15fps_5s_v1"].final_height == 522
    assert BUCKETS["nearqhd_16x9_15fps_5s_v1"].final_width == 928
    assert BUCKETS["nearqhd_16x9_15fps_5s_v1"].final_frames == 75
    assert BUCKETS["nearqhd_16x9_15fps_5s_v1"].final_fps == 15
    assert BUCKETS["nearqhd_16x9_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["live_16x9_15fps_5s_v1"].video_tokens == 4928
    assert BUCKETS["live_16x9_15fps_5s_v1"].final_height == 504
    assert BUCKETS["live_16x9_15fps_5s_v1"].final_width == 896
    assert BUCKETS["live_16x9_15fps_5s_v1"].final_frames == 75
    assert BUCKETS["live_16x9_15fps_5s_v1"].final_fps == 15
    assert BUCKETS["live_16x9_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["livestream_16x9_15fps_5s_v1"].video_tokens == 3696
    assert BUCKETS["livestream_16x9_15fps_5s_v1"].final_height == 432
    assert BUCKETS["livestream_16x9_15fps_5s_v1"].final_width == 768
    assert BUCKETS["livestream_16x9_15fps_5s_v1"].final_frames == 75
    assert BUCKETS["livestream_16x9_15fps_5s_v1"].final_fps == 15
    assert BUCKETS["livestream_16x9_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].video_tokens == 4032
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].latent_frames == 12
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].final_height == 432
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].final_width == 768
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].final_frames == 75
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].final_fps == 15
    assert BUCKETS["livestream_16x9_15fps_5s_overlap14_v1"].output_duration_s == 5.0
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].video_tokens == 4368
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].latent_frames == 13
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].final_height == 432
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].final_width == 768
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].final_frames == 75
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].final_fps == 15
    assert BUCKETS["livestream_16x9_15fps_5s_overlap22_v1"].output_duration_s == 5.0
    assert BUCKETS["rolling_16x9_15fps_2s_v1"].video_tokens == 1680
    assert BUCKETS["rolling_16x9_15fps_2s_v1"].final_height == 432
    assert BUCKETS["rolling_16x9_15fps_2s_v1"].final_width == 768
    assert BUCKETS["rolling_16x9_15fps_2s_v1"].final_frames == 29
    assert BUCKETS["rolling_16x9_15fps_2s_v1"].final_fps == 15
    assert BUCKETS["rolling_16x9_15fps_4s_v1"].video_tokens == 3024
    assert BUCKETS["rolling_16x9_15fps_4s_v1"].final_height == 432
    assert BUCKETS["rolling_16x9_15fps_4s_v1"].final_width == 768
    assert BUCKETS["rolling_16x9_15fps_4s_v1"].final_frames == 60
    assert BUCKETS["rolling_16x9_15fps_4s_v1"].final_fps == 15
    assert BUCKETS["rolling_16x9_15fps_4s_v1"].output_duration_s == 4.0
    assert BUCKETS["fast_16x9_25fps_5s_v1"].video_tokens == 9792
    assert BUCKETS["fast_16x9_25fps_5s_v1"].final_height == 576
    assert BUCKETS["fast_16x9_25fps_5s_v1"].final_width == 1024
    assert BUCKETS["fast_16x9_25fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["fast_25fps_5s_v1"].video_tokens == 9792
    assert BUCKETS["fast_25fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["fast_15fps_5s_v1"].video_tokens == 6336
    assert BUCKETS["fast_15fps_5s_v1"].final_frames == 75
    assert BUCKETS["fast_15fps_5s_v1"].final_fps == 15
    assert BUCKETS["fast_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["std_25fps_5s_v1"].video_tokens == 15640
    assert BUCKETS["std_25fps_5s_v1"].final_width == 720
    assert BUCKETS["std_25fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_20fps_5s_v1"].video_tokens == 12880
    assert BUCKETS["hd_16x9_20fps_5s_v1"].final_height == 720
    assert BUCKETS["hd_16x9_20fps_5s_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_20fps_5s_v1"].final_frames == 100
    assert BUCKETS["hd_16x9_20fps_5s_v1"].final_fps == 20
    assert BUCKETS["hd_16x9_20fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_15fps_5s_v1"].video_tokens == 10120
    assert BUCKETS["hd_16x9_15fps_5s_v1"].final_height == 720
    assert BUCKETS["hd_16x9_15fps_5s_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_15fps_5s_v1"].final_frames == 75
    assert BUCKETS["hd_16x9_15fps_5s_v1"].final_fps == 15
    assert BUCKETS["hd_16x9_15fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].video_tokens == 11960
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].latent_frames == 13
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].final_height == 720
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].final_frames == 75
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].final_fps == 15
    assert BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_10fps_5s_v1"].video_tokens == 7360
    assert BUCKETS["hd_16x9_10fps_5s_v1"].final_height == 720
    assert BUCKETS["hd_16x9_10fps_5s_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_10fps_5s_v1"].final_frames == 50
    assert BUCKETS["hd_16x9_10fps_5s_v1"].final_fps == 10
    assert BUCKETS["hd_16x9_10fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].video_tokens == 8280
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].latent_frames == 9
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].final_height == 720
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].final_frames == 50
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].final_fps == 10
    assert BUCKETS["hd_16x9_10fps_5s_overlap10_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_24fps_5s_v1"].video_tokens == 14720
    assert BUCKETS["hd_16x9_24fps_5s_v1"].final_height == 720
    assert BUCKETS["hd_16x9_24fps_5s_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_24fps_5s_v1"].final_frames == 120
    assert BUCKETS["hd_16x9_24fps_5s_v1"].final_fps == 24
    assert BUCKETS["hd_16x9_24fps_5s_v1"].output_duration_s == 5.0
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].video_tokens == 16560
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].latent_frames == 18
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].final_height == 720
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].final_width == 1280
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].final_frames == 120
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].final_fps == 24
    assert BUCKETS["hd_16x9_24fps_5s_overlap22_v1"].output_duration_s == 5.0
    assert BUCKETS["fast_10s_v1"].video_tokens == 17856
    assert BUCKETS["std_10s_v1"].video_tokens == 28520
    assert BUCKETS["premium_5s_v1"].final_height == 1920
    assert BUCKETS["premium_5s_v1"].final_width == 1080
    assert BUCKETS["premium_lowfps_5s_v1"].video_tokens == 10200
    assert BUCKETS["premium_ultralowfps_5s_v1"].video_tokens == 8160
    assert BUCKETS["premium_ultralowfps_5s_v1"].final_height == 1920
    assert BUCKETS["premium_ultralowfps_5s_v1"].final_width == 1080
    assert BUCKETS["premium_25fps_5s_v1"].video_tokens == 34680
    assert BUCKETS["premium_25fps_5s_v1"].final_frames == 125
    assert BUCKETS["premium_25fps_5s_v1"].final_fps == 25
    assert BUCKETS["premium_25fps_5s_v1"].output_duration_s == 5.0


def test_choose_bucket() -> None:
    assert ROLLING_BUCKET_NAME == ROLLING_2S_BUCKET_NAME
    assert LIVE_BASE_BUCKET_NAME == "livestream_16x9_15fps_5s_v1"
    assert LIVE_BUCKET_NAME == "livestream_16x9_15fps_5s_overlap14_v1"
    assert LIVE_OVERLAP_BUCKET_NAME == LIVE_BUCKET_NAME
    assert LIVE_OVERLAP3_BUCKET_NAME == "livestream_16x9_15fps_5s_overlap22_v1"
    assert choose_bucket(2, "realtime").name == ROLLING_2S_BUCKET_NAME
    assert choose_bucket(4, "realtime").name == ROLLING_4S_BUCKET_NAME
    assert choose_bucket(5, "realtime").name == LIVE_BUCKET_NAME
    assert choose_bucket(5, "fast").name == FAST_BUCKET_NAME
    assert choose_bucket(5, "standard").name == PRODUCTION_BUCKET_NAME
    assert choose_bucket(5, "premium").name == PRODUCTION_BUCKET_NAME
    with pytest.raises(ValueError, match="supports 2s/4s realtime chunks or exactly 5s reels"):
        choose_bucket(10, "premium")


def test_resolve_bucket_accepts_explicit_internal_bucket() -> None:
    assert resolve_bucket(5, "premium", "premium_16x9_15fps_5s_v1").name == "premium_16x9_15fps_5s_v1"
    assert resolve_bucket(5, "fast", None).name == FAST_BUCKET_NAME
    with pytest.raises(ValueError, match="Unsupported bucket"):
        resolve_bucket(5, "premium", "missing_bucket")
