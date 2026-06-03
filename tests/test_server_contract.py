import time

from fastapi.testclient import TestClient

from ltx_serve.buckets import BUCKETS, LIVE_BUCKET_NAME
from ltx_serve import server
from ltx_serve.schemas import GenerateRequest, LiveActionState
from ltx_serve.server import app


def test_generate_contract() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/generate",
            json={"prompt": "contract smoke", "duration_s": 5, "tier": "realtime", "audio": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["bucket"] == LIVE_BUCKET_NAME
    assert data["state"] == "queued"


def test_generate_request_accepts_server_assigned_stream_id() -> None:
    req = GenerateRequest(prompt="stream id smoke", job_id="session_0001_action")

    assert req.job_id == "session_0001_action"


def test_generate_request_accepts_explicit_internal_bucket() -> None:
    req = GenerateRequest(prompt="bucket smoke", bucket="premium_16x9_15fps_5s_v1")

    assert req.bucket == "premium_16x9_15fps_5s_v1"


def test_live_session_accepts_three_latent_overlap_request() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/live/sessions",
        json={
            "prompt": "three latent overlap smoke",
            "duration_s": 5,
            "tier": "realtime",
            "audio": True,
            "continuity_frames": 22,
            "continuity_strength": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["continuity_frames"] == 22


def test_live_session_accepts_explicit_internal_bucket() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/live/sessions",
        json={
            "prompt": "explicit 720p live bucket smoke",
            "duration_s": 5,
            "tier": "realtime",
            "bucket": "hd_16x9_15fps_5s_overlap22_v1",
            "audio": True,
            "continuity_frames": 22,
            "continuity_strength": 1.0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["bucket"] == "hd_16x9_15fps_5s_overlap22_v1"
    assert data["target_duration_s"] == BUCKETS["hd_16x9_15fps_5s_overlap22_v1"].output_duration_s


def test_live_session_create_reuses_identical_stream_config() -> None:
    client = TestClient(app)
    payload = {
        "prompt": "single local viewer duplicate load smoke",
        "duration_s": 5,
        "tier": "realtime",
        "bucket": "hd_16x9_15fps_5s_overlap22_v1",
        "audio": True,
        "seed": 20260602,
        "continuity_frames": 22,
        "continuity_strength": 1.0,
    }

    first = client.post("/api/live/sessions", json=payload)
    second = client.post("/api/live/sessions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session_id"] == first.json()["session_id"]
    assert len(second.json()["actions"]) == 1


def test_live_auto_continue_appends_hidden_segment(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_AUTO_CONTINUE", "1")
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="auto_continue_smoke",
        base_prompt="a hamster keeps moving",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
    )
    session.actions.append(
        server.LiveActionRecord(
            action_id="ready_initial",
            text="initial",
            prompt=session.base_prompt,
            segment_index=0,
            created_at=now,
            updated_at=now,
            state=LiveActionState.ready,
            user_visible=False,
        )
    )

    server._append_auto_continue_if_needed_locked(session)

    assert len(session.actions) == 2
    assert session.actions[1].text == "continue"
    assert session.actions[1].user_visible is False
    assert "No new text command is present" in session.actions[1].prompt


def test_closed_live_session_does_not_auto_continue(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_AUTO_CONTINUE", "1")
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="closed_auto_continue_smoke",
        base_prompt="a hamster keeps moving",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
        closed=True,
    )
    session.actions.append(
        server.LiveActionRecord(
            action_id="ready_initial",
            text="initial",
            prompt=session.base_prompt,
            segment_index=0,
            created_at=now,
            updated_at=now,
            state=LiveActionState.ready,
            user_visible=False,
        )
    )

    server._append_auto_continue_if_needed_locked(session)

    assert len(session.actions) == 1


def test_stop_live_session_marks_queued_actions_failed() -> None:
    client = TestClient(app)
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="bounded_rollout_stop_smoke",
        base_prompt="bounded rollout stop smoke",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
    )
    session.actions.append(
        server.LiveActionRecord(
            action_id="queued_initial",
            text="initial",
            prompt=session.base_prompt,
            segment_index=0,
            created_at=now,
            updated_at=now,
            user_visible=False,
        )
    )
    with server._live_lock:
        server._live_sessions[session.session_id] = session

    stop = client.delete(f"/api/live/sessions/{session.session_id}")

    assert stop.status_code == 200
    data = stop.json()
    assert data["closed"] is True
    assert data["actions"][0]["state"] == "failed"
    assert data["actions"][0]["error"] == "session_stopped"


def test_closed_live_session_rejects_new_action() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/live/sessions",
        json={
            "prompt": "closed action smoke",
            "duration_s": 5,
            "tier": "realtime",
            "bucket": "hd_16x9_15fps_5s_overlap22_v1",
            "audio": True,
            "continuity_frames": 22,
            "continuity_strength": 1.0,
        },
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert client.delete(f"/api/live/sessions/{session_id}").status_code == 200

    action = client.post(f"/api/live/sessions/{session_id}/actions", json={"text": "jump"})

    assert action.status_code == 409


def test_live_magcache_policy_calibrates_then_uses_hidden_continue(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_MAGCACHE", "1")
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="magcache_policy_smoke",
        base_prompt="a hamster runs on a wheel",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
    )
    initial = server.LiveActionRecord(
        action_id="initial",
        text="initial",
        prompt=session.base_prompt,
        segment_index=0,
        created_at=now,
        updated_at=now,
        user_visible=False,
    )

    policy = server._select_live_cache_policy(session, initial)
    assert policy.mode == "calibrate"
    assert policy.reason == "initial_segment"

    initial.cache_policy = policy.mode
    initial.cache_refresh = policy.refresh
    initial.residual_cache = {"mode": "calibrate", "mag_calibration_ratios": [1.1, 0.9, 1.0]}
    server._update_live_cache_after_result(session, initial)

    assert session.magcache_ratio_version == 1
    assert session.magcache_ratios == [1.1, 0.9, 1.0]

    hidden = server.LiveActionRecord(
        action_id="hidden",
        text="continue",
        prompt=server._compose_live_prompt(session, "continue", 1),
        segment_index=1,
        created_at=now,
        updated_at=now,
        user_visible=False,
    )
    policy = server._select_live_cache_policy(session, hidden)

    assert policy.mode == "magcache"
    assert policy.reason == "calibrated_hidden_continue"
    assert policy.ratio_version == 1
    assert list(policy.ratios) == [1.1, 0.9, 1.0]


def test_live_magcache_user_text_and_guard_refusal_force_refresh(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_MAGCACHE", "1")
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="magcache_refresh_smoke",
        base_prompt="a hamster runs on a wheel",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
        magcache_ratios=[1.1, 0.9, 1.0],
        magcache_ratio_version=2,
    )
    user_action = server.LiveActionRecord(
        action_id="user",
        text="hamster turns purple",
        prompt="",
        segment_index=3,
        created_at=now,
        updated_at=now,
        user_visible=True,
    )

    assert server._select_live_cache_policy(session, user_action).mode == "calibrate"

    hidden = server.LiveActionRecord(
        action_id="hidden",
        text="continue",
        prompt="",
        segment_index=4,
        created_at=now,
        updated_at=now,
        user_visible=False,
        cache_policy="magcache",
        residual_cache={"mode": "magcache", "skipped_steps": []},
    )
    server._update_live_cache_after_result(session, hidden)

    assert session.magcache_force_refresh_next is True
    assert server._select_live_cache_policy(session, hidden).mode == "calibrate"


def test_live_cache_request_kwargs_pin_conservative_magcache_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_MAGCACHE_THRESHOLD", "0.02")
    policy = server.LiveCachePolicy(
        mode="magcache",
        reason="calibrated_hidden_continue",
        refresh=False,
        ratio_version=1,
        ratios=(1.1, 0.9),
    )

    kwargs = server._live_cache_request_kwargs(policy)

    assert kwargs["residual_cache_mode"] == "magcache"
    assert kwargs["allow_quality_risk_residual_cache"] is True
    assert kwargs["residual_cache_threshold"] == 0.02
    assert kwargs["residual_cache_max_skips"] == 1
    assert kwargs["residual_cache_retention_ratio"] == 0.25
    assert kwargs["residual_cache_metric_element_stride"] == 64
    assert kwargs["residual_cache_mag_ratios"] == [1.1, 0.9]


def test_live_session_cache_overrides_are_session_scoped(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_MAGCACHE", "0")
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="cache_override_smoke",
        base_prompt="a hamster runs on a wheel",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
        live_cache_mode="magcache",
        live_cache_threshold=0.2,
        live_cache_max_skips=3,
        live_cache_retention_ratio=0.125,
        live_cache_metric_element_stride=128,
        live_cache_refresh_interval=2,
        magcache_ratios=[1.0, 0.9],
        magcache_ratio_version=1,
    )
    hidden = server.LiveActionRecord(
        action_id="hidden",
        text="continue",
        prompt="",
        segment_index=1,
        created_at=now,
        updated_at=now,
        user_visible=False,
    )

    policy = server._select_live_cache_policy(session, hidden)
    kwargs = server._live_cache_request_kwargs(policy, session)

    assert policy.mode == "magcache"
    assert server._live_magcache_enabled(session) is True
    assert kwargs["residual_cache_threshold"] == 0.2
    assert kwargs["residual_cache_max_skips"] == 3
    assert kwargs["residual_cache_retention_ratio"] == 0.125
    assert kwargs["residual_cache_metric_element_stride"] == 128
    assert kwargs["residual_cache_mag_ratios"] == [1.0, 0.9]
    assert server._live_magcache_refresh_interval(session) == 2


def test_live_teacache_mode_does_not_require_calibration(monkeypatch) -> None:
    monkeypatch.setenv("LTX_LIVE_MAGCACHE", "1")
    monkeypatch.setenv("LTX_LIVE_CACHE_MODE", "teacache")
    now = time.time()
    session = server.LiveSessionRecord(
        session_id="teacache_policy_smoke",
        base_prompt="a hamster runs on a wheel",
        tier="realtime",
        duration_s=5,
        audio=True,
        seed=1,
        initial_image_path=None,
        initial_image_strength=1.0,
        continuity_frames=22,
        continuity_strength=1.0,
        bucket="hd_16x9_15fps_5s_overlap22_v1",
        created_at=now,
        updated_at=now,
    )
    action = server.LiveActionRecord(
        action_id="initial",
        text="initial",
        prompt=session.base_prompt,
        segment_index=0,
        created_at=now,
        updated_at=now,
        user_visible=False,
    )

    policy = server._select_live_cache_policy(session, action)
    kwargs = server._live_cache_request_kwargs(policy)

    assert policy.mode == "teacache"
    assert policy.reason == "teacache_chunk"
    assert kwargs["residual_cache_mode"] == "teacache"
    assert "residual_cache_mag_ratios" not in kwargs


def test_live_session_accepts_extended_latent_context_request() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/live/sessions",
        json={
            "prompt": "extended context smoke",
            "duration_s": 5,
            "tier": "fast",
            "audio": True,
            "continuity_frames": 40,
            "continuity_strength": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["continuity_frames"] == 40


def test_predict_remote_stream_url_uses_stable_job_id(monkeypatch) -> None:
    monkeypatch.setattr(server, "REMOTE_BACKEND_URL", "http://127.0.0.1:9000")

    assert server._predict_remote_stream_url("session_0001_action") == "http://127.0.0.1:9000/streams/session_0001_action"


def test_predict_remote_stream_url_uses_local_proxy_when_dns_override_is_pinned(monkeypatch) -> None:
    monkeypatch.setattr(server, "REMOTE_BACKEND_URL", "https://modal-worker.example")
    monkeypatch.setenv("LTX_REMOTE_BACKEND_RESOLVE_IPS", "127.0.0.1")

    assert server._predict_remote_stream_url("session_0001_action") == "/api/remote/streams/session_0001_action"


def test_public_output_url_rewrites_worker_localhost_stream(monkeypatch) -> None:
    monkeypatch.setattr(server, "REMOTE_BACKEND_URL", "https://modal-worker.example")

    assert (
        server._public_output_url("session_0001_action", "http://127.0.0.1:9000/streams/session_0001_action")
        == "https://modal-worker.example/streams/session_0001_action"
    )


def test_remote_dns_override_is_scoped_to_backend_host(monkeypatch) -> None:
    import socket

    monkeypatch.setattr(server, "REMOTE_BACKEND_URL", "https://example.invalid")
    monkeypatch.setenv("LTX_REMOTE_BACKEND_RESOLVE_IPS", "127.0.0.1")

    original = socket.getaddrinfo
    with server._remote_dns_override():
        assert socket.getaddrinfo is not original
        assert socket.getaddrinfo("example.invalid", 443)[0][4][0] == "127.0.0.1"
    assert socket.getaddrinfo is original


def test_job_status_exposes_start_streaming_metrics() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={"prompt": "metric smoke", "duration_s": 5, "tier": "premium", "audio": True},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    for _ in range(20):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] == "complete":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("mock job did not complete")

    assert status["time_to_first_video_byte_s"] is not None
    assert status["first_byte_realtime_factor"] is not None
    assert status["faster_than_realtime"] is not None
    assert status["media_type"] == "video/mp4"
    assert "mock_generate_s" in status["stage_times"]


def test_live_session_queues_text_actions() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/live/sessions",
        json={"prompt": "third person game in a forest", "duration_s": 5, "tier": "realtime", "audio": True},
    )
    assert response.status_code == 200
    session = response.json()
    session_id = session["session_id"]
    assert session["bucket"] == LIVE_BUCKET_NAME
    assert session["target_duration_s"] == BUCKETS[LIVE_BUCKET_NAME].output_duration_s
    assert session["actions"][0]["state"] in {"queued", "running", "ready"}

    response = client.post(
        f"/api/live/sessions/{session_id}/actions",
        json={"text": "press W and walk forward"},
    )
    assert response.status_code == 200
    session = response.json()
    assert len(session["actions"]) == 2
    assert session["actions"][1]["text"] == "press W and walk forward"

    for _ in range(80):
        session = client.get(f"/api/live/sessions/{session_id}").json()
        if len([item for item in session["actions"] if item["state"] == "ready"]) == 2:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("mock live session did not drain")

    assert session["actions"][0]["output_url"]
    assert session["actions"][1]["output_url"]
    assert session["actions"][0]["continuation_frames"] == 0
    assert session["actions"][0]["continuation_video_path"] is None
    assert session["actions"][1]["continuation_frames"] == 14
    assert session["actions"][1]["continuation_video_path"]
    assert session["actions"][1]["first_byte_realtime_factor"] is not None
