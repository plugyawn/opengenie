from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from ltx_serve.buckets import resolve_bucket
from ltx_serve.mock_video import make_mock_video
from ltx_serve.schemas import (
    GenerateRequest,
    GenerateResponse,
    JobState,
    JobStatus,
    LiveActionRequest,
    LiveActionState,
    LiveActionStatus,
    LiveSessionCreateRequest,
    LiveSessionStatus,
)


APP_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = APP_ROOT / "web"
OUTPUT_DIR = Path(os.getenv("LTX_SERVE_OUTPUT_DIR", APP_ROOT / "outputs"))
REMOTE_BACKEND_URL = os.getenv("LTX_REMOTE_BACKEND_URL", "").rstrip("/")

app = FastAPI(title="LTX-2.3 Reel Runtime")
_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()
_live_sessions: dict[str, "LiveSessionRecord"] = {}
_live_lock = threading.Lock()
_remote_generate_lock = threading.Lock()


@dataclass
class LiveActionRecord:
    action_id: str
    text: str
    prompt: str
    segment_index: int
    created_at: float
    updated_at: float
    state: LiveActionState = LiveActionState.queued
    user_visible: bool = True
    job_id: str | None = None
    bucket: str | None = None
    output_url: str | None = None
    output_path: str | None = None
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
    stage_times: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class LiveCachePolicy:
    mode: str
    reason: str
    refresh: bool
    ratio_version: int
    ratios: tuple[float, ...] = ()


@dataclass
class LiveSessionRecord:
    session_id: str
    base_prompt: str
    tier: str
    duration_s: int
    audio: bool
    seed: int | None
    initial_image_path: str | None
    initial_image_strength: float
    continuity_frames: int
    continuity_strength: float
    bucket: str
    created_at: float
    updated_at: float
    live_cache_mode: str | None = None
    live_cache_threshold: float | None = None
    live_cache_max_skips: int | None = None
    live_cache_retention_ratio: float | None = None
    live_cache_metric_element_stride: int | None = None
    live_cache_refresh_interval: int | None = None
    closed: bool = False
    actions: list[LiveActionRecord] = field(default_factory=list)
    processing: bool = False
    current_action_id: str | None = None
    current_output_url: str | None = None
    magcache_ratios: list[float] = field(default_factory=list)
    magcache_ratio_version: int = 0
    magcache_last_refresh_segment: int | None = None
    magcache_chunks_since_refresh: int = 0
    magcache_force_refresh_next: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        bucket = resolve_bucket(req.duration_s, req.tier, req.bucket)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    status = JobStatus(
        job_id=job_id,
        state=JobState.queued,
        prompt=req.prompt,
        bucket=bucket.name,
        created_at=now,
        updated_at=now,
        duration_s=bucket.output_duration_s,
        target_runtime_s=bucket.output_duration_s,
    )
    with _lock:
        _jobs[job_id] = status
    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()
    return GenerateResponse(job_id=job_id, state=JobState.queued, bucket=bucket.name, status_url=f"/api/jobs/{job_id}")


@app.post("/api/live/sessions", response_model=LiveSessionStatus)
def create_live_session(req: LiveSessionCreateRequest) -> LiveSessionStatus:
    try:
        bucket = resolve_bucket(req.duration_s, req.tier, req.bucket)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = time.time()
    session_id = uuid.uuid4().hex[:12]
    session = LiveSessionRecord(
        session_id=session_id,
        base_prompt=req.prompt,
        tier=req.tier,
        duration_s=req.duration_s,
        audio=req.audio,
        seed=req.seed,
        initial_image_path=req.initial_image_path,
        initial_image_strength=req.initial_image_strength,
        continuity_frames=req.continuity_frames,
        continuity_strength=req.continuity_strength,
        bucket=bucket.name,
        live_cache_mode=req.live_cache_mode,
        live_cache_threshold=req.live_cache_threshold,
        live_cache_max_skips=req.live_cache_max_skips,
        live_cache_retention_ratio=req.live_cache_retention_ratio,
        live_cache_metric_element_stride=req.live_cache_metric_element_stride,
        live_cache_refresh_interval=req.live_cache_refresh_interval,
        created_at=now,
        updated_at=now,
    )
    session.actions.append(
        LiveActionRecord(
            action_id=uuid.uuid4().hex[:12],
            text="initial",
            prompt=_compose_live_prompt(session, "initial", 0),
            segment_index=0,
            created_at=now,
            updated_at=now,
            user_visible=False,
        )
    )
    reusable_session_id = None
    with _live_lock:
        reusable = _find_reusable_live_session_locked(req, bucket.name)
        if reusable is None:
            _live_sessions[session_id] = session
        else:
            reusable_session_id = reusable.session_id
    if reusable_session_id is not None:
        _ensure_live_worker(reusable_session_id)
        return _live_session_status(_get_live_session_record(reusable_session_id))
    _ensure_live_worker(session_id)
    return _live_session_status(session)


@app.get("/api/live/sessions/{session_id}", response_model=LiveSessionStatus)
def get_live_session(session_id: str) -> LiveSessionStatus:
    return _live_session_status(_get_live_session_record(session_id))


@app.post("/api/live/sessions/{session_id}/actions", response_model=LiveSessionStatus)
def queue_live_action(session_id: str, req: LiveActionRequest) -> LiveSessionStatus:
    with _live_lock:
        session = _live_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="live session not found")
        if session.closed:
            raise HTTPException(status_code=409, detail="live session is closed")
        now = time.time()
        segment_index = len(session.actions)
        action = LiveActionRecord(
            action_id=uuid.uuid4().hex[:12],
            text=req.text,
            prompt=_compose_live_prompt(session, req.text, segment_index),
            segment_index=segment_index,
            created_at=now,
            updated_at=now,
            user_visible=True,
        )
        session.actions.append(action)
        session.updated_at = now
    _ensure_live_worker(session_id)
    return _live_session_status(session)


@app.delete("/api/live/sessions/{session_id}", response_model=LiveSessionStatus)
def stop_live_session(session_id: str) -> LiveSessionStatus:
    with _live_lock:
        session = _live_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="live session not found")
        session.closed = True
        now = time.time()
        for action in session.actions:
            if action.state == LiveActionState.queued:
                action.state = LiveActionState.failed
                action.updated_at = now
                action.error = "session_stopped"
        session.updated_at = now
    return _live_session_status(session)


@app.get("/api/live/sessions/{session_id}/actions/{action_id}/content")
def get_live_action_content(session_id: str, action_id: str) -> FileResponse:
    session = _get_live_session_record(session_id)
    action = _find_live_action(session, action_id)
    if action.state != LiveActionState.ready or action.output_path is None:
        raise HTTPException(status_code=409, detail="action segment is not ready")
    return FileResponse(action.output_path, media_type=action.media_type or "video/mp4")


@app.get("/api/remote/streams/{job_id}")
def proxy_remote_stream(job_id: str) -> StreamingResponse:
    if not REMOTE_BACKEND_URL:
        raise HTTPException(status_code=404, detail="remote backend is not configured")
    if not _remote_stream_proxy_enabled():
        raise HTTPException(status_code=404, detail="remote stream proxy is disabled")

    def stream_bytes():
        with _remote_dns_override(), httpx.Client(timeout=None, follow_redirects=True) as client:
            with client.stream("GET", f"{REMOTE_BACKEND_URL}/streams/{job_id}") as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk

    return StreamingResponse(stream_bytes(), media_type="video/mp4")


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    with _lock:
        status = _jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@app.get("/api/jobs/{job_id}/content")
def get_content(job_id: str) -> FileResponse:
    with _lock:
        status = _jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    if status.state != JobState.complete or status.output_url is None:
        raise HTTPException(status_code=409, detail="job is not complete")
    if status.output_url.startswith("http://") or status.output_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="remote content should be fetched from output_url")
    return FileResponse(status.output_url, media_type=status.media_type or "video/mp4")


def _set_job(job_id: str, **updates: object) -> None:
    with _lock:
        current = _jobs[job_id]
        data = current.model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        _jobs[job_id] = JobStatus(**data)


def _run_job(job_id: str, req: GenerateRequest) -> None:
    _set_job(job_id, state=JobState.running)
    try:
        result = _execute_generation(job_id, req)
        _set_job(
            job_id,
            state=JobState.complete,
            **result,
        )
    except Exception as exc:  # noqa: BLE001 - job errors must be visible in UI.
        _set_job(job_id, state=JobState.failed, error=repr(exc))


def _execute_generation(job_id: str, req: GenerateRequest) -> dict[str, object]:
    start = time.perf_counter()
    bucket = resolve_bucket(req.duration_s, req.tier, req.bucket)
    if REMOTE_BACKEND_URL:
        remote = _run_remote(job_id, req)
        output_url = _public_output_url(job_id, str(remote["output_url"]))
        runtime_s = float(remote["runtime_s"])
        time_to_first_video_byte_s = _optional_float(remote.get("time_to_first_video_byte_s"))
        bucket_name = str(remote.get("bucket") or bucket.name)
        duration_s = float(remote.get("target_duration_s") or bucket.output_duration_s)
        media_type = str(remote.get("media_type") or "video/mp4")
        stage_times = _stage_times_from_remote(remote)
        encode_detached = bool(remote.get("encode_detached", False))
        encode_complete = bool(remote.get("encode_complete", True))
        output_path = str(remote.get("output_path") or "")
        residual_cache = remote.get("residual_cache") if isinstance(remote.get("residual_cache"), dict) else None
    else:
        output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
        make_mock_video(Path(output_path), req.prompt, duration_s=float(req.duration_s))
        runtime_s = time.perf_counter() - start
        time_to_first_video_byte_s = runtime_s
        output_url = output_path
        bucket_name = bucket.name
        duration_s = bucket.output_duration_s
        media_type = "video/mp4"
        stage_times = {"mock_generate_s": runtime_s}
        encode_detached = False
        encode_complete = True
        residual_cache = None

    realtime_factor = runtime_s / duration_s
    first_byte_realtime_factor = time_to_first_video_byte_s / duration_s if time_to_first_video_byte_s is not None else None
    faster_than_realtime = (
        first_byte_realtime_factor < 1.0 if first_byte_realtime_factor is not None else realtime_factor < 1.0
    )
    return {
        "bucket": bucket_name,
        "duration_s": duration_s,
        "target_runtime_s": duration_s,
        "runtime_s": runtime_s,
        "time_to_first_video_byte_s": time_to_first_video_byte_s,
        "realtime_factor": realtime_factor,
        "first_byte_realtime_factor": first_byte_realtime_factor,
        "faster_than_realtime": faster_than_realtime,
        "output_url": output_url,
        "output_path": output_path,
        "media_type": media_type,
        "encode_detached": encode_detached,
        "encode_complete": encode_complete,
        "continuation_video_path": req.continuation_video_path,
        "continuation_frames": req.continuation_frames if req.continuation_video_path else 0,
        "residual_cache": residual_cache,
        "stage_times": stage_times,
    }


def _get_live_session_record(session_id: str) -> LiveSessionRecord:
    with _live_lock:
        session = _live_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="live session not found")
    return session


def _find_live_action(session: LiveSessionRecord, action_id: str) -> LiveActionRecord:
    for action in session.actions:
        if action.action_id == action_id:
            return action
    raise HTTPException(status_code=404, detail="live action not found")


def _find_reusable_live_session_locked(req: LiveSessionCreateRequest, bucket_name: str) -> LiveSessionRecord | None:
    requested_key = (
        req.prompt,
        req.tier,
        req.duration_s,
        req.audio,
        req.seed,
        req.initial_image_path,
        req.initial_image_strength,
        req.continuity_frames,
        req.continuity_strength,
        bucket_name,
        req.live_cache_mode,
        req.live_cache_threshold,
        req.live_cache_max_skips,
        req.live_cache_retention_ratio,
        req.live_cache_metric_element_stride,
        req.live_cache_refresh_interval,
    )
    for session in _live_sessions.values():
        existing_key = (
            session.base_prompt,
            session.tier,
            session.duration_s,
            session.audio,
            session.seed,
            session.initial_image_path,
            session.initial_image_strength,
            session.continuity_frames,
            session.continuity_strength,
            session.bucket,
            session.live_cache_mode,
            session.live_cache_threshold,
            session.live_cache_max_skips,
            session.live_cache_retention_ratio,
            session.live_cache_metric_element_stride,
            session.live_cache_refresh_interval,
        )
        if existing_key != requested_key:
            continue
        if any(action.state != LiveActionState.failed for action in session.actions):
            return session
    return None


def _append_auto_continue_if_needed_locked(session: LiveSessionRecord) -> None:
    if session.closed:
        return
    if not _live_auto_continue_enabled():
        return
    if not session.actions:
        return
    if any(action.state in {LiveActionState.queued, LiveActionState.running} for action in session.actions):
        return
    if session.actions[-1].state != LiveActionState.ready:
        return
    now = time.time()
    segment_index = len(session.actions)
    action = LiveActionRecord(
        action_id=uuid.uuid4().hex[:12],
        text="continue",
        prompt=_compose_live_prompt(session, "continue", segment_index),
        segment_index=segment_index,
        created_at=now,
        updated_at=now,
        user_visible=False,
    )
    session.actions.append(action)
    session.updated_at = now


def _live_auto_continue_enabled() -> bool:
    raw = os.getenv("LTX_LIVE_AUTO_CONTINUE")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no", "off"}
    return bool(REMOTE_BACKEND_URL)


def _live_magcache_enabled(session: LiveSessionRecord | None = None) -> bool:
    if session is not None and session.live_cache_mode == "off":
        return False
    if session is not None and session.live_cache_mode in {"magcache", "teacache"}:
        return True
    raw = os.getenv("LTX_LIVE_MAGCACHE")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no", "off"}
    return bool(REMOTE_BACKEND_URL)


def _live_magcache_threshold(session: LiveSessionRecord | None = None) -> float:
    if session is not None and session.live_cache_threshold is not None:
        return float(session.live_cache_threshold)
    return float(os.getenv("LTX_LIVE_MAGCACHE_THRESHOLD", "0.02"))


def _live_magcache_max_skips(session: LiveSessionRecord | None = None) -> int:
    if session is not None and session.live_cache_max_skips is not None:
        return int(session.live_cache_max_skips)
    return int(os.getenv("LTX_LIVE_MAGCACHE_MAX_SKIPS", "1"))


def _live_magcache_retention_ratio(session: LiveSessionRecord | None = None) -> float:
    if session is not None and session.live_cache_retention_ratio is not None:
        return float(session.live_cache_retention_ratio)
    return float(os.getenv("LTX_LIVE_MAGCACHE_RETENTION_RATIO", "0.25"))


def _live_magcache_metric_element_stride(session: LiveSessionRecord | None = None) -> int:
    if session is not None and session.live_cache_metric_element_stride is not None:
        return int(session.live_cache_metric_element_stride)
    return int(os.getenv("LTX_LIVE_MAGCACHE_METRIC_ELEMENT_STRIDE", "64"))


def _live_magcache_refresh_interval(session: LiveSessionRecord | None = None) -> int:
    if session is not None and session.live_cache_refresh_interval is not None:
        return int(session.live_cache_refresh_interval)
    return int(os.getenv("LTX_LIVE_MAGCACHE_REFRESH_INTERVAL", "6"))


def _live_cache_mode(session: LiveSessionRecord | None = None) -> str:
    mode = (session.live_cache_mode if session is not None and session.live_cache_mode else None) or os.getenv(
        "LTX_LIVE_CACHE_MODE", "magcache"
    )
    mode = mode.strip().lower()
    if mode not in {"magcache", "teacache"}:
        raise ValueError(f"unsupported live cache mode: {mode}")
    return mode


def _select_live_cache_policy(session: LiveSessionRecord, action: LiveActionRecord) -> LiveCachePolicy:
    if not _live_magcache_enabled(session):
        return LiveCachePolicy(mode="off", reason="disabled", refresh=False, ratio_version=0)
    live_cache_mode = _live_cache_mode(session)
    if live_cache_mode == "teacache":
        return LiveCachePolicy(
            mode="teacache",
            reason="teacache_chunk",
            refresh=False,
            ratio_version=session.magcache_ratio_version,
        )
    if action.segment_index == 0:
        return LiveCachePolicy(
            mode="calibrate",
            reason="initial_segment",
            refresh=True,
            ratio_version=session.magcache_ratio_version,
        )
    if action.user_visible:
        return LiveCachePolicy(
            mode="calibrate",
            reason="user_text_changed",
            refresh=True,
            ratio_version=session.magcache_ratio_version,
        )
    if session.magcache_force_refresh_next:
        return LiveCachePolicy(
            mode="calibrate",
            reason="previous_cache_guard_refused_skip",
            refresh=True,
            ratio_version=session.magcache_ratio_version,
        )
    if not session.magcache_ratios:
        return LiveCachePolicy(
            mode="calibrate",
            reason="no_calibrated_ratios",
            refresh=True,
            ratio_version=session.magcache_ratio_version,
        )
    refresh_interval = _live_magcache_refresh_interval(session)
    if refresh_interval > 0 and session.magcache_chunks_since_refresh >= refresh_interval:
        return LiveCachePolicy(
            mode="calibrate",
            reason="periodic_refresh",
            refresh=True,
            ratio_version=session.magcache_ratio_version,
        )
    return LiveCachePolicy(
        mode="magcache",
        reason="calibrated_hidden_continue",
        refresh=False,
        ratio_version=session.magcache_ratio_version,
        ratios=tuple(session.magcache_ratios),
    )


def _live_cache_request_kwargs(policy: LiveCachePolicy, session: LiveSessionRecord | None = None) -> dict[str, object]:
    if policy.mode == "off":
        return {}
    kwargs: dict[str, object] = {
        "residual_cache_mode": policy.mode,
        "allow_quality_risk_residual_cache": True,
        "residual_cache_threshold": _live_magcache_threshold(session),
        "residual_cache_max_skips": _live_magcache_max_skips(session),
        "residual_cache_retention_ratio": _live_magcache_retention_ratio(session),
        "residual_cache_metric_element_stride": _live_magcache_metric_element_stride(session),
    }
    if policy.mode == "magcache":
        kwargs["residual_cache_mag_ratios"] = list(policy.ratios)
    return kwargs


def _update_live_cache_after_result(session: LiveSessionRecord, action: LiveActionRecord) -> None:
    if not _live_magcache_enabled(session) or action.cache_policy == "off":
        return
    stats = action.residual_cache if isinstance(action.residual_cache, dict) else None
    if stats is None:
        if action.cache_refresh:
            session.magcache_force_refresh_next = True
        return
    mode = str(stats.get("mode") or action.cache_policy)
    if mode == "calibrate":
        ratios = _float_list(stats.get("mag_calibration_ratios"))
        if ratios:
            session.magcache_ratios = ratios
            session.magcache_ratio_version += 1
            session.magcache_last_refresh_segment = action.segment_index
            session.magcache_chunks_since_refresh = 0
            session.magcache_force_refresh_next = False
            action.cache_ratio_version = session.magcache_ratio_version
        else:
            session.magcache_force_refresh_next = True
        return
    if mode == "magcache":
        skipped_steps = _int_list(stats.get("skipped_steps"))
        if skipped_steps:
            session.magcache_chunks_since_refresh += 1
            session.magcache_force_refresh_next = False
        else:
            # A guarded MagCache run that skips nothing bought no latency and did
            # not refresh ratios, so make the next hidden chunk dense/calibrating.
            session.magcache_force_refresh_next = True


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    parsed: list[float] = []
    for item in value:
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            return []
    return parsed


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    parsed: list[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            return []
    return parsed


def _ensure_live_worker(session_id: str) -> None:
    with _live_lock:
        session = _live_sessions.get(session_id)
        if session is None:
            return
        _append_auto_continue_if_needed_locked(session)
        if session.processing:
            return
        session.processing = True
        session.updated_at = time.time()
    threading.Thread(target=_drain_live_session, args=(session_id,), daemon=True).start()


def _drain_live_session(session_id: str) -> None:
    while True:
        with _live_lock:
            session = _live_sessions.get(session_id)
            if session is None:
                return
            if session.closed:
                now = time.time()
                for item in session.actions:
                    if item.state == LiveActionState.queued:
                        item.state = LiveActionState.failed
                        item.updated_at = now
                        item.error = "session_stopped"
            action = next((item for item in session.actions if item.state == LiveActionState.queued), None)
            if action is None:
                session.processing = False
                session.current_action_id = None
                session.updated_at = time.time()
                return
            action.state = LiveActionState.running
            action.updated_at = time.time()
            action.job_id = f"{session.session_id}_{action.segment_index:04d}_{action.action_id}"
            if REMOTE_BACKEND_URL:
                action.output_url = _predict_remote_stream_url(action.job_id)
                action.media_type = "video/mp4"
            session.current_action_id = action.action_id
            session.updated_at = action.updated_at
            previous_ready = (
                session.actions[action.segment_index - 1]
                if action.segment_index > 0 and session.actions[action.segment_index - 1].state == LiveActionState.ready
                else None
            )
            continuation_video_path = (
                previous_ready.output_path
                if previous_ready is not None and session.continuity_frames > 0
                else None
            )
            action.prompt = _compose_live_prompt(session, action.text, action.segment_index)
            action.continuation_video_path = continuation_video_path
            action.continuation_frames = session.continuity_frames if continuation_video_path is not None else 0
            cache_policy = _select_live_cache_policy(session, action)
            action.cache_policy = cache_policy.mode
            action.cache_reason = cache_policy.reason
            action.cache_refresh = cache_policy.refresh
            action.cache_ratio_version = cache_policy.ratio_version
            req = GenerateRequest(
                prompt=action.prompt,
                duration_s=session.duration_s,
                tier=session.tier,  # type: ignore[arg-type]
                bucket=session.bucket,
                seed=(session.seed + action.segment_index if session.seed is not None else None),
                audio=session.audio,
                image_conditioning_path=session.initial_image_path if action.segment_index == 0 else None,
                image_conditioning_frame_idx=0,
                image_conditioning_strength=session.initial_image_strength,
                continuation_mode="latent",
                continuation_video_path=continuation_video_path,
                continuation_frames=action.continuation_frames,
                continuation_strength=session.continuity_strength,
                **_live_cache_request_kwargs(cache_policy, session),
            )
            job_id = action.job_id
        try:
            result = _execute_generation(job_id, req)
            with _live_lock:
                session = _live_sessions.get(session_id)
                if session is None:
                    return
                action = _find_live_action(session, action.action_id)
                action.state = LiveActionState.ready
                action.updated_at = time.time()
                action.bucket = str(result["bucket"])
                action.output_path = str(result.get("output_path") or "")
                raw_output_url = str(result["output_url"])
                action.output_url = _public_output_url(job_id, raw_output_url)
                if not action.output_url.startswith(("http://", "https://", "/api/remote/streams/")):
                    action.output_url = f"/api/live/sessions/{session_id}/actions/{action.action_id}/content"
                action.media_type = str(result.get("media_type") or "video/mp4")
                action.runtime_s = _optional_float(result.get("runtime_s"))
                action.time_to_first_video_byte_s = _optional_float(result.get("time_to_first_video_byte_s"))
                action.realtime_factor = _optional_float(result.get("realtime_factor"))
                action.first_byte_realtime_factor = _optional_float(result.get("first_byte_realtime_factor"))
                action.faster_than_realtime = bool(result.get("faster_than_realtime"))
                action.encode_detached = bool(result.get("encode_detached", False))
                action.encode_complete = bool(result.get("encode_complete", True))
                action.continuation_video_path = (
                    str(result["continuation_video_path"])
                    if result.get("continuation_video_path") is not None
                    else action.continuation_video_path
                )
                action.continuation_frames = int(result.get("continuation_frames") or action.continuation_frames)
                action.residual_cache = (
                    dict(result["residual_cache"])
                    if isinstance(result.get("residual_cache"), dict)
                    else None
                )
                _update_live_cache_after_result(session, action)
                action.stage_times = {
                    str(key): float(value)
                    for key, value in dict(result.get("stage_times") or {}).items()
                    if isinstance(value, (int, float))
                }
                session.current_output_url = action.output_url
                session.updated_at = action.updated_at
                _append_auto_continue_if_needed_locked(session)
        except Exception as exc:  # noqa: BLE001 - queue state must show failures.
            with _live_lock:
                session = _live_sessions.get(session_id)
                if session is None:
                    return
                action = _find_live_action(session, action.action_id)
                action.state = LiveActionState.failed
                action.updated_at = time.time()
                action.error = repr(exc)
                session.updated_at = action.updated_at


def _compose_live_prompt(session: LiveSessionRecord, action_text: str, segment_index: int) -> str:
    if segment_index == 0:
        return session.base_prompt
    if action_text == "continue":
        return (
            f"{session.base_prompt}\n"
            "Continue the same scene, character identity, camera language, lighting, and audio bed. "
            "No new text command is present; keep the existing motion and world state continuing naturally."
        )
    return (
        f"{session.base_prompt}\n"
        "Continue the same scene, character identity, camera language, lighting, and audio bed. "
        f"Next action: {action_text}."
    )


def _live_session_status(session: LiveSessionRecord) -> LiveSessionStatus:
    with _live_lock:
        actions = [_live_action_status(action) for action in session.actions]
        return LiveSessionStatus(
            session_id=session.session_id,
            base_prompt=session.base_prompt,
            tier=session.tier,  # type: ignore[arg-type]
            duration_s=session.duration_s,
            target_duration_s=resolve_bucket(session.duration_s, session.tier, session.bucket).output_duration_s,
            audio=session.audio,
            seed=session.seed,
            initial_image_path=session.initial_image_path,
            initial_image_strength=session.initial_image_strength,
            continuity_frames=session.continuity_frames,
            continuity_strength=session.continuity_strength,
            bucket=session.bucket,
            created_at=session.created_at,
            updated_at=session.updated_at,
            closed=session.closed,
            processing=session.processing,
            current_action_id=session.current_action_id,
            current_output_url=session.current_output_url,
            cache_enabled=_live_magcache_enabled(session),
            cache_policy=f"{_live_cache_mode(session)}-refresh" if _live_magcache_enabled(session) else "off",
            cache_ratio_version=session.magcache_ratio_version,
            cache_ratio_count=len(session.magcache_ratios),
            cache_chunks_since_refresh=session.magcache_chunks_since_refresh,
            cache_refresh_interval=_live_magcache_refresh_interval(session),
            actions=actions,
        )


def _live_action_status(action: LiveActionRecord) -> LiveActionStatus:
    return LiveActionStatus(
        action_id=action.action_id,
        state=action.state,
        text=action.text,
        prompt=action.prompt,
        segment_index=action.segment_index,
        user_visible=action.user_visible,
        created_at=action.created_at,
        updated_at=action.updated_at,
        job_id=action.job_id,
        bucket=action.bucket,
        output_url=action.output_url,
        media_type=action.media_type,
        runtime_s=action.runtime_s,
        time_to_first_video_byte_s=action.time_to_first_video_byte_s,
        realtime_factor=action.realtime_factor,
        first_byte_realtime_factor=action.first_byte_realtime_factor,
        faster_than_realtime=action.faster_than_realtime,
        encode_detached=action.encode_detached,
        encode_complete=action.encode_complete,
        continuation_video_path=action.continuation_video_path,
        continuation_frames=action.continuation_frames,
        cache_policy=action.cache_policy,
        cache_reason=action.cache_reason,
        cache_refresh=action.cache_refresh,
        cache_ratio_version=action.cache_ratio_version,
        residual_cache=action.residual_cache,
        stage_times=action.stage_times,
        error=action.error,
    )


def _predict_remote_stream_url(job_id: str) -> str:
    if _remote_stream_proxy_enabled():
        return f"/api/remote/streams/{job_id}"
    return f"{REMOTE_BACKEND_URL}/streams/{job_id}"


def _public_output_url(job_id: str, raw_output_url: str) -> str:
    if REMOTE_BACKEND_URL and raw_output_url.startswith(("http://127.0.0.1", "http://localhost")):
        return _predict_remote_stream_url(job_id)
    if REMOTE_BACKEND_URL and raw_output_url.endswith(f"/streams/{job_id}"):
        return _predict_remote_stream_url(job_id)
    if REMOTE_BACKEND_URL and raw_output_url.startswith("/"):
        return _predict_remote_stream_url(job_id)
    return raw_output_url


def _remote_stream_proxy_enabled() -> bool:
    raw = os.getenv("LTX_PROXY_REMOTE_STREAMS")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no", "off"}
    return bool(os.getenv("LTX_REMOTE_BACKEND_RESOLVE_IPS", "").strip())


def _run_remote(job_id: str, req: GenerateRequest) -> dict[str, object]:
    start = time.perf_counter()
    payload = req.model_dump()
    payload["job_id"] = job_id
    with _remote_generate_lock:
        last_exc: httpx.TransportError | None = None
        for attempt in range(4):
            try:
                with _remote_dns_override(), httpx.Client(timeout=None, follow_redirects=True) as client:
                    response = client.post(f"{REMOTE_BACKEND_URL}/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()
                break
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == 3:
                    raise
                time.sleep(1.0 + attempt * 2.0)
        else:
            raise RuntimeError("remote generation failed without a response") from last_exc
    data.setdefault("runtime_s", time.perf_counter() - start)
    return data


@contextmanager
def _remote_dns_override():
    raw_ips = os.getenv("LTX_REMOTE_BACKEND_RESOLVE_IPS", "")
    if not raw_ips.strip() or not REMOTE_BACKEND_URL:
        yield
        return
    host = urlparse(REMOTE_BACKEND_URL).hostname
    ips = [item.strip() for item in raw_ips.split(",") if item.strip()]
    if not host or not ips:
        yield
        return
    original_getaddrinfo = socket.getaddrinfo

    def patched_getaddrinfo(name, port, family=0, type=0, proto=0, flags=0):  # type: ignore[no-untyped-def]
        if name == host:
            results = []
            for ip in ips:
                results.extend(original_getaddrinfo(ip, port, family, type, proto, flags))
            return results
        return original_getaddrinfo(name, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _stage_times_from_remote(remote: dict[str, object]) -> dict[str, float]:
    raw = remote.get("stage_times")
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    stage_keys = (
        "prompt_s",
        "denoise_s",
        "decode_video_s",
        "encode_first_byte_s",
        "encode_video_s",
        "decode_audio_s",
        "generation_s",
        "save_s",
    )
    return {key: float(remote[key]) for key in stage_keys if isinstance(remote.get(key), (int, float))}


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("ltx_serve.server:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
