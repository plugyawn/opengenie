from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


DEFAULT_PROMPT = (
    "Third-person videogame camera following one adventurer walking through a dense pine forest. "
    "Stable character identity, coherent forest layout, natural forward camera motion, generated ambient forest audio, "
    "cinematic game footage, no HUD, no UI, no text overlays, no subtitles."
)

DEFAULT_ACTIONS = [
    "press forward, walk along the forest trail",
    "turn slightly right while continuing forward",
    "slowly approach a mossy fallen tree",
    "step around the tree and keep the camera behind the character",
    "continue forward into a darker cluster of pines",
]


def _read_actions(raw: list[str], path: Path | None) -> list[str]:
    if path is not None:
        actions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [line for line in actions if line and not line.startswith("#")]
    return raw or list(DEFAULT_ACTIONS)


def _url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _session_ready(session: dict[str, Any], expected_actions: int) -> bool:
    actions = session.get("actions") or []
    if len(actions) < expected_actions:
        return False
    return all(action.get("state") in {"ready", "failed"} for action in actions)


def _download_segment(client: httpx.Client, base_url: str, action: dict[str, Any], output_path: Path) -> None:
    raw_url = str(action.get("output_url") or "")
    if not raw_url:
        raise RuntimeError(f"action {action.get('segment_index')} has no output_url")
    url = raw_url if raw_url.startswith(("http://", "https://")) else _url(base_url, raw_url)
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
    if output_path.stat().st_size <= 0:
        raise RuntimeError(f"downloaded empty segment for action {action.get('segment_index')} from {url}")


def _rsync_remote_jobs(
    *,
    ssh: str,
    ssh_option: list[str],
    remote_root: str,
    job_ids: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for job_id in job_ids:
        remote = f"{ssh}:{remote_root.rstrip('/')}/{job_id}/"
        local = output_dir / job_id
        local.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-az"]
        if ssh_option:
            cmd.extend(["-e", "ssh " + " ".join(ssh_option)])
        cmd.extend(
            [
                "--include=video_latent.pt",
                "--include=audio_latent.pt",
                "--exclude=*",
                remote,
                str(local) + "/",
            ]
        )
        subprocess.run(cmd, check=True)


def _python_with_torch(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [sys.executable, "python", "/opt/anaconda3/bin/python", "python3"]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-c", "import torch"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return candidate
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and collect a multi-action live continuity session.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8768")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--actions-file", type=Path)
    parser.add_argument("--duration-s", type=int, default=5)
    parser.add_argument("--tier", default="realtime", choices=["realtime", "fast", "standard", "premium"])
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--initial-image-path", default=None)
    parser.add_argument("--initial-image-strength", type=float, default=1.0)
    parser.add_argument("--continuity-frames", type=int, default=22)
    parser.add_argument("--continuity-strength", type=float, default=1.0)
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=420.0)
    parser.add_argument("--remote-ssh", default="")
    parser.add_argument("--remote-output-root", default="/home/ubuntu/video_gen/outputs/live_worker")
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-python", default=None, help="Python interpreter with torch for latent .pt eval.")
    args = parser.parse_args()

    actions = _read_actions(args.action, args.actions_file)
    expected_actions = 1 + len(actions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=None) as client:
        create_payload = {
            "prompt": args.prompt,
            "duration_s": args.duration_s,
            "tier": args.tier,
            "seed": args.seed,
            "audio": True,
            "initial_image_path": args.initial_image_path,
            "initial_image_strength": args.initial_image_strength,
            "continuity_frames": args.continuity_frames,
            "continuity_strength": args.continuity_strength,
        }
        response = client.post(_url(args.base_url, "/api/live/sessions"), json=create_payload)
        response.raise_for_status()
        session = response.json()
        _write_json(args.output_dir / "session_create.json", session)
        session_id = session["session_id"]

        for index, action in enumerate(actions, start=1):
            response = client.post(_url(args.base_url, f"/api/live/sessions/{session_id}/actions"), json={"text": action})
            response.raise_for_status()
            _write_json(args.output_dir / f"queue_action_{index}.json", response.json())

        deadline = time.monotonic() + args.timeout_s
        while True:
            response = client.get(_url(args.base_url, f"/api/live/sessions/{session_id}"))
            response.raise_for_status()
            session = response.json()
            _write_json(args.output_dir / "session_latest.json", session)
            if _session_ready(session, expected_actions):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for live session {session_id}")
            time.sleep(args.poll_s)

        _write_json(args.output_dir / "session_final.json", session)
        failed = [action for action in session["actions"] if action.get("state") == "failed"]
        if failed:
            raise RuntimeError(f"live session has failed actions: {failed}")

        for action in session["actions"]:
            segment_path = args.output_dir / f"segment_{int(action['segment_index'])}.mp4"
            _download_segment(client, args.base_url, action, segment_path)

    job_ids = [str(action["job_id"]) for action in session["actions"] if action.get("job_id")]
    if args.remote_ssh:
        _rsync_remote_jobs(
            ssh=args.remote_ssh,
            ssh_option=args.ssh_option,
            remote_root=args.remote_output_root,
            job_ids=job_ids,
            output_dir=args.output_dir / "remote_jobs",
        )

    if args.run_eval:
        remote_jobs_dir = args.output_dir / "remote_jobs"
        if not remote_jobs_dir.exists():
            raise RuntimeError("--run-eval requires --remote-ssh or preexisting remote_jobs directory")
        eval_python = _python_with_torch(args.eval_python)
        subprocess.run(
            [
                eval_python,
                "scripts/eval_live_continuity.py",
                "--session",
                str(args.output_dir / "session_final.json"),
                "--output-dir",
                str(args.output_dir),
                "--remote-jobs-dir",
                str(remote_jobs_dir),
            ],
            check=True,
        )

    print(args.output_dir / "session_final.json")


if __name__ == "__main__":
    main()
