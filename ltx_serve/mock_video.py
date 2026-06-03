from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def make_mock_video(output_path: Path, prompt: str, duration_s: float = 5.0) -> None:
    """Create a small MP4 for local UI validation without a GPU model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Keep the API contract testable even on machines without ffmpeg.
        output_path.write_bytes(b"")
        return

    label = prompt.replace(":", " ").replace("'", "").replace("\\", " ")
    label = label[:72]
    vf = (
        "color=c=0x141820:size=768x432:rate=15,"
        "format=yuv420p,"
        "drawtext=text='LOCAL MOCK PLACEHOLDER - remote LTX backend is not configured':"
        "x=24:y=32:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.65,"
        f"drawtext=text='{label}':x=24:y=82:fontcolor=0xcfd7e6:fontsize=20:"
        "box=1:boxcolor=black@0.45"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        vf,
        "-t",
        f"{duration_s:.3f}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
