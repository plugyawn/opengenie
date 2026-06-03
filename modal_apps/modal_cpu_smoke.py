from __future__ import annotations

import time

import modal


app = modal.App("video-gen-cpu-smoke")


@app.function(timeout=60)
def smoke() -> dict[str, object]:
    return {"ok": True, "ts": time.time(), "platform": "modal-cpu"}


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())
