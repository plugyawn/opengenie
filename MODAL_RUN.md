# Modal Run Instructions

Install dependencies:

```bash
uv sync --extra dev
touch env.local
```

Run the 25s prompt-only world-continuation example on one Modal H100:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run modal run \
  modal_apps/ltx23_official_warm_experiments.py \
  --mode forest_prompt_right_model \
  --bucket std_16x9_25fps_5s_overlap32_v1 \
  --attention sdpa-cudnn \
  --seed 20260611 \
  --continuation-frames 17 \
  --plan-limit 5
```

Run the live H100 worker:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run modal serve \
  modal_apps/ltx23_live_h100_worker.py
```

Check or stop Modal apps:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run modal app list
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run modal app stop --yes <app-id>
```
