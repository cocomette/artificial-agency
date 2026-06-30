# Run The Runtime

Copy-paste these commands from the repository root.

## First Setup

For the light runtime environment:

```bash
uv sync --no-dev
```

For the full development environment:

```bash
uv sync --group dev
```

For the local debug dashboard:

```bash
uv sync --group debug
```

The runtime talks to vLLM through an OpenAI-compatible Chat Completions
endpoint. Start that vLLM server separately and set `models.shared_vlm.base_url`
in the YAML config. The `openai` Python package is only the HTTP client for this
vLLM endpoint.

## Create Or Refresh The Game Catalog

Run this once before using `game_index` from
`src/face_of_agi/runtime/configs/starter_loop.yaml`.

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

From the full development environment:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --list-games
```

This writes:

```text
src/face_of_agi/environment/local_games.json
```

## Run The Runtime

With the light runtime environment:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

From the full development environment:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

The runtime starts orchestration, serializes ARC observations into text, sends
text plus cropped image frames to frame-consuming vLLM-backed model roles, and
prints a condensed trace for each frame turn.

## Clear Runtime Memory

Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```

Use the matching `uv run --group dev ...` variant for the environment you
synced.

## Ready-To-Run Configs

These configs preserve the vLLM-only model contract. Edit `base_url`, `model`,
and hardware-specific fields for your vLLM deployment.

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml
```

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_qwen36_35b_fp8_debug.yaml
```

## Terminal-Friendly Rendering

If `render_mode: human` cannot open a window, edit the selected YAML config:

```yaml
enable_visualization: true
render_mode: terminal
```

Then run the same shell command again.

## Useful Config Values

Common top-level values to change:

- `game_index`: selected game from the catalog printed by `--list-games`
- `max_actions_per_level`: action budget before stopping
- `enable_visualization`: show frames while running
- `render_mode`: `human`, `terminal`, or `terminal-fast`
- `use_learned_contexts`: hydrate contexts from SQLite when available
- `experimental_memory_turn_buffer`: latest frame turns kept in rolling `E`
  memory; defaults to `2`
- `agent_action_history_window`: prior frame-turn actions included in each X
  prompt; defaults to `8`
- `agent_updater_action_history_window`: prior action rows sent to the agent
  updater; defaults to `8`
- `agent_context_history_window`: prior context revisions summarized by the
  historizer; defaults to `8`
- `debug_keep_all_m_states`: keep every `M` frame-turn row after a successful
  run; defaults to `false`
- `debug_trace`: stdout trace mode: `off`, `minimal`, `agent_decision`,
  `verbose`, or `model_inputs`; defaults to `minimal`
- `debug_color`: Rich color mode for debug traces: `auto`, `always`, or
  `never`; defaults to `auto`

Common model values:

- `models.observation_text.crop_cells`: crop each 64x64 ARC grid by this many
  border cells before model serialization; defaults to `3`
- `models.observation_text.overflow_chars_per_frame`: character budget before
  component listings are omitted; defaults to `12000`
- `models.observation_text.include_rows`: set `false` to omit the serialized
  char-level `#### rows` block from model-facing observation text; defaults to
  `true`
- `models.observation_text.include_components`: set `false` to omit component
  listings and component-ID delta lines from all model-facing observation text;
  defaults to `true`
- `models.observation_text.include_component_runs`: set `false` to keep
  component ids, symbols, area, bbox, and centroid while omitting the verbose
  `runs=` cell-span field; defaults to `true`
- `models.observation_text.compact_components`: set `true` to group
  same-symbol, same-shape components into compact `symbol`, `size`, `nb`, and
  `box` lines; defaults to `false`
- `models.change.max_frames_per_call`: maximum retained transition frames per
  change-summary call before balanced overlapping text chunks are used; `null`
  sends the full retained bundle in one call
- `models.change.reduce_chunk_summaries`: set `false` to skip the final
  reducer call for multi-chunk transition summaries; defaults to `true`
- `models.change.reducer_keyframe_limit`: maximum row-only first/final/boundary
  keyframes sent to the reducer, with matching cropped images; defaults to `6`
- vLLM frame-input keys: `input_image_size`, `input_image_resample`,
  `input_image_detail`, `image_mime_type`, and `frame_scale`; shared vLLM
  configs default `input_image_size` to `2048x2048`
- `models.shared_vlm.backend`: must be `vllm` when shared defaults are used
- `models.shared_vlm.model`: default model id for vLLM roles
- `models.shared_vlm.base_url`: vLLM OpenAI-compatible `/v1` endpoint
- `models.shared_vlm.api_key`: API key passed to the vLLM endpoint; many local
  setups use `EMPTY`
- `models.agent.backend`: must be `vllm`
- `models.change.backend`: must be `vllm`
- `models.historizer.backend`: optional, but must be `vllm` when configured
- `models.updater.agent.backend`: must be `vllm`
- `models.updater.general.backend`: must be `vllm`

## Text Observations

Model-facing observations are `ObservationText` strings built from native ARC
2D integer grids. The serializer crops original coordinates `x=3..60` and
`y=3..60`, prints cropped rows with original `0..63` coordinate labels and
uppercase hex symbols `0..F`, lists 4-connected same-symbol components unless
disabled by config or the per-frame overflow budget is exceeded, and includes
component-level deltas for frame bundles and change prompts. Component listings
fall back from exact `runs=` spans to compact component fields before being
omitted. `compact_components` can group components more aggressively and then
delta text keeps changed-cell counts only. Rows can be hidden while retaining
components and deltas. ACTION6 model-facing coordinates are ARC grid
coordinates in `0..63`; new Agent X ACTION6 outputs also include target text.
For long retained transition bundles, change summaries are chunked and then
optionally reduced through a final call using ordered partial summaries,
row-only keyframes, and cropped keyframe images.

SQLite memory still stores observations normally. Observation text is computed
on demand for prompts.

## Debug Trace Modes

The compact default output is:

```yaml
debug_trace: minimal
debug_color: auto
```

Use `debug_trace: verbose` for colored sections covering run start/stop, frame
turns, control policy, selected actions, reasoning summaries, transitions, and
M-state persistence.

Use `debug_trace: agent_decision` to show only the Agent X decision panel for
each frame turn.

Use `debug_trace: model_inputs` when you need to inspect model inputs. This
adds sanitized agent/change/historizer/updater input sections, final prompts,
and request metadata. Persisted model-input debug records keep raw image data
URLs, while terminal output summarizes them instead of printing full base64.
