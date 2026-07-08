# Runtime Config

Runtime YAML config is loaded by `face_of_agi.environment.config`.

## Required Runtime Keys

- `max_actions_per_level`
- one of `game_index`, `game_indices`, `game_ids`, or `game_selection` for most
  normal runs
- `models`

## Active Model Shape

```yaml
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  change:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  compacter:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  updater:
    agent:
      backend: vllm
      model: Qwen/Qwen3.6-35B-A3B-FP8
```

`models.change`, `models.compacter`, and
`models.updater.agent` are required active slots. `models.agent` is required by
runtime assembly. `models.shared_vlm` supplies defaults for local shared
Ollama/vLLM configs.

Supported active real backends are OpenAI, Ollama, and vLLM where implemented
by each role. Configurable and Hugging Face updater placeholders remain
development hooks and are not used by committed runtime configs.

## Modal Launcher Keys

Top-level `modal` keys are launcher-only metadata. The core runtime ignores
them when loading the game-loop config.

```yaml
modal:
  gpu: RTX-PRO-6000
```

`face_of_agi.runtime.modal_app` resolves `modal.gpu` into the single
`MODAL_GPU` value before registering the Modal runner class. If omitted, Modal
uses `H100`.

Modal run artifacts are grouped under `/vol/runs/<commit-id>/` in the
`face-of-agi-runs` Volume. The commit id is resolved by the local Modal
entrypoint before the remote runner starts; `--run-folder-name` can override it
for manual launches.

## Action Windows

`compacter_action_history_window` configures how many recent controllable
action groups are included in the compacter input. `updater_context_history_window`
configures how many raw action-history entries and strategy snapshots are
visible to updater P. `1` means only the latest raw entry/snapshot. Both
default to `20` and are non-negative integers. `0` disables the corresponding
history window.

`updater_actions_window` configures exactly how many ordered actions the agent
updater must return in `next_actions`. It is a positive integer and defaults to
`1`. A value above `1` makes the runtime execute a fixed-length
updater-planned action chain while only running change summary and
action-history compilation between middle actions.

## Animation Bundles

Post-action animation bundles are passed to change summary as full ordered
frame arrays. The runtime advances directly to the final bundle frame as the
next controllable frame. Change-summary animation inputs are cropped normally,
then split into overlapping chunks of at most `max_frames_per_call` frames
before provider calls; the default is `10`, and each later chunk starts with
the previous chunk's final frame. Each chunk is then resized so it fits within
`animation_frame_budget_coefficient` configured input-frame areas on the change
model config. The coefficient defaults to `2` and values below `2` are clamped
to `2`. Change-summary prompts always include deterministic same-color
connected-component facts. `max_nb_components` defaults to `50` and caps the
cumulative number of listed components per frame. Component rows are sorted by
larger shape first and lower duplicate count first for same-size shapes. Each
frame's component facts are extracted from the full cropped frame.
Updater configs may set `max_nb_components` to cap deterministic current-frame
components in the agent updater prompt.
Updater components use the updater crop config and render one current-frame
section with one-word rendered color names, counts, and normalized boxes.
Compacter calls attach only the final current frame. Compacter configs may
also set `max_nb_components` to cap the same deterministic current-frame
component section in the compacter prompt, using the compacter crop config.

## Memory Compatibility

The current state-memory schema stores agent context only. Local SQLite run
databases created by older runtime shapes should be reset instead of migrated.
