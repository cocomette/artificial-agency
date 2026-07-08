# Runtime Config Reference

Runtime configs are YAML files loaded by `face_of_agi.environment.config`.

## Top-Level Keys

Common top-level keys:

- `game_index`, `game_indices`, `game_ids`, or `game_selection`
- `max_actions_per_level`
- `max_parallel_games`
- `max_game_retries`
- `operation_mode`
- `game_catalog_path`
- `environments_dir`
- `recordings_dir`
- `seed`
- `save_recording`
- `experimental_memory_turn_buffer`
- `agent_action_history_window`
- `agent_context_history_window`
- `agent_updater_action_history_window`
- `animation_keyframe_pixel_threshold`
- `debug_keep_all_m_states`
- `debug_trace`
- `debug_color`
- `live_turn_monitor`

## Model Roles

Current model role keys:

```yaml
models:
  shared_vlm:
    backend: vllm
    model: ...
  scheduler:
    enabled: true
    max_concurrent_calls: 8
    max_concurrent_calls_per_game: 1
    queue_policy: fifo
  agent:
    backend: vllm
  change:
    backend: vllm
  historizer:
    backend: vllm
  memory:
    backend: vllm
  updater:
    agent:
      backend: vllm
    general:
      backend: vllm
```

Removed keys `models.world`, `models.goal`, `models.updater.world`, and
`models.updater.goal` are rejected by config loading in this branch.

## Shared VLM

`models.shared_vlm` supplies defaults to matching local VLM roles. Shared
fields include backend, model, repair attempts, and provider runtime options
such as vLLM server settings.

Role-specific values override shared values.

Set `models.memory.backend: none` to disable same-run game memory generation.
The runtime still provides the prompt-facing `not available` memory sentinel, so
agent and updater calls continue through the normal game loop contract.

## Model Scheduler

`models.scheduler` optionally enables a process-wide vLLM call scheduler. The
current scheduler is strict FIFO across eligible calls, with `max_concurrent_calls`
and `max_concurrent_calls_per_game` limiting shared provider pressure. Calls
from a game already at its per-game limit do not block eligible queued calls
from other games.

When enabled, vLLM role request timeouts default to
`clamp(90, 300, 60 + thinking_token_budget / 16)` unless the role sets an
explicit `timeout`. `queue_timeout_seconds` can override queue wait timeout;
otherwise the same computed request timeout is used.

## Structured Output Caps

Supported cap fields:

- `models.change.summary_max_chars`
- `models.change.summary_max_elements`
- `models.historizer.field_max_chars`
- `models.memory.memory_max_chars`
- `models.updater.general.general_context_max_chars`
- `models.updater.agent.agent_game_context_max_chars`
- `models.updater.agent.agent_game_context_field_max_chars`
- `repair_invalid_output_preview_chars` on vLLM structured-output roles

The RTX6000 vLLM configs set these explicitly:

- change/historizer field caps: `2000`
- change element cap: `20`
- memory cap: `10000`
- updater general cap: `20000`
- updater agent total cap: `12000`
- updater agent field cap: `6000`
- vLLM invalid-output preview cap: `8000`

## Observability

SQLite memory databases include `model_call_events` and
`environment_step_events` tables for runtime timing analysis. Model events
record queue lifecycle, provider start/end/error, request timeout, queue wait,
role, provider, and run/game/turn metadata. Environment events record action,
status, duration, remaining actions, and run/game/turn metadata.

## Provider Notes

Provider config keys not modeled directly by `ModelRoleConfig` are preserved in
the role `options` map and expanded into role-local dataclass config objects by
runtime assembly.

OpenAI, Ollama, and vLLM roles require explicit model names unless the selected
role config obtains one from `shared_vlm`.
