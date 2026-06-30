# Runtime Config

Runtime configs are YAML files in `src/face_of_agi/runtime/configs/`. They
select the ARC game, debug behavior, vLLM transport defaults, text observation
serialization, and model role wiring. Runtime loads the config and wires
dependencies; orchestration owns the game loop.

Run one config with:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config path/to/config.yaml
```

## Overall Shape

The config is one YAML mapping with runtime fields at the top and model wiring
under `models`:

```yaml
game_index: <catalog index>
game_id: <optional resolved game id>
game_indices: <optional catalog indices>
game_ids: <optional explicit game ids>
game_selection: <optional all_available>
max_parallel_games: <optional worker count>
max_game_retries: <retry count>
seed: <environment seed>
operation_mode: <arc operation mode>
game_catalog_path: <path to local game catalog>
environments_dir: <path to local games>
recordings_dir: <path to ARC recordings>
enable_visualization: <true | false>
save_recording: <true | false>
render_mode: <renderer mode or null>

max_actions_per_level: <real action budget>
max_levels_per_game: <optional completed-level cap>
use_learned_contexts: <true | false>
experimental_memory_turn_buffer: <turn count>
agent_action_history_window: <action count>
agent_updater_action_history_window: <action count>
agent_context_history_window: <context revision count>
animation_keyframe_pixel_threshold: <cell count>
action_suppression_zero_changed_pixel_turns: <turn count>
updater_stagnation_warning_zero_changed_pixel_turns: <turn count>

debug_keep_all_m_states: <true | false>
debug_trace: <off | minimal | agent_decision | verbose | model_inputs>
debug_color: <auto | always | never>
live_turn_monitor: <true | false>

models:
  observation_text:
    crop_cells: 3
    overflow_chars_per_frame: 12000
    include_rows: true
    include_component_runs: true
    compact_components: false
  shared_vlm:
    backend: vllm
    model: <vLLM model id>
    base_url: http://127.0.0.1:8000/v1
    api_key: EMPTY
  agent:
    backend: vllm
    max_tool_calls: 0
    repair_attempts: 1
  change:
    backend: vllm
    max_frames_per_call: 5
    reduce_chunk_summaries: true
    reducer_keyframe_limit: 6
  historizer:
    backend: vllm
  updater:
    agent:
      backend: vllm
    general:
      backend: vllm
```

Required top-level keys are one game selector, `max_actions_per_level`, and
`models`. `models.change`, `models.updater.agent`, and
`models.updater.general` are required. `models.agent` is required by runtime
assembly. `models.historizer` is optional; when present, its backend must be
`vllm`.

## Configure In This Order

1. Pick the game: set exactly one of `game_index`, `game_indices`, `game_ids`,
   or `game_selection`.
2. Pick the run length: set `max_actions_per_level` and optionally
   `max_levels_per_game`.
3. Point `models.shared_vlm` at a running vLLM OpenAI-compatible server.
4. Keep every real model role on `backend: vllm`.
5. Tune `models.observation_text` only if the prompt budget requires it.
6. Pick debug output: set `debug_trace` and `debug_keep_all_m_states`.

## Top-Level Fields

Required:

| Field | Meaning |
| --- | --- |
| game selector | Exactly one of `game_index`, `game_indices`, `game_ids`, or `game_selection`. |
| `max_actions_per_level` | Real ARC action budget for each level. |
| `models` | Model role and vLLM transport wiring. |

Game/environment:

| Field | Meaning |
| --- | --- |
| `game_catalog_path` | JSON index-to-game-id catalog. Refresh with `--list-games`. |
| `game_id` | Usually leave null; normal shell resolves it from `game_index`. |
| `game_indices` | Multiple catalog indices for parallel runs. |
| `game_ids` | Explicit game ids for parallel runs. |
| `game_selection` | `all_available` to run every catalog entry. |
| `max_parallel_games` | Optional worker limit for parallel runs. |
| `max_game_retries` | Retry count for failed parallel games. |
| `operation_mode` | Keep `offline` for normal local runs. |
| `environments_dir` | Local ARC game files directory. |
| `seed` | ARC environment seed. |

Output/rendering:

| Field | Meaning |
| --- | --- |
| `recordings_dir` | ARC native recording directory. |
| `save_recording` | True asks ARC to save native recordings. It is unrelated to SQLite memory. |
| `enable_visualization` | True enables local rendering. |
| `render_mode` | Usually null; use a renderer mode such as `human` only with visualization. |

Context/memory:

| Field | Meaning |
| --- | --- |
| `use_learned_contexts` | True hydrates contexts from prior SQLite memory when available. |
| `experimental_memory_turn_buffer` | Recent experiment turns kept in `E`; must be at least 1. |
| `agent_action_history_window` | Recent action rows shown to Agent X; must be non-negative. |
| `agent_updater_action_history_window` | Recent action rows shown to the agent updater; must be non-negative. |
| `agent_context_history_window` | Recent agent context revisions summarized by the historizer; must be non-negative. |

Transition heuristics:

| Field | Meaning |
| --- | --- |
| `animation_keyframe_pixel_threshold` | Cropped-cell change threshold for animation keyframes. |
| `action_suppression_zero_changed_pixel_turns` | Repeated zero-change count before simple actions are omitted or exact ACTION6 coordinates are prompt-suppressed. |
| `updater_stagnation_warning_zero_changed_pixel_turns` | Repeated zero-change count before updater warning evidence is shown. |

Debug:

| Field | Values |
| --- | --- |
| `debug_keep_all_m_states` | `true` or `false`; true keeps persisted rows for inspection. |
| `debug_trace` | `off`, `minimal`, `agent_decision`, `verbose`, `model_inputs`. |
| `debug_color` | `auto`, `always`, `never`; use `never` for logs/CI. |
| `live_turn_monitor` | `true` or `false`; emits aggregate live progress during parallel runs. |

## Models

The implemented real model roles are:

| Field | Role |
| --- | --- |
| `models.agent` | Agent X: chooses actions. |
| `models.change` | Transition change summarizer. |
| `models.historizer` | Agent context history summarizer. |
| `models.updater.agent` | Agent game-context updater. |
| `models.updater.general` | Agent general-context updater. |

`models.shared_vlm` supplies defaults to any role whose backend is `vllm`.
Role-specific keys override shared keys.

Runtime rejects removed provider keys:

- `models.world`
- `models.goal`
- `models.updater.world`
- `models.updater.goal`

Runtime also rejects real backends other than `vllm`, including OpenAI, Ollama,
HuggingFace, and Diffusers. The OpenAI Python SDK remains only as the HTTP
client used to call vLLM's OpenAI-compatible Chat Completions API.

## Observation Text

`models.observation_text` controls shared model-facing observation
serialization:

| Key | Default | Meaning |
| --- | --- | --- |
| `crop_cells` | `3` | Border cells cropped from each side of a 64x64 ARC grid. |
| `overflow_chars_per_frame` | `12000` | Per-frame text budget before component listings are omitted. |
| `include_rows` | `true` | Whether to include serialized char-level cropped row text in model-facing observations. |
| `include_components` | `true` | Whether to include component listings and component-ID delta lines in model-facing observation text. |
| `include_component_runs` | `true` | Whether component listings include exact `runs=` row-span geometry. When `false`, compact component ids, symbols, area, bbox, and centroid remain. |
| `compact_components` | `false` | Whether component listings are grouped by same symbol and shape as compact `symbol`, `size`, `nb`, and `box` lines. Compact mode omits individual component ids, runs, centroids, and component-ID delta lines. |

The serializer accepts native 2D ARC integer grids only. It rejects non-ARC
frames, non-integer symbols, booleans, floats, and symbols outside `0..15`.

Model-facing text uses:

- crop bounds `x=3..60`, `y=3..60` with original ARC coordinates
- original `0..63` coordinate labels
- cropped rows with uppercase hex ARC symbols `0..F` when `include_rows` is
  enabled
- all 4-connected same-symbol components unless overflow omits the component
  list
- optional exact component `runs=` row spans when they are enabled and fit the
  per-frame budget; otherwise the serializer retries compact components before
  omitting the component list
- component-level deltas for bundled frames and change prompts
- grouped component boxes instead of individual component ids when
  `compact_components` is enabled

ACTION6 coordinates are also model-facing ARC grid coordinates in `0..63`.
New Agent X ACTION6 outputs must include a non-empty `target` string describing
the visible object, cell, or region selected by the coordinates. No normalized
image coordinates are used.

## vLLM Role Keys

Each vLLM role can use:

| Key | Meaning |
| --- | --- |
| `backend` | Must be `vllm`. |
| `model` | vLLM model id. Required after shared defaults are applied. |
| `base_url` | vLLM OpenAI-compatible `/v1` endpoint. |
| `api_key` | API key value passed to the endpoint. Local vLLM often accepts `EMPTY`. |
| `api_key_env` | Environment variable for the API key; defaults to `VLLM_API_KEY`. |
| `timeout` | Optional request timeout. |
| `max_retries` | Optional SDK retry count. |
| `default_headers` | Optional headers passed to the SDK client. |
| `default_query` | Optional query params passed to the SDK client. |
| `max_tokens` | Chat completion token cap. |
| `max_completion_tokens` | Alternate completion token cap. |
| `temperature` | Sampling temperature. |
| `top_p` | Nucleus sampling setting. |
| `seed` | Optional generation seed. |
| `max_context_tokens` | Optional prompt context limit used for overflow recovery. When omitted, shared `server.max_model_len` is passed to vLLM roles. |
| `truncate_context_on_overflow` | Whether vLLM context-overflow errors should be retried with mutable message text truncated; defaults to true. |
| `context_truncation_margin_tokens` | Prompt-token safety margin reserved below the context limit during overflow recovery; defaults to 256. |
| `context_overflow_retries` | Number of overflow truncation retries before surfacing the provider error; defaults to 3. |
| `repair_attempts` | Invalid-output repair attempts where supported. |
| `input_image_detail` | OpenAI-compatible image detail for frame-consuming roles; defaults to `auto`. |
| `input_image_size` | Optional request image resize target; runtime configs default frame-consuming roles to `2048x2048`. |
| `input_image_resample` | Image resize resampling mode; defaults to `nearest` so ARC cells stay crisp. |
| `image_mime_type` | Image data URL MIME type; defaults to `image/png`. |
| `frame_scale` | Cell-render scale before crop/resize; defaults to `4`. |
| `extra_request_options` | Additional Chat Completions request options. |

`models.change.max_frames_per_call` limits retained transition frames per
change-summary model call. Longer retained bundles are split into balanced
overlapping chunks. The removed `max_evidence_frames` key is rejected.
`models.change.reduce_chunk_summaries` defaults to `true`; when multiple chunks
are produced, a final reducer call reconciles ordered partial summaries plus
cropped images for selected keyframes. `models.change.reducer_keyframe_limit`
defaults to `6` and caps the row-only first/final/boundary keyframes sent to
that reducer.

Simple actions that repeatedly produce zero changed cells are removed from the
prompt-facing allowed action list when suppression triggers. ACTION6 is never
removed as a whole action; only the exact repeated `x,y` coordinate is listed
as prompt-suppressed, and the agent is instructed to choose a different
coordinate.

Agent X also accepts `max_tool_calls`; the current vLLM runtime expects `0`.

The shared vLLM provider preserves system/developer instructions, JSON schemas,
and image content parts when recovering from context overflow. It truncates only
mutable user/assistant text and retries the failed request. This is a last-resort
transport guard; prompt-specific budgets such as observation text overflow,
history windows, and evidence-frame limits remain the preferred way to preserve
high-value context.

The removed normalized crop override is not valid model-facing config. Image
crops always follow `models.observation_text.crop_cells`:

- `input_image_crop_box_normalized`

No README setup command is required for these image inputs. The existing vLLM
endpoint must simply support multimodal `image_url` content parts.

## Failure Checks

The loader fails early for missing required fields, invalid debug values,
negative window sizes, `experimental_memory_turn_buffer < 1`, removed
world/goal config keys, unknown backends, and real vLLM roles without a model
after shared defaults are applied.

Runtime can still fail if the game selector is invalid, the catalog has not
been created, vLLM is unavailable during startup, or the configured model is
not served. During competition-oriented game execution, orchestration is
fail-open after config load: Agent X, change-summary, historizer, and updater
exceptions degrade to deterministic role fallbacks, and unexpected loop
exceptions produce terminal `GameRunResult` records with explicit fallback
metadata instead of escaping as worker failures. SQLite memory writes and
learned-context hydration are best-effort in the game loop; failures there skip
the persistence side effect rather than stopping action submission. Worker setup
exceptions that occur before the game loop still remain visible as parallel
failure records, but the Kaggle entrypoint keeps the notebook exit status at
zero so one worker failure cannot fail the batch process.
