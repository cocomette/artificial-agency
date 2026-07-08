# Runtime Config

Runtime configs are YAML files in `src/face_of_agi/runtime/configs/`. They
select the ARC game, debug behavior, model backends, and updater backends.
Runtime loads the config and wires dependencies; orchestration owns the game
loop.

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
seed: <environment seed>
operation_mode: <arc operation mode>
game_catalog_path: <path to local game catalog>
environments_dir: <path to local games>
recordings_dir: <path to ARC recordings>
enable_visualization: <true | false>
save_recording: <true | false>
render_mode: <renderer mode or null>

max_actions_per_level: <real action budget>
use_learned_contexts: <true | false>
experimental_memory_turn_buffer: <turn count>
action_history_window: <action count>

debug_keep_all_m_states: <true | false>
debug_trace: <off | minimal | agent_decision | verbose | model_inputs>
debug_color: <auto | always | never>

models:
  shared_vlm:
    backend: <none | ollama>
    model: <provider model id>
  agent:
    backend: <openai | ollama>
    model: <provider model id>
    <agent provider keys>: <values>
  world:
    backend: <openai | ollama>
    model: <provider model id>
    <world provider keys>: <values>
  goal:
    backend: <openai | ollama>
    model: <provider model id>
    <goal provider keys>: <values>
  updater:
    world:
      backend: <openai | ollama>
      model: <provider model id>
      <updater provider keys>: <values>
    goal:
      backend: <openai | ollama>
      model: <provider model id>
      <updater provider keys>: <values>
    agent:
      backend: <openai | ollama>
      model: <provider model id>
      <updater provider keys>: <values>
    general:
      backend: <openai | ollama>
      model: <provider model id>
      <updater provider keys>: <values>
```

Required top-level keys are `game_index`, `max_actions_per_level`, and
`models`. Inside `models.updater`, all four slots are required: `world`,
`goal`, `agent`, and `general`.

## Configure In This Order

1. Pick the game: set `game_index`.
2. Pick the run length: set `max_actions_per_level`.
3. Pick debug output: set `debug_trace` and `debug_keep_all_m_states`.
4. Pick model backends: set `models.agent`, `models.world`, `models.goal`.
5. Pick updater backends: set all four `models.updater` slots.

## Top-Level Fields

Required:

| Field | Meaning |
| --- | --- |
| `game_index` | Index in `game_catalog_path`; runtime resolves it to a game id. |
| `max_actions_per_level` | Real ARC action budget for the run. |
| `models` | Model and updater wiring. |

Game/environment:

| Field | Meaning |
| --- | --- |
| `game_catalog_path` | JSON index-to-game-id catalog. Refresh with `--list-games`. |
| `game_id` | Usually leave null; normal shell resolves it from `game_index`. |
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
| `use_learned_contexts` | True hydrates learned `K`/`L` contexts and prior prediction-error baseline from SQLite at startup. False starts from configured contexts without deleting existing rows. |
| `experimental_memory_turn_buffer` | Recent experiment turns kept in `E`; must be at least 1. |
| `action_history_window` | Recent real actions shown to Agent X; must be non-negative. |

Debug:

| Field | Values |
| --- | --- |
| `debug_keep_all_m_states` | `true` or `false`; true keeps persisted rows for inspection. |
| `debug_trace` | `off`, `minimal`, `agent_decision`, `verbose`, `model_inputs`. |
| `debug_color` | `auto`, `always`, `never`; use `never` for logs/CI. |

## Models

`models` wires four roles:

| Field | Role |
| --- | --- |
| `models.agent` | Agent X: chooses actions. |
| `models.world` | World role S: predicts an action-conditioned future description after committed decisions. |
| `models.goal` | Goal role G: predicts or explains goal-relevant descriptions after committed decisions. |
| `models.updater` | Updater P: updates context after observed transitions. |
| `models.shared_vlm` | Optional shared local VLM defaults for Ollama roles. |

World and goal models maintain contexts that feed Agent X decisions and updater
P revisions.

## Generic Role Keys

Each role can use:

| Key | Meaning |
| --- | --- |
| `backend` | Provider adapter name. |
| `model` | Provider model id. Required for real OpenAI/Ollama updater slots. |
| `max_tool_calls` | Agent X tool-call budget when tool specs are configured. |
| `repair_attempts` | Agent X invalid-response repair budget. |
| provider keys | Direct provider fields such as `reasoning`, `host`, `keep_alive`, or schema/format controls. |

Unknown role keys are passed toward provider config and unsupported keys are
ignored. If a provider field is itself named `options`, use `options.options`
for those values. This currently matters for Ollama generation options.

## Supported Backends

| Role | Backends |
| --- | --- |
| `models.agent` | `openai`, `ollama` |
| `models.world` | `openai`, `ollama` |
| `models.goal` | `openai`, `ollama` |
| updater slots | `openai`, `ollama` |

Configurable Agent X and updater providers are reserved but not implemented.

## Agent X

Use `backend: openai` for OpenAI Responses. Important keys:

- `model`: OpenAI text/reasoning model.
- `reasoning`: usually `effort: low` for cheap tests.
- `input_image_size`: resize observations before sending.
- `input_image_resample`: `nearest`, `bilinear`, `bicubic`, or `lanczos`.
- `api_key_env`: defaults to `OPENAI_API_KEY`.

Use `backend: ollama` for local Ollama. Important keys:

- `model`: local Ollama model name.
- `host`: optional Ollama server URL.
- `think`: Ollama thinking flag.
- `format`: response format or schema.
- `keep_alive`: Ollama keep-alive.
- `options.options`: Ollama generation options like `temperature`, `num_ctx`,
  `num_predict`.

## World And Goal

Use `backend: openai` for hosted description-producing tools. Important keys:

- `model`: OpenAI text/reasoning model.
- `input_image_size`: resize source observation before sending.
- `reasoning`: Responses reasoning config.
- `api_key_env`: defaults to `OPENAI_API_KEY`.

World is action-conditioned. Goal is not.

Use `backend: ollama` for local description-producing tools. Important keys:

- `model`: local Ollama model name; defaults from `models.shared_vlm.model`
  when omitted and `models.shared_vlm.backend: ollama`.
- `host`, `think`, `keep_alive`, and `options`: Ollama behavior controls. These
  can inherit from `models.shared_vlm` and be overridden per role.
- `input_image_size` and `input_image_resample`: resize source observations
  before sending.

## Updater P

`models.updater` must define four slots:

| Slot | Updates |
| --- | --- |
| `world` | World game context `L^S`. |
| `goal` | Goal game context `L^G`. |
| `agent` | Agent game context `L^X`. |
| `general` | Shared general context `K` updater. |

Use `openai` for hosted text updating:

- `model`: required.
- `reasoning`: optional Responses reasoning config.
- `max_output_tokens`: cap updater output length.
- `instruction_dir`: optional prompt directory override.
- `api_key_env`: defaults to `OPENAI_API_KEY`.

Use `ollama` for local text updating:

- `model`: required.
- `format`: prefer a schema requiring `updated_context`.
- `think`, `keep_alive`, `instruction_dir`: optional behavior controls.
- `options.options`: Ollama generation options.

Updater outputs must include `updated_context`.

## Common Choices

Hosted OpenAI run:

- agent/world/goal/updater slots `backend: openai`
- set every real role's `model`
- keep `max_actions_per_level` small
- provide `OPENAI_API_KEY`

Mixed local/hosted run:

- agent `backend: ollama`
- world/goal `backend: openai`
- updater slots can be OpenAI or Ollama independently

## Failure Checks

The loader fails early for missing required fields, invalid debug values,
negative `action_history_window`, `experimental_memory_turn_buffer < 1`,
unknown backends, missing updater slots, or real OpenAI/Ollama updater slots
without a `model`.

Runtime can still fail if the game index is not in the catalog, credentials
are missing, Ollama is unavailable, a disabled tool is requested, world/goal
prediction roles are enabled without world/goal models, or provider
responses fail the description schema.
