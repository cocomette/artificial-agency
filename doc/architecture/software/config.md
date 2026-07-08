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
    backend: <none | ollama | vllm>
    model: <provider model id>
    server: <Modal-only vLLM server options>
  agent:
    backend: <openai | ollama | vllm>
    model: <provider model id>
    <agent provider keys>: <values>
  world:
    backend: <openai | ollama | vllm>
    model: <provider model id>
    <world provider keys>: <values>
  goal:
    backend: <openai | ollama | vllm>
    model: <provider model id>
    <goal provider keys>: <values>
  updater:
    world:
      backend: <openai | ollama | vllm>
      model: <provider model id>
      <updater provider keys>: <values>
    goal:
      backend: <openai | ollama | vllm>
      model: <provider model id>
      <updater provider keys>: <values>
    agent:
      backend: <openai | ollama | vllm>
      model: <provider model id>
      <updater provider keys>: <values>
    general:
      backend: <openai | ollama | vllm>
      model: <provider model id>
      <updater provider keys>: <values>
```

Required top-level keys are `game_index`, `max_actions_per_level`, and
`models`. Inside `models.updater`, the active slots are required: `world`,
`agent`, and `general`. The dormant `goal` updater slot is optional and ignored
by normal runtime assembly.

## Configure In This Order

1. Pick the game: set `game_index`.
2. Pick the run length: set `max_actions_per_level`.
3. Pick debug output: set `debug_trace` and `debug_keep_all_m_states`.
4. Pick model backends: set `models.agent` and `models.world`.
5. Pick updater backends: set `models.updater.world`, `models.updater.agent`,
   and `models.updater.general`.

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
| `models.goal` | Optional dormant Goal role G config; accepted but not built or called by normal runtime. |
| `models.updater` | Updater P: updates context after observed transitions. |
| `models.shared_vlm` | Optional shared local VLM defaults for matching Ollama or Modal-managed vLLM roles. |

World model context feeds Agent X decisions through updater P revisions. Goal
context/config remains available as dormant storage and model code.

## Generic Role Keys

Each role can use:

| Key | Meaning |
| --- | --- |
| `backend` | Provider adapter name. |
| `model` | Provider model id. Required for real OpenAI/Ollama updater slots and every vLLM role unless inherited from `models.shared_vlm`. |
| `max_tool_calls` | Agent X tool-call budget when tool specs are configured. |
| `repair_attempts` | Agent X invalid-response repair budget. |
| `include_output_schema_in_instructions` | Optional boolean, defaults to `false`. When true, appends the role's structured-output JSON schema to the system instructions as a plain prompt hint while still using provider-native structured-output settings. |
| provider keys | Direct provider fields such as `reasoning`, `host`, `keep_alive`, `base_url`, or schema/format controls. |

Unknown role keys are passed toward provider config and unsupported keys are
ignored. If a provider field is itself named `options`, use `options.options`
for those values. This currently matters for Ollama generation options. vLLM
passes unknown request options through Chat Completions `extra_body`.

## Supported Backends

| Role | Backends |
| --- | --- |
| `models.agent` | `openai`, `ollama`, `vllm` |
| `models.world` | `openai`, `ollama`, `vllm` |
| `models.goal` | `openai`, `ollama`, `vllm` accepted as optional dormant config but not built or called by normal runtime |
| active updater slots | `openai`, `ollama`, `vllm` |

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

Use `backend: vllm` for Modal-managed H100 runs. Important keys:

- `model`: vLLM-served Hugging Face model id; can inherit from
  `models.shared_vlm.model`.
- `base_url`: local OpenAI-compatible vLLM URL; defaults to
  `http://127.0.0.1:8000/v1`.
- `api_key` or `api_key_env`: OpenAI SDK authentication value; the sample uses
  `EMPTY`.
- `temperature`, `top_p`, `max_tokens`, and `seed`: Chat Completions sampling
  controls.
- `input_image_size` and `input_image_resample`: resize observations before
  sending.
- `models.shared_vlm.server`: Modal-only `vllm serve` options such as `host`,
  `port`, `max_model_len`, `reasoning_parser`, and `extra_args`.

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

Use `backend: vllm` for Modal-managed H100 description-producing tools.
Important keys match Agent X's vLLM Chat Completions keys. World and goal vLLM
roles require a model directly or through `models.shared_vlm.model`.

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
- `format`: prefer a schema requiring `updated_context`; the world and agent
  game updaters use stricter map schemas for that field.
- `think`, `keep_alive`, `instruction_dir`: optional behavior controls.
- `options.options`: Ollama generation options.

Use `vllm` for Modal-managed H100 text updating:

- `model`: required directly or through `models.shared_vlm.model`.
- `base_url`, `api_key`, `api_key_env`, `temperature`, `top_p`, `max_tokens`,
  and `seed`: vLLM Chat Completions behavior controls.
- `instruction_dir`: optional prompt directory override.
- `input_image_size` and `input_image_resample`: resize updater images before
  sending.

Updater outputs must include `updated_context`. For world and agent
game-context updates, `updated_context` is a complete role-specific map that the
adapter serializes back into context text.

## Common Choices

Hosted OpenAI run:

- agent, world, and active updater slots `backend: openai`
- set every real role's `model`
- keep `max_actions_per_level` small
- provide `OPENAI_API_KEY`

Mixed local/hosted run:

- agent `backend: ollama`
- world `backend: openai`
- updater slots can be OpenAI or Ollama independently

Modal H100 vLLM run:

- `models.shared_vlm.backend: vllm`
- `models.shared_vlm.model: Qwen/Qwen3.6-35B-A3B-FP8`
- agent, world, and active updater slots `backend: vllm`
- Modal starts the local vLLM server before invoking the runtime shell
- the sample Qwen3.6 FP8 server options use Triton GDN prefill and disable
  thinking in the vLLM chat template

## Failure Checks

The loader fails early for missing required fields, invalid debug values,
negative `action_history_window`, `experimental_memory_turn_buffer < 1`,
unknown backends, missing updater slots, real OpenAI/Ollama updater slots
without a `model`, or vLLM roles without a direct or inherited `model`.

Runtime can still fail if the game index is not in the catalog, credentials
are missing, Ollama is unavailable, a disabled tool is requested, the world
prediction role is enabled without a world model, provider
responses fail the description schema, or a Modal-managed vLLM server does not
become ready before the runtime shell starts.
