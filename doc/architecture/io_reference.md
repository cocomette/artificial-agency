# Model I/O Reference

This page is the living implementation reference for current model-role inputs
and outputs. The real backend is vLLM through an OpenAI-compatible Chat
Completions endpoint. The `openai` Python package is only the HTTP client used
to reach that vLLM endpoint.

The implemented model roles are:

- orchestrator agent `X`
- transition change summary model
- agent context historizer
- updater `P`, split into agent game-context and agent general-context tasks

OpenAI, Ollama, HuggingFace, Diffusers, world-provider, and goal-provider paths
are not part of the current implementation. The implemented vLLM roles that
consume ARC frames use OpenAI-compatible multimodal Chat Completions content
parts with PNG data URLs.

## Source Files

Primary contracts:

- `src/face_of_agi/contracts.py`
- `src/face_of_agi/models/orchestrator_agent/contracts.py`
- `src/face_of_agi/models/change/contracts.py`
- `src/face_of_agi/models/historizer/contracts.py`
- `src/face_of_agi/models/updater/contracts.py`
- `src/face_of_agi/models/observation_text.py`
- `src/face_of_agi/models/image_inputs.py`
- `src/face_of_agi/models/color_glossary.py`

Prompt and instruction assembly:

- `src/face_of_agi/models/orchestrator_agent/tooling.py`
- `src/face_of_agi/models/orchestrator_agent/providers/vllm.py`
- `src/face_of_agi/models/change/adapter.py`
- `src/face_of_agi/models/change/providers/vllm.py`
- `src/face_of_agi/models/historizer/adapter.py`
- `src/face_of_agi/models/historizer/providers/vllm.py`
- `src/face_of_agi/models/updater/adapter.py`
- `src/face_of_agi/models/updater/providers/vllm.py`
- `src/face_of_agi/models/providers/vllm.py`

Human-editable instruction files:

- `src/face_of_agi/models/orchestrator_agent/instructions/system_prompt.md`
- `src/face_of_agi/models/change/instructions/instruction_prompt.md`
- `src/face_of_agi/models/change/instructions/reducer_instruction_prompt.md`
- `src/face_of_agi/models/historizer/instructions/instruction_prompt.md`
- `src/face_of_agi/models/updater/instructions/agent_game_context_updater_prompt.md`
- `src/face_of_agi/models/updater/instructions/agent_general_context_updater_prompt.md`

## Shared Data Shapes

`RoleContext` is the mutable context document passed to model roles:

- `general`: game-agnostic context `K^m`
- `game`: game-specific context `L^m`
- `composed()`: `K^m + L^m`, joined with a blank line when both parts exist

`ActionSpec` is one candidate, proposed, or submitted action:

- `action_id`: ARC `GameAction` or string action id
- `data`: optional structured payload for complex actions
- `name`: stable display string derived from `action_id`
- `NONE`: internal synthetic action for non-controllable animation frames

`Observation` is a resolved observed frame or frame bundle:

- `id`
- `step`
- `frame`
- `frames`
- `raw_frame_data`
- `metadata`
- `frame_count()`

`ObservationRef` is the reusable reference form:

- `memory`: `state` for persistent `M`, or `experimental` for rolling `E`
- `id`: record or observation id inside that memory domain

`AgentTrace` records a model decision:

- final action
- reasoning summary
- any provider-neutral tool call/result records
- provider metadata

Tool-call contracts remain provider-neutral, but the current vLLM runtime does
not wire real world or goal providers. Starter configs keep `max_tool_calls: 0`.

## ObservationText

Model-facing frame evidence uses `ObservationText` for exact symbols,
coordinates, components, and deterministic counts. For frame-consuming vLLM
roles, the same crop is also rendered as cropped PNG image content parts.
Serialized rows remain authoritative when text and image appearance conflict.

`ObservationTextConfig`:

- `crop_cells`: defaults to `3`
- `overflow_chars_per_frame`: defaults to `12000`
- `include_rows`: defaults to `true`; when `false`, observation text omits
  serialized char-level cropped rows
- `include_components`: defaults to `true`; when `false`, observation text
  omits component listings and component-ID delta lines
- `include_component_runs`: defaults to `true`; when `false`, component
  listings keep ids, symbols, area, bbox, and centroid but omit exact `runs=`
  row-span geometry
- `compact_components`: defaults to `false`; when `true`, component listings
  group same-symbol, same-shape components as compact `symbol`, `size`, `nb`,
  and `box` lines, and component deltas keep only changed-cell counts

Serializer constraints:

- input must be a native 2D ARC integer grid
- frame dimensions must be `64x64`
- symbols must be integers in `0..15`
- booleans, floats, non-2D arrays, and non-ARC symbols are rejected

Serialized content:

- crop bounds `x=3..60`, `y=3..60`
- original `0..63` coordinate labels
- cropped rows with uppercase hex ARC symbols `0..F` when `include_rows` is
  enabled
- all 4-connected same-symbol components unless overflow or
  `include_components: false` omits the component list
- exact component `runs=` row spans when enabled and under budget; otherwise
  compact components without `runs=` are tried before full component omission
- component-level deltas for frame bundles and change prompts; when components
  are omitted or compacted, deltas keep only changed-cell counts

`cropped_changed_cell_count()` is the model-facing changed-cell evidence used
by transition prompts, action history, and no-change suppression.

## Observation Images

`models/image_inputs.py` renders ARC frames through the shared frame renderer and
derives the image crop from `ObservationTextConfig.crop_cells`. With the default
crop, attached images cover the same original-grid bounds as serialized rows:
`x=3..60` and `y=3..60`.

Frame-consuming vLLM roles attach image content as OpenAI-compatible Chat
Completions parts:

- content order is text first, then one or more `image_url` parts
- image URLs are PNG data URLs by default
- `input_image_size` defaults to `2048x2048` in runtime configs
- `input_image_detail`, `input_image_resample`, `image_mime_type`, and
  `frame_scale` control request image rendering

The canonical prompt glossary is generated by `models/color_glossary.py` from
the ARC renderer palette symbols and uses color names only:

`0 white`, `1 light gray`, `2 gray`, `3 dark gray`, `4 charcoal`, `5 black`,
`6 magenta`, `7 pink`, `8 red`, `9 blue`, `A light cyan`, `B yellow`,
`C orange`, `D dark red`, `E green`, and `F purple`.

## Orchestrator Agent `X`

Contract:

```python
OrchestratorAgentModel.decide(
    context: RoleContext,
    current_observation: Observation,
    action_space: Sequence[ActionSpec],
    tool_runtime: AgentToolRuntime | None = None,
) -> DecisionResult
```

Prompt-backed vLLM input:

- immutable system instructions from
  `models/orchestrator_agent/instructions/system_prompt.md`
- `role_context`: `context.composed()`
- `current_observation`: `ObservationText` plus one matching cropped image
- allowed action list
- recent action history
- action outcome evidence based on cropped changed-cell counts; simple actions
  may be omitted after repeated zero-change attempts, while ACTION6 remains
  allowed and only exact repeated `x,y` coordinates are named as suppressed
- tool policy metadata, with no real tools in the current starter configs

First/current observation references remain in `AgentTrace` and persisted
frame-turn state, and previous-frame references remain available to
orchestration/memory. They are not serialized into the current Agent `X` prompt.

Output:

- `DecisionResult.final_action`: one validated `ActionSpec`
- `DecisionResult.trace`: `AgentTrace`

ACTION6 `data.x` and `data.y` use visible cropped ARC grid coordinates,
matching serialized rows and component bounding boxes. New Agent X ACTION6
outputs must also include a non-empty top-level `target` string. Historical and
allowed-action placeholder `ActionSpec` values may omit `target`.

## Transition Change Summary

Contract:

```python
ChangeSummaryModel.summarize_transition(
    previous_observation: Observation,
    current_observation: Observation,
    action: ActionSpec | None = None,
) -> ChangeSummaryResult
```

Prompt-backed vLLM input:

- immutable instructions from `models/change/instructions/instruction_prompt.md`
- previous observation serialized as `ObservationText` plus matching cropped image
- current observation serialized as `ObservationText` plus matching cropped image
- component-level deltas for the transition
- optional submitted or synthetic action
- cropped changed-cell count
- deterministic `change_detected` evidence across all retained adjacent frames

Output:

- `summary`
- `change_detected`
- first-to-final cropped changed-cell count and visible-crop percentage
- provider metadata

When no retained adjacent frame pair changes in the cropped visible area, the
change model is skipped and deterministic no-change text is used. Longer
retained bundles are split by `models.change.max_frames_per_call` into balanced
overlapping text chunks. If more than one chunk is produced and
`models.change.reduce_chunk_summaries` is true, a final reducer call receives
ordered partial summaries, full deterministic changed-cell metrics, action
context, selected row-only keyframes, and cropped images for those selected
keyframes. The keyframes are first frame, final frame, and chunk-boundary overlap
frames capped by
`models.change.reducer_keyframe_limit`. The reducer output uses the same
`summary` plus `change_detected` JSON schema and is validated against the full
deterministic evidence. If reducer repair fails, orchestration falls back to the
deterministic chronological merge of chunk summaries.

## Agent Context Historizer

Contract:

```python
AgentContextHistorizerModel.summarize_history(
    history_input: AgentContextHistoryInput,
) -> AgentContextHistorySummary
```

Prompt-backed vLLM input:

- immutable instructions from `models/historizer/instructions/instruction_prompt.md`
- recent agent context revisions selected by orchestration
- current agent context fields

Output:

- structured summary over the fields `goals`, `game_mechanics`, `policy`,
  `history`, and `extras`
- provider metadata

The historizer is text-only and does not receive frames directly.

## Updater `P`

The updater does not read SQLite, resolve memory refs, or persist contexts
directly. Orchestration chooses the updater task and applies the returned
context.

Implemented task slots:

| Slot | Updates |
| --- | --- |
| `agent` | Agent game context `L^X`. |
| `general` | Agent general context `K^X` at end of run. |

Prompt-backed updaters use `PromptUpdateRequest`:

- `target`: role, segment, task, and previous context
- `instructions`: human-editable instruction text
- `text`: final text payload sent to vLLM
- `images`: prompt images for agent game-context updates only
- `metadata`: backend/model metadata

Agent game-context updater requests include the current observation as
`ObservationText` plus one cropped image. General updater requests remain
text-only.

Real backend responses are parsed with the same output contract:

```json
{"updated_context": "complete revised context text"}
```

Updater return behavior:

- agent game task replaces `RoleContext.game` and preserves
  `RoleContext.general`
- general task replaces `RoleContext.general` and preserves `RoleContext.game`

## Debug Capture

Debug model-input capture stores the full final prompt request and metadata for
vLLM-backed roles. SQLite-backed model-input debug records keep raw image data
URLs so replay and inspection can see the exact provider payload. Terminal
`debug_trace: model_inputs` output is sanitized and summarizes data URLs instead
of printing full base64 payloads.

Sensitive-looking keys such as API keys, authorization headers, cookies, and
tokens are redacted by the debug sanitizer.
