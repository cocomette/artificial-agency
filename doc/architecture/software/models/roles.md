# Model Roles

## Orchestrator Agent `X`

`X` is the decision-making agent. It receives the current agent context,
serialized observation text plus a cropped observation image, transition
evidence, recent action history, and current ARC action space. It returns one
final `ActionSpec` and an
`AgentTrace`.

The vLLM adapter sends OpenAI-compatible multimodal Chat Completions messages.
Observation payloads contain an `ObservationText` string generated from native
ARC grids plus a PNG data-URL image cropped to the same configured bounds.
Role instructions include the canonical ARC symbol color glossary.

ACTION6 history data uses original ARC grid coordinates. Model-facing guidance
and validation require new ACTION6 decisions to use visible cropped
coordinates, matching the active serialized crop in `ObservationText`
(`x/y=3..60` for the default `crop_cells=3`), plus a non-empty target
description.

### Tool Runtime Framework

The provider-neutral agent contract still has a controlled tool runtime shape,
but the current vLLM-only runtime does not wire real world or goal providers.
The starter configs keep `max_tool_calls: 0`.

`X` does not read memory, write SQLite, or call model adapters directly.
Orchestration builds the frame-turn input, calls `X`, validates the returned
action, and owns persistence.

Output:

- final `ActionSpec`
- full `AgentTrace`

## Transition Change Summary

The change model receives previous and current observations as `ObservationText`
strings plus cropped images for every serialized evidence frame in the current
call. It summarizes visible changes, returns structured change fields, and uses
cropped ARC-grid changed-cell counts for authoritative evidence.

For frame bundles and transition prompts, component-level deltas are generated
directly from adjacent serialized frames. Large retained animation bundles are
budgeted at the adapter boundary with balanced overlapping text chunks. There
is no object matching heuristic;
component IDs are frame-local labels, and omitted component sections suppress
component-ID delta lines.

When chunking produces multiple change-summary calls, a final reducer may
reconcile ordered partial summaries. The reducer sees deterministic
changed-cell metrics, action context, selected row-only keyframes drawn from
first/final/chunk-boundary frames, and cropped images for those selected
keyframes, then returns the same `summary` plus `change_detected` schema.
Reducer `change_detected` is validated against the full deterministic evidence,
and repair exhaustion falls back to the chronological deterministic merge.

Output:

- transition summary text
- changed-cell evidence and structured fields used by orchestration/updaters

## Agent Context Historizer

The historizer summarizes recent agent context revisions before the updater
builds the next context. It is a text-only vLLM role and does not receive
frames directly.

Output:

- structured summary of recent agent context evolution

## Updater `P`

`P` runs after observed transitions. In the frame-unrolled game loop, animation
frame transitions compare the current frame to the next buffered frame, while
controllable final-frame transitions compare the selected action against the
first frame returned by the next real environment step.

The implemented updater slots are:

| Slot | Updates |
| --- | --- |
| `agent` | Agent game context `L^X`. |
| `general` | Agent general context `K^X` at end of run. |

Agent game updater prompts include observation serialization plus a cropped
current-frame image where frame context is needed. Agent game updater
instructions use the active visible crop for future ACTION6 policy guidance.
The updater does not own persistence. Its
outputs return to orchestration, which applies them to live working context
documents and persists the resulting state into `M`.

Output:

- updated `L^X` during frame/game-loop updates
- updated `K^X` at end of run through the shared general updater task

## Model Adapter Rule

Adapters translate between role contracts and vLLM Chat Completions calls.
They do not own the runtime loop, environment stepping, or SQLite persistence.
They also do not read memory directly; memory access is mediated by
orchestration.

Provider-specific adapters live in `providers/` folders under each model role.
Shared vLLM transport utilities live in `models/providers/vllm.py`.
