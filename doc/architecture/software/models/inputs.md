# Model Inputs

## Shared Context Inputs

Each prompt-backed model role receives immutable role instructions plus the
role-specific mutable context assembled by orchestration.

Agent context uses:

```text
C^X_i,t = K^X + L^X_i,t
```

Where:

- `K^X` is game-agnostic agent context
- `L^X_i,t` is game-specific agent context

The immutable instruction is not part of `C`. Agent `X` receives
`models/orchestrator_agent/instructions/system_prompt.md` separately from its
mutable `role_context`. Updater `P` mutates only agent `K` and `L` in the
current implementation.

## Observation Inputs

Model-facing observations combine `ObservationText` strings with cropped PNG
image attachments for implemented frame-consuming vLLM roles. The serializer
accepts native 2D ARC integer grids, crops to original coordinates
`x=3..60` and `y=3..60`, optionally prints cropped rows with `0..63`
coordinate labels and uppercase ARC symbols `0..F`, lists 4-connected
same-symbol components unless disabled by config or overflow applies, falls
back from exact component `runs=` row spans to compact component fields when
needed, and emits component-level deltas for bundles/change prompts. When
components are omitted, deltas keep only changed-cell counts. Agent, updater,
and change-summary user prompts use compact observation headers: the label is
followed directly by configured frame evidence; observation id, step,
crop-bound, coordinate-system, and symbol metadata remain internal serializer
metadata rather than repeated prompt text.

Observation-facing role instructions include a static ARC symbol color
glossary. Serialized rows and coordinates remain authoritative when text and
image appearance seem to conflict. The vLLM adapters attach PNG data URLs as
OpenAI-compatible `image_url` content parts for Agent X, change summary, change
reducer keyframes, and the agent-game updater.

## Agent `X` Inputs

Agent `X` receives inputs only for controllable final frames.

- composed agent context `C^X`
- current observation as `ObservationText` plus one matching cropped image
- current action space
- bounded recent action history from prior frame turns
- deterministic action outcome evidence, including cropped changed-cell counts
- `AgentToolRuntime`, currently with no available tools

The recent action history is compact action memory, not frame history. The
model-facing prompt contains only the bounded list of prior action payloads and
summaries.

New ACTION6 outputs are validated against the active visible crop range derived
from `ObservationTextConfig.crop_cells`; with the default crop this is
`x/y=3..60`. Historical ACTION6 rows remain rendered as recorded original-grid
coordinates.

Agent traces keep first/current observation references, and persisted frame-turn
state keeps previous-frame references for orchestration and memory inspection,
but the current Agent `X` prompt carries only the current observation evidence.

## Change Summary Inputs

The change summary model receives:

- previous observation as `ObservationText` plus matching cropped image
- current observation as `ObservationText` plus matching cropped image
- a bounded first/intermediate/final sample of retained animation frames when a
  filtered frame bundle is larger than the change-summary evidence-frame budget
- component-level deltas for the transition or frame bundle
- submitted real action or synthetic `NONE`
- glossary/action-space information when available

It uses cropped-frame changed counts for model-facing evidence and no-change
short-circuiting.

## Historizer Inputs

The historizer receives recent agent context revisions selected by
orchestration and the current agent context fields. It is text-only and does
not receive frames directly.

## Updater `P` Inputs

Updater inputs are built by orchestration from live in-turn objects. The
updater does not read SQLite or resolve memory references directly.

Agent game context updates use `AgentGameContextUpdateInput`:

- previous agent context for `X`
- current observed frame after the last action, `o_t+1`, as `ObservationText`
  plus one matching cropped image
- full live `AgentTrace`
- transition change summary
- bounded prior action history
- agent context history summary
- reward/update quantities and turn metrics

End-of-run general context updates use `GeneralKnowledgeUpdateInput`:

- target role: `agent`
- previous agent context, including the final game context and current general
  context
- run id, game id, stop reason, step count, completed levels, final state, and
  state record ids
