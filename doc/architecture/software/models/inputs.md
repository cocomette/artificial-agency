# Model Inputs

## Shared Context Inputs

Each model role receives a composed role context:

```text
C^m_i,t = K^m + L^m_i,t
```

Where:

- `K^m` is game-agnostic role context
- `L^m_i,t` is game-specific role context
- `m` is one of `S`, `G`, or `X`

## Agent `X` Inputs

- composed agent context `C^X`
- first observation for the game
- current observation
- current action space
- `AgentToolRuntime`, a controlled per-turn tool interface exposed by
  orchestration
- memory reference ids returned from prior tool calls or exposed from `M`

`X` uses `AgentToolRuntime.invoke(ToolCall)` to request world or goal tools.
The runtime returns the tool result and an experimental observation reference;
it does not expose SQLite, raw model adapters, or direct memory writes.

Inside the model layer, provider adapters may translate this turn into
provider-specific messages or requests. Those internal DTOs are not
orchestration inputs.

## World Tool `S` Inputs

- composed world context `C^S`
- candidate `ActionSpec`
- resolved source `Observation` or prediction, supplied by orchestration from a
  memory reference

## Goal Tool `G` Inputs

- composed goal context `C^G`
- resolved source `Observation` or prediction, supplied by orchestration from a
  memory reference

Model tools receive resolved data only after orchestration has looked up the
reference. The agent-facing tool input remains the reference id.

## Updater `P` Inputs

Updater inputs are built by orchestration from live in-turn objects. The
updater does not read SQLite or resolve memory references directly.

World and goal context updates share `ToolContextUpdateInput`:

- role: `world` or `goal`
- previous role context for `S` or `G`
- current observation reference
- actual next observation reference when available
- committed post-decision predictions when available
- matching world or goal `ToolResult` values from the live `AgentTrace`
- submitted real action or synthetic `NONE` action when relevant
- reward/update quantities
- transition metadata

Agent context updates use `AgentContextUpdateInput`:

- previous agent context for `X`
- current observation reference
- actual next observation reference when available
- full live `AgentTrace`
- committed post-decision prediction packet
- submitted real action or synthetic `NONE` action when relevant
- reward/update quantities
- transition metadata
