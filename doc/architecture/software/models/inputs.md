# Model Inputs

## Shared Context Inputs

Each prompt-backed model role receives immutable role instructions plus a
composed mutable role context:

```text
C^m_i,t = K^m + L^m_i,t
```

Where:

- `K^m` is game-agnostic role context
- `L^m_i,t` is game-specific role context
- `m` is one of `S`, `G`, or `X`

The immutable role instruction is not part of `C`. Agent `X` receives
`models/orchestrator_agent/instructions/system_prompt.md` as the provider
instruction/system content, while its mutable `C^X` is carried in the user
prompt. World `S` carries its role-local `instructions/instruction_prompt*.md`
files in provider instruction/system content. Its user payload carries the
composed mutable role context and the committed action because `S` is
action-conditioned. Goal `G` still has model files and context contracts, but
it is dormant in the normal runtime loop. Updater `P` mutates only the active
`K` and `L` contexts.

## Agent `X` Inputs

Agent `X` receives inputs only for controllable final frames.

- composed agent context `C^X`, carried in the user prompt
- history-anchor observation for the bounded recent action history
- current observation
- current action space
- bounded recent action history from prior frame turns

World game context is not a direct Agent `X` input. The active agent
game-context updater summarizes relevant world context into `L^X`, which is
then visible through composed agent context `C^X`. Goal context is dormant and
is not fed to `X` or the agent updater.

Provider adapters attach exactly two images in this semantic order:
`history_anchor`, then `current`. `history_anchor` is the observation frame
that precedes the oldest action in the bounded recent action history. When no
recent action history is visible, it is the initial run observation.

The recent action history is compact action memory, not frame history. It
carries only the prior action and whether that frame turn was controllable.
Internal state ids, turn ids, frame refs, frame indexes, control reasons, and
reasoning summaries stay out of the model-facing history DTO.

Recent action history is passed as a normal Agent X input.

Inside the model layer, provider adapters may translate this turn into
provider-specific messages or requests. OpenAI carries the immutable Agent X
instructions through top-level `instructions`; Ollama carries them as the first
`system` message. Those internal DTOs are not orchestration inputs.

## World Prediction Model `S` Inputs

- composed world context `C^S`
- committed `ActionSpec`
- current `Observation`, supplied by orchestration

## Goal Prediction Model `G` Inputs

- composed goal context `C^G`
- current `Observation`, supplied by orchestration

These inputs describe the dormant goal adapter contract. The normal runtime
loop does not call `G`.

## Updater `P` Inputs

Updater inputs are built by orchestration from live in-turn objects. The
updater does not read SQLite or resolve memory references directly.

World game-context prompt updates receive:

- previous role context for `S`
- submitted real action or synthetic `NONE` action when relevant
- committed world post-decision description prediction
- `current_observation_frame`, the observed frame after the action/frame turn

The world prompt updater does not receive Agent X tool results, transition
timing, or score/progress metadata.

Goal game-context prompt updates are dormant in the normal runtime loop. Their
standalone adapter contract receives:

- previous role context for `G`
- committed goal post-decision description prediction
- `current_observation_frame`, the observed frame after the action/frame turn

The goal prompt updater does not receive an action, Agent X tool results,
transition timing, or score/progress metadata.

Agent game context updates use `AgentGameContextUpdateInput`:

- previous agent context for `X`
- previous observed frame, `o_t`
- current observed frame after the last action, `o_t+1`
- current-turn world game context from the same pre-update context generation
  as the world updater input
- previous-turn world game context when a prior source context exists
- compact action history: bounded prior actions plus the submitted real action
  or synthetic `NONE` that produced the current frame
- action-progress `time_cost`, `cumulative_score`, and
  `agent_context_word_count`; compute timing such as `trace_cost` is not
  passed to the agent updater

The full `AgentTrace` remains a persistence and debug artifact, not a
model-facing agent updater prompt field.

End-of-run general context updates use `GeneralKnowledgeUpdateInput` for the
active world and agent roles:

- role: `world` or `agent`
- previous role context, including the final game context and current general
  context
- run id, game id, stop reason, step count, completed levels, final state, and
  state record ids
