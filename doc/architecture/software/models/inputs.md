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
instruction/system content, with its mutable `C^X` appended there as
`AGENT_CONTEXT`. World `S` and goal `G` carry their role-local
`instructions/instruction_prompt*.md` files in provider instruction/system
content. Their user payload carries the composed mutable role context; world
also carries the committed action because `S` is action-conditioned. Updater
`P` mutates only `K` and `L`.

## Agent `X` Inputs

Agent `X` receives inputs only for controllable final frames.

- composed agent context `C^X`, carried in the provider instruction/system
  content rather than the user JSON payload
- world model game context `L^S`
- goal model game context `L^G`
- initial run observation as the history-anchor image
- current observation
- current action space
- bounded recent action history from prior frame turns

The world and goal contexts are the game-specific `L` documents only. Their
general `K` documents remain role-local and are not copied into `X`.

Provider adapters attach exactly two images in this semantic order:
`history_anchor`, then `current`. `history_anchor` is the initial run
observation. The immediately previous orchestration frame is not included in
`X` input.

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

World and goal receive resolved data directly from orchestration.

## Updater `P` Inputs

Updater inputs are built by orchestration from live in-turn objects. The
updater does not read SQLite or resolve memory references directly.

World game-context prompt updates receive:

- previous role context for `S`
- submitted real action or synthetic `NONE` action when relevant
- committed world post-decision description prediction
- `previous_observation_frame`, the observed frame before the transition
- `current_observation_frame`, the observed frame after the transition

The world prompt updater does not receive Agent X tool results, transition
timing, or score/progress metadata.

Goal game-context prompt updates receive:

- previous role context for `G`
- committed goal post-decision description prediction
- `previous_observation_frame`, the observed frame before the transition
- `current_observation_frame`, the observed frame after the transition

The goal prompt updater does not receive an action, Agent X tool results,
transition timing, or score/progress metadata.

Agent game context updates use `AgentGameContextUpdateInput`:

- previous agent context for `X`
- previous observed frame, `o_t`
- current observed frame after the last action, `o_t+1`
- current-turn world game context from the same pre-update context generation
  as the world updater input
- current-turn goal game context from the same pre-update context generation as
  the goal updater input
- previous-turn world game context when a prior source context exists
- full live `AgentTrace`
- action-progress `time_cost` and `score_delta` when available; compute timing
  such as `trace_cost` is not passed to the agent updater

`AgentTrace` remains the source of the last final action and reasoning summary.

End-of-run general context updates use `GeneralKnowledgeUpdateInput`:

- role: `world`, `goal`, or `agent`
- previous role context, including the final game context and current general
  context
- run id, game id, stop reason, step count, completed levels, final state, and
  state record ids
