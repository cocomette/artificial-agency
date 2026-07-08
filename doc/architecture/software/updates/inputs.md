# Update Inputs

Updater `P` receives data only at orchestration-owned update boundaries. In a
simple step loop this is after a real environment step. In the frame-unrolled
loop this is after every observed frame decision. These in-loop calls target
role-specific game updater tasks for `L` only. At the end of a game loop/run,
orchestration invokes the shared general updater task twice, once for world
`K^S` and once for agent `K^X`.

Inputs are assembled by orchestration from live transition objects and memory
references. The updater does not read from SQLite directly.

Turn metrics are assembled by orchestration for persistence and updater
boundaries. Real environment steps include elapsed-step metadata, Agent X
decision duration, and `cumulative_score` when available. Agent X prompt
updater input also carries the previous agent game-context word count as a
compactness reward to minimize. Animation-frame updates may include frame-turn
timing and no new real action, but still carry `cumulative_score` when the
observed frame exposes progress metadata.

## World Prompt Updater Inputs

The world game-context prompt updater receives only:

- previous world context `L^S_i,t`
- submitted real action or synthetic `NONE` action for the turn
- committed world post-decision `predicted_description`
- `current_observation_frame`, the observed frame after the action/frame turn

It does not receive Agent X live tool results, transition timing, or
score/progress metadata.

## Dormant Goal Prompt Updater Inputs

The goal game-context prompt updater remains available as a standalone model
contract, but the normal runtime loop does not call it. Its direct adapter
contract receives only:

- previous goal context `L^G_i,t`
- committed goal post-decision `predicted_description`
- `current_observation_frame`, the observed frame after the action/frame turn

It does not receive the selected action, Agent X live tool results,
transition timing, or score/progress metadata.

## Agent Context Update Inputs

- previous agent context `L^X_i,t`
- previous observed frame, `o_t`
- current observed frame after the last action, `o_t+1`
- current-turn world game context
- previous-turn world game context when available
- compact action history: bounded prior actions plus the submitted real action
  or synthetic `NONE` that produced the current frame
- action-progress `time_cost`, `cumulative_score`, and
  `agent_context_word_count` when available

## General Context Update Inputs

- target role: world `S` or agent `X`
- previous role context, including final game context `L^m` and current
  general context `K^m`
- run id and game id
- stop reason, step count, completed levels, and final state
- persisted state record ids available for replay
- terminal metadata assembled by orchestration
