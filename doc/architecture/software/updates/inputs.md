# Update Inputs

Updater `P` receives data only at orchestration-owned update boundaries. In a
simple step loop this is after a real environment step. In the frame-unrolled
loop this is after every observed frame decision. These in-loop calls target
role-specific game updater tasks for `L` only. At the end of a game loop/run,
orchestration invokes the shared general updater task three times, once for
each role-specific `K`.

Inputs are assembled by orchestration from live transition objects and memory
references. The updater does not read from SQLite directly.

Turn metrics are assembled by orchestration for persistence and updater
boundaries. Real environment steps include elapsed-step metadata, Agent X
decision duration, and `score_delta` when available. Animation-frame updates
may include frame-turn timing and usually no `score_delta` because they do not
spend a real environment step.

## World Prompt Updater Inputs

The world game-context prompt updater receives only:

- previous world context `L^S_i,t`
- submitted real action or synthetic `NONE` action for the turn
- committed world post-decision `predicted_description`
- `previous_observation_frame`, the observed frame before the transition
- `current_observation_frame`, the observed frame after the transition

It does not receive Agent X live tool results, transition timing, or
score/progress metadata.

## Goal Prompt Updater Inputs

The goal game-context prompt updater receives only:

- previous goal context `L^G_i,t`
- committed goal post-decision `predicted_description`
- `previous_observation_frame`, the observed frame before the transition
- `current_observation_frame`, the observed frame after the transition

It does not receive the selected action, Agent X live tool results,
transition timing, or score/progress metadata.

## Agent Context Update Inputs

- previous agent context `L^X_i,t`
- previous observed frame, `o_t`
- current observed frame after the last action, `o_t+1`
- full live agent trace
- tool calls and tool results carried by the trace
- committed S/G post-decision area arrays, limited to `bbox_2d` coordinate
  arrays and `description`
- transition timing and `score_delta` when available

## General Context Update Inputs

- target role: world `S`, goal `G`, or agent `X`
- previous role context, including final game context `L^m` and current
  general context `K^m`
- run id and game id
- stop reason, step count, completed levels, and final state
- persisted state record ids available for replay
- terminal metadata assembled by orchestration
