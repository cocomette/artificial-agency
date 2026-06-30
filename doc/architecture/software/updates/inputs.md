# Update Inputs

Updater `P` receives data only at orchestration-owned update boundaries. In the
frame-unrolled loop this is after every observed frame decision. These in-loop
calls target the agent game updater task for `L^X`. At the end of a run,
orchestration invokes the agent general updater task for `K^X`.

Inputs are assembled by orchestration from live transition objects and memory
references. The updater does not read from SQLite directly.

Reward/update quantities are computed by deterministic runtime logic before
the updater is called. They are component signals rather than one scalar
reward. Model-facing transition evidence uses cropped ARC-grid changed-cell
counts.

## Agent Context Update Inputs

- previous agent context `L^X_i,t`
- previous observed frame, `o_t`, serialized as `ObservationText`
- current observed frame after the last action, `o_t+1`, serialized as
  `ObservationText`
- full live agent trace
- submitted real action or synthetic `NONE`
- transition change summary
- bounded prior action history
- agent context history summary
- reward/update quantities and turn metrics
- score/progress markers when available

## General Context Update Inputs

- target role: agent `X`
- previous agent context, including final game context `L^X` and current
  general context `K^X`
- run id and game id
- stop reason, step count, completed levels, and final state
- persisted state record ids available for replay
- terminal metadata assembled by orchestration
