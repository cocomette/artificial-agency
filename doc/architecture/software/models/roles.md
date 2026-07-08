# Model Roles

## Agent X

Agent X chooses the final action on controllable frame turns. Orchestration
builds its input from the current frame, allowed actions, recent action
history, active agent context, and game memory.

Output:

- final `ActionSpec`
- `AgentTrace`

If Agent X fails with a model/provider/output error, orchestration chooses a
deterministic legal fallback action. Non-model errors still raise.

## Change Summary

The change-summary role summarizes visible transition evidence after a frame
turn resolves. It receives oldest-to-newest transition frames, the submitted or
synthetic action, action glossary data, and previous change elements.

Output:

- structured `ChangeSummaryElement` values
- deterministic changed-pixel metadata

If the model fails with a model/provider/output error, orchestration records a
deterministic fallback summary based on visible frame differences.

## Historizer

The historizer summarizes how prior agent game-context fields evolved. It runs
only when enough prior complete agent game contexts are available.

Output:

- field evolution for `goals`, `game_mechanics`, `policy`, `history`, and
  `extras`

If the model fails with a model/provider/output error, orchestration uses the
explicit `not_available` summary with fallback metadata.

## Game Memory

The memory role produces structured provider output for a compact same-run game
memory document. Providers return JSON with one top-level `memory` string.

Orchestration calls the memory role after controllable real-action turns, after
change summary and before updater P. The input contains same-run action history
through the latest real action, the first game frame, the latest
post-transition frame, and non-identifying metadata.

Output:

- validated `{"memory": "<string>"}` provider output
- parsed prompt-facing game memory text

If the model fails with a model/provider/output error, orchestration keeps the
previous memory document and marks memory as not updated this turn.

## Updater P

Updater P runs after observed transitions and at end-of-run.

Current slots:

- `agent`: updates agent game context `L^X`
- `general`: updates agent general context `K^X`

The agent game updater receives the active game memory, action history,
context-history summary, progress feedback, and current observation evidence.
The general updater receives run-level stop metadata.

If an updater fails with a model/provider/output error, orchestration preserves
the previous context. Non-model errors still raise.
