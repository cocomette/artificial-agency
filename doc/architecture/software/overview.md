# Software Architecture Overview

This folder describes the active ARC-AGI-3 runtime architecture. The running
program is coordinated by orchestration, which owns environment interaction,
model calls, context updates, and SQLite persistence.

## Active Runtime Shape

Active model roles:

- Change summary converts observed frame transitions into compact action
  history evidence.
- Compacter updates the current world model and rolling action/strategy
  summaries, and stores solved-level compact summaries when completion is
  observed.
- Updater P revises compact agent game strategy context during a run and
  returns the next controllable action chain.

Agent X adapter code remains present but is dormant in the current runtime game
loop.

The durable state database stores agent context only. Older run databases with
the previous wider `m_states` table are intentionally incompatible and should be
reset before running this branch.

## Ownership Rule

Only orchestration coordinates cross-module side effects during a game step.
Models do not read or write persistence directly. The runtime module starts the
program and assembles dependencies, but the game loop remains owned by
orchestration.

## Frame Turn Flow

1. Read the current observation and action space from the environment.
2. Prewrite the current source row in `M` when state memory is enabled.
3. For non-initial turns, compute visible changed-pixel percentage; summarize
   direct changed transitions or bundled animation arrays with the change
   summary role using prior compacter context and prior element output as
   non-authoritative focus context, reconstruct action-history summary text
   from the returned elements, and locally mark identical direct transitions
   while carrying forward the latest element names and descriptions for
   prompt context.
4. Clear any remaining queued updater actions when the latest real action
   produced zero net first-to-last visible frame change.
5. When fresh context is needed, run the compacter and run updater P for that
   actionable state with the current raw action and strategy windows plus
   compact summaries. On a level-completion turn, the compacter sees the latest
   retained frame from the solved level instead of the new-level frame.
6. If updater P selects an action already observed from the exact same
   current-frame hash, orchestration may enter the known-state simulation
   sidecar. The sidecar replays real historical transition evidence, persists
   simulated rows marked `simulated`, refreshes compacter context from the
   growing simulated history before each simulated updater decision, submits a shortest
   historical catch-up path immediately without model calls, and queues only
   the first unknown exit action for normal processing.
7. Submit the updater-selected or queued exit action on controllable frames.
8. Persist the completed turn, decision trace, transition metrics, and agent
   context into `M`.
