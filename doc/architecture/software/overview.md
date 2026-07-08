# Software Architecture Overview

This folder describes the active ARC-AGI-3 runtime architecture. The running
program is coordinated by orchestration, which owns environment interaction,
model calls, context updates, and SQLite persistence.

## Active Runtime Shape

Active model roles:

- Change summary converts observed frame transitions into compact action
  history evidence.
- World model updates the current world description, special-event memory, and
  per-action effect summaries.
- Agent-context historizer summarizes recent same-run agent context evolution
  and tried strategy outcomes.
- Agent creator optionally reviews completed game states and maintains a
  separate learned-role SQLite database.
- Updater P revises agent probing or policy game context during a
  run, returns the next controllable action, and updates agent general context
  at run end.

Agent X adapter code remains present but is dormant in the current runtime game
loop.

The durable state database stores agent context only. Agent creator role
revisions, tool-call runs, and pending game-review requests live in a separate
SQLite store when creator models are configured. Created roles are retained in
that database only; they are not proposed to the historizer and are not used by
the runtime game loop. Older run databases with the previous wider `m_states`
table are intentionally incompatible and should be reset before running this
branch.

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
   summary role using prior world-model context and prior element output as
   non-authoritative focus context, reconstruct action-history summary text
   from the returned elements, and locally mark identical direct transitions.
4. Clear any remaining queued updater actions when the latest real action
   produced zero net first-to-last visible frame change.
5. When fresh context is needed, update world-model context from transition
   evidence, summarize recent agent context history, select probing/policy
   mode, apply the deterministic probing cap, and run the selected updater P
   agent game task for that actionable state.
6. Submit the updater-selected action on controllable frames.
7. Persist the completed turn, decision trace, transition metrics, and agent
   context into `M`.
8. If the agent creator sidecar is configured, enqueue a lightweight game
   review after the completed `M` row is durable.
9. At run end, run updater P for agent general context.
