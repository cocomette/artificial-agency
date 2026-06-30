# Orchestration Game Loop

This folder defines the high-level ARC-AGI game loop owned by orchestration.

Orchestration owns deterministic execution and side effects. The orchestrator
agent model `X` owns decision selection.

Random selection, when used by the default `X` adapter, lives behind the `X`
model role and not directly in orchestration.

## Documents

- [`concepts.md`](concepts.md): frame bundles, frame unrolling, controllable
  frames, and synthetic `NONE`.
- [`state_machine.md`](state_machine.md): state-by-state loop behavior and
  invariants.
- [`diagrams.md`](diagrams.md): sequence diagram and state-machine block
  diagram.
- [`interfaces.md`](interfaces.md): architecture-level contract sketches.
- [`test_scenarios.md`](test_scenarios.md): observable game-loop behavior.

## Scope

The source implementation lives in
`src/face_of_agi/orchestration/game_loop/state_machine.py` as
`GameLoopStateMachine`, which is called by the top-level `Orchestrator`. The
state machine unrolls frame bundles, validates synthetic `NONE`, coordinates
the updater boundary, and records frame-turn state when memory is wired.

Runtime remains a bootstrap/delegation layer and should not take ownership of
game-step logic again.
