# Step 03: Minimal Orchestration Loop

## Objective

Implement a near-empty orchestration shell and single-game runtime loop.

## Implementation

- Keep orchestration limited to random action selection from the current valid
  action list.
- Start the single configured game.
- On each playable turn:
  - receive incoming frames
  - receive current valid actions
  - randomly choose one action
  - apply that action
- On `GameState.GAME_OVER`, reset the environment and play again.
- When `levels_completed` increases while the state stays playable, treat that
  as level progress and reset the per-level action budget.
- On final `GameState.WIN`, stop the program.
- On action-budget exhaustion for the level, stop the program.

## Dependencies

Depends on Steps 01 and 02.

## Acceptance Check

- Runtime can execute the shell flow for one selected game.
- Action selection is random from the valid current actions only.
- Restart and stop behavior is unambiguous under real ARC `GameState`
  semantics.
