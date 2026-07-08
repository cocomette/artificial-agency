# Software Architecture Overview

The active software architecture is a small online learner around ARC
environment interaction and SQLite memory.

## Modules

- `environment`: ARC adapter, game selection, config loading, visualization.
- `online`: frozen Transformers backbone, online learner state, replay, and
  planner.
- `orchestration`: frame-unrolled game loop and lifecycle handling.
- `memory`: SQLite persistence for learner turns and artifacts.
- `runtime`: local shell, Kaggle entrypoint, parallel worker isolation.
- `debug`: typed debug events and terminal/dashboard inspection.

## Turn Flow

1. Orchestration unrolls observation bundles into frame turns.
2. Controllable frames are encoded by the frozen backbone and planned by the
   learner. Animation frames synthesize `NONE`.
3. Real actions step the ARC environment.
4. The observed transition is added to the prioritized buffer.
5. Bounded local update and replay run under per-turn budgets.
6. SQLite stores the learner trace, learner snapshot, metrics, and metadata.

Legacy prompt-role model providers and context update paths are not part of the
active architecture.
