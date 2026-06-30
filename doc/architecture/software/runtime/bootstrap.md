# Runtime Bootstrap

Runtime bootstrap is responsible for assembling the process before a game run
starts.

## Bootstrap Steps

1. Load runtime and environment configuration.
2. Resolve the selected ARC game.
3. Initialize SQLite-backed memory domains.
4. Register model adapters in the model registry.
5. Initialize role context documents.
6. Construct the orchestration layer with all dependencies.
7. Start the orchestration-owned game loop.

## Boundary Rule

Runtime can decide how the process starts. It should not decide how each game
step proceeds once orchestration has started.
