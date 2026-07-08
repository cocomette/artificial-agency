# Update Outputs

Updater `P` returns revised game-specific context documents.

## Output Contexts

- `L^S_i,t+1`: revised world-model game context
- `L^G_i,t+1`: revised goal-model game context
- `L^X_i,t+1`: revised agent game context

## Persistence Rule

Updater outputs go back to orchestration. Orchestration applies them to the
live working `ContextDocuments`, persists the resulting contexts into `M`, and
uses them when composing the next model calls.

The updater does not own a separate memory store and does not write directly
to SQLite.

## Scope Rule

Game-specific contexts may be updated during a game. Game-agnostic contexts
`K^m` are updated only after finishing a game, if that behavior is enabled.
