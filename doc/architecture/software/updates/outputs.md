# Updater Outputs

Updater P returns updated agent context documents:

- agent game context `L^X` during frame turns
- agent general context `K^X` at end-of-run

Orchestration applies updater outputs to the live context and persists the
resulting state to SQLite.
