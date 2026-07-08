# Step 03: Runtime Shells

## Objective

Add empty component shells that name the main architecture roles while avoiding
premature interfaces.

## Implementation

- Add an environment adapter shell.
- Add an orchestrator shell.
- Add model adapter and registry shells.
- Add state memory `M` and experimental memory `E` shells.
- Add context document, tool router, updater, and runtime loop shells.
- Use docstrings to tie each shell back to the architecture docs.
- Do not add methods, schemas, validation, model calls, SQLite tables, or ARC integration logic.

## Dependencies

Depends on Step 02.

## Acceptance Check

- The source tree shows where each architecture role will live.
- Shell classes have no concrete behavior.
