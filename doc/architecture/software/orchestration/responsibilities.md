# Orchestration Responsibilities

- Own ARC reset/step calls.
- Own frame unrolling and animation-frame `NONE` decisions.
- Own runtime deadlines, action limits, level limits, and scorecard lifecycle.
- Validate learner actions before environment submission.
- Persist learner snapshots/traces through memory.
- Keep parallel workers isolated by game-specific runtime state and SQLite
  files.

Online model updates belong to `online`. SQLite implementation belongs to
`memory`.
