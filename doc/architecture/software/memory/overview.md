# Memory Overview

Memory is SQLite-backed and learner-oriented.

- `m_states`: committed frame turns with observation, chosen action, learner
  snapshot, learner trace, metrics, metadata, and timestamps.
- `learner_artifacts`: optional debug artifacts keyed by run/game/turn/kind.
- `model_input_debug_records`: passive debug request records.
- `run_metadata`: runtime startup and run-level facts.

Old disposable databases are reset instead of migrated.
