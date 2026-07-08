# Orchestration Inputs

- `RuntimeConfig`: run id, database path, optional deadline.
- `EnvironmentConfig`: game selection, limits, debug flags, and `agent:`
  learner config.
- `EnvironmentAdapter`: ARC environment boundary.
- `OnlineLearnerAgent`: decision/update boundary.
- `StateMemory`: optional SQLite persistence.
