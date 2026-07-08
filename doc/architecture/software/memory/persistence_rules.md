# Persistence Rules

- Orchestration prewrites an `m_states` row before a frame decision when state
  memory is enabled.
- Orchestration completes that row after the observed transition and learner
  update are available.
- Incomplete source rows are visible only through source-read helpers; normal
  list/latest reads return complete rows.
- Schema mismatches fail fast with instructions to reset the disposable DB.
