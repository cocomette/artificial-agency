# Orchestration Outputs

The game loop emits and persists:

- Agent X decisions and traces
- candidate action proposals
- World predictions for candidate actions
- environment step results
- change summaries and action-history entries
- Reward Judge scores
- reward records, including proxy learning-progress
- Memory documents
- Goal predictions
- completed `M` state rows
- debug events and model-input records for runtime inspection
