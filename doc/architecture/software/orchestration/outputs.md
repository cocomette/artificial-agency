# Orchestration Outputs

The game loop emits and persists:

- Agent X decisions and traces
- candidate action proposals
- World predictions for candidate actions
- environment step results
- change summaries and action-history entries
- Reward Judge scores
- reward records
- Memory documents
- Goal predictions
- replay samples for online LoRA
- LoRA update attempts
- completed `M` state rows
- debug events for runtime inspection
