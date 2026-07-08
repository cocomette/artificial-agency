# Orchestration Outputs

The game loop emits and persists:

- decision traces wrapping updater-selected actions
- environment step results
- change summaries and action-history entries
- updated agent context
- completed `M` state rows
- debug events for runtime inspection

End-of-run orchestration also applies the general updater to the agent context.
