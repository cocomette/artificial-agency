# Update Outputs

There are no online update outputs in this branch. The only learning-related
output is prompt-visible feedback:

- `TurnReward.learning_progress` stores the immediate Reward Judge prediction
  accuracy proxy.
- Action history renders concise reward components for World, Interest, and
  Agent prompts.
- Memory ledger rows carry the same compact reward feedback so future Memory
  and Goal calls can preserve what worked or failed.

SQLite does not store replay-sample rows or adapter-update rows.
