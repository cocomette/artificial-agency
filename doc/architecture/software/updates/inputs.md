# Update Inputs

There are no online update inputs in this branch. Runtime feedback inputs are
ordinary model context:

- recent action history with reward and proxy learning-progress fields
- Memory ledger rows with action, change summary, reward, proxy
  learning-progress, prediction accuracy, goal delta, progress bonus, resource
  cost, and short judge notes/error tags
- persisted candidate predictions, judge scores, Goal predictions, rewards,
  and model-input debug records for inspection

No trainable model base, replay bundle, adapter root, trainer quantization, or
training schedule is configured.
