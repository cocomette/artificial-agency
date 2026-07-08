# Orchestration Overview

Orchestration owns the game loop and coordinates environment, models, memory,
debug tracing, and runtime lifecycle.

Per controllable frame turn it:

1. builds a frame snapshot and optional source `M` row
2. reuses the latest Memory and Goal outputs
3. builds candidates from simple actions plus Agent X coordinate proposals
4. runs World on each candidate
5. runs Interest once on the full candidate set and World predictions
6. asks Agent X to select the final candidate from the World/Interest table
7. submits the action to the environment
8. summarizes the observed transition
9. judges the executed World prediction
10. calls Goal once with previous Memory plus the next frame for reward-only
   Goal delta, computes immediate reward, and appends the finalized ledger
   entry
11. regenerates Memory from action/change/reward ledger rows and calls Goal
   again for next-turn state
12. persists the `M` row plus v1 artifact tables

There is no online trainer or adapter coordinator in this branch. Reward Judge
prediction accuracy is written into `TurnReward.learning_progress` as proxy
feedback. Recent action history and Memory ledger rows carry concise reward
components so later model calls can use text feedback about which actions and
transitions worked.

Animation-unroll frames synthesize `NONE`; Agent X is skipped, but transition
summary, ledger append, Memory, and Goal still run for retained animation
keyframes.
