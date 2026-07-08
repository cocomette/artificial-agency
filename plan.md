

## Highest-priority conceptual fixes


| Fix                                                                                    | Why it matters                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Replace live “learning progress” with true delayed LP.**                             | Current live LP is just World prediction accuracy. True LP should be old-vs-new predictive improvement on future held-out transitions, with update and evaluation data disjoint. This is the central objective in the paper.                                                                                                                            |
| **Separate three signals: prediction accuracy, learning progress, and task progress.** | Prediction accuracy trains/evaluates the World model. Learning progress should reward actions that cause future predictive improvement. Task progress should capture score/level/subgoal progress.                                                                                                                                                      |
| **Fix credit assignment for LP.**                                                      | Current batch-level LP is broadcast to multiple actions. This prevents the system from knowing which action actually produced useful learning.                                                                                                                                                                                                          |
| **Add explicit action-efficiency pressure.**                                           | ARC-AGI-3 scoring strongly penalizes excessive environment actions. Curiosity without action cost will over-explore. Use an action penalty that increases with level step count and expected remaining steps.                                                                                                                                           |
| **Use resource-cost objectives beyond actions.**                                       | Per real environment step, observation tokens may be roughly fixed, but compute is not fixed: candidate count, World calls, reasoning tokens, memory length, retries, and LoRA updates vary. Penalize model calls, generated tokens, memory tokens, and update steps. This directly matches the paper’s observation/action/compute/memory cost framing. |
|                                                                                        |                                                                                                                                                                                                                                                                                                                                                         |


## Highest-priority implementation fixes


| Fix                                                             | Why it matters                                                                                                                                                                   |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                                                 |                                                                                                                                                                                  |
| **Fix ACTION6 suppression.**                                    | Current no-change suppression can suppress the whole ACTION6 class rather than specific coordinates/regions. That can prevent exploration of valid click targets.                |
| **Make LoRA adapters truly per-game and per-version in vLLM.**  | Training data is intended to be per-game, but served adapter names are shared. Parallel games can overwrite each other’s adapters. Use names like `game_03_world_v002`.          |
| **Make LoRA cumulative.**                                       | Current LoRA updates appear to restart from the base model and train only on the latest samples, so earlier learned mechanics are discarded. Continue from the previous adapter. |
|                                                                 |                                                                                                                                                                                  |
| **Use the local Kaggle model path for LoRA base model.**        | The config infers a hub model path, which is unsafe/offline in Kaggle. Set `online_lora.base_model` explicitly to the local weights path.                                        |
| **Replace GRPO for World with supervised transition learning.** | World prediction has direct targets from observed transitions. GRPO plus a VLM judge is expensive and high-variance. Structured supervised targets are more sample-efficient.    |


## Memory and representation fixes


| Fix                                             | Why it matters                                                                                                                                                                                                                                                  |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                                 |                                                                                                                                                                                                                                                                 |
|                                                 |                                                                                                                                                                                                                                                                 |
| **make Memory a compressed mechanic model.**    | Current Memory mixes actual transitions with unexecuted World hypotheses. Memory should contain confirmed rules, goal hypotheses, known no-ops, useful actions, failure causes, and open experiments and be based mainly on the action + change summary history |
| **Do not erase knowledge on game-over/reset.**  | Reset currently clears memory/goal. Instead, persist “this action/sequence caused failure/reset” so the retry is informed.                                                                                                                                      |
| **Finalize reward before regenerating Memory.** | Current Memory can be one turn stale with respect to finalized reward/goal-after metadata. Reorder the loop so Memory sees canonical finalized events.                                                                                                          |


## Agent/planning fixes


| Fix                                      | Why it matters                                                                                       |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
|                                          |                                                                                                      |
|                                          |                                                                                                      |
| **Prefer reversible experiments early.** | Use ACTION7/undo when available. Exploration should gather information while preserving solvability. |
|                                          |                                                                                                      |
|                                          |                                                                                                      |




# Others

1. we should never have a case where interest or agent fails, this is a hard requirement and assumption for the rest of the framework.

