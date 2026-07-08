# Online LoRA Outputs

Online update attempts are persisted as `LoRAUpdateRecord` rows.

## Attempt States

- `queued`
- `running`
- `succeeded`
- `failed`

Successful attempts record the versioned adapter name, adapter path, trained
replay sample ids, rolling eval sample ids, maximum trained replay sample id,
and sample count. World success rows also record old/new train and heldout
Reward Judge scores, per-sample delayed LP for executed train rows, aggregate
heldout LP for update health, clipping bounds, worker load status, and local
activation status. In parallel runs, contributor games each receive per-game
status rows for the shared update; metadata includes shared batch/sample
references because SQLite replay sample ids are only unique inside one game
database. vLLM load happens in the background worker under unique shared
versioned names. The next game-boundary poll locally switches World, Interest,
and Agent together only after every role has staged successfully.

Failed trainer, load, evaluation, backfill, rescore, or activation attempts
record the error for contributor games, unload any newly staged vLLM adapters,
delete partial staged adapter directories, restore previous local adapter
names, and raise. Failed shared online updates are fatal for all registered
games by design.

## Quality Signal

World SFT trains image-aware supervised completions that target
`{"predicted_change": "<observed Change Summary>"}`. Before a staged World
adapter is loaded, the coordinator scores old World predictions on the train
and heldout World replay rows. After load, it scores new predictions by
explicitly requesting the staged versioned adapter name. Missing or failing
Reward Judge fails the World update.

Interest rewards compare the generated executed-candidate value row against
that turn's delayed per-sample LP and Goal-delta labels, with a small
confidence calibration term.
Agent rewards parse the generated action, match it against the candidate score
table in replay metadata, and return the within-prompt advantage-normalized
blended score. Invalid JSON, malformed actions, and non-candidate actions
receive `min(valid_candidate_rewards) - 1e-6`, or `-1.0` when the candidate
table is missing.

World learning progress is measured separately from the immediate turn reward.
For each executed World train row, the coordinator computes clipped signed LP
as `new_score - old_score` and backfills the paired Interest and Agent replay
rows for that same source turn. Heldout aggregate LP remains update-health
metadata; it is not broadcast as the label for every action. Agent replay
rewards are recomputed as:

`lp_weight * learning_progress + goal_weight * goal_delta + progress_bonus - resource_cost`

That recomputed scalar remains persisted for diagnostics. Agent GRPO policy
improvement uses candidate-table advantages from the latest Interest rescore,
so the Agent can be rewarded for choosing a high-value candidate even when it
differs from the action historically executed in the turn.
