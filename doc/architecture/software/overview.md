# Software Architecture Overview

This folder describes the active ARC-AGI-3 runtime architecture. The running
program is coordinated by orchestration, which owns environment interaction,
model calls, reward computation, proxy learning-progress feedback, and SQLite
persistence.

## Active Runtime Shape

Active model roles:

- Change Summary converts observed frame transitions and animation bundles into
  compact ground-truth text.
- Memory regenerates a fresh free-form run memory document from the original
  first frame, current frame, and a sanitized action/change/reward ledger.
- World predicts change-summary-style text for candidate actions from the
  current frame, candidate action, and Memory.
- Goal predicts structured `goal`, `subgoals`, `steps_remaining`, and
  `confidence` from Memory.
- Interest scores the full candidate set for expected World learning-progress
  proxy and expected Goal delta after World predictions are available.
- Reward Judge scores World prediction text against Change Summary text.
- Agent X proposes coordinate candidates and selects the final action from
  World-evaluated and Interest-scored candidates.

The no-LoRA branch is a static vLLM inference runtime. It does not train,
schedule, load, unload, or activate adapters. `learning_progress` remains in the
reward contract as immediate proxy feedback equal to Reward Judge prediction
accuracy, not measured pre/post model improvement.

The durable state database keeps `m_states` turn rows plus explicit v1 tables
for turn ledgers, candidate predictions, judge scores, goal predictions,
rewards, and model-input debug records. Replay-sample and adapter-update tables
are not part of this branch. Older run databases are intentionally incompatible
and should be reset before running this branch.

## Ownership Rule

Only orchestration coordinates cross-module side effects during a game step.
Models do not read or write persistence directly. The runtime module starts the
program and assembles dependencies, but the game loop remains owned by
orchestration.

## Frame Turn Flow

1. Read the current observation and action space from the environment.
2. Prewrite the current source row in `M` when state memory is enabled.
3. Bootstrap or reuse the latest Memory and Goal outputs.
4. Build candidates from all simple actions plus up to the candidate cap of
   Agent-proposed coordinate actions.
5. Run World on each candidate, run Interest once on the full candidate table,
   and ask Agent X to select one final candidate.
6. Submit real actions only on controllable final frames; synthesize `NONE` for
   animation-unroll frames.
7. Summarize the observed transition with Change Summary.
8. Judge the executed World prediction, call Goal once for reward-only Goal
   delta, compute immediate reward, and append the finalized ledger row.
9. Regenerate Memory from action/change/reward ledger rows and call Goal for
   next-turn state.
10. Persist `m_states` plus the v1 artifact tables.

The next turn's World, Interest, Agent, and Memory calls receive recent reward
and proxy learning-progress feedback through action history and Memory ledger
rows.
