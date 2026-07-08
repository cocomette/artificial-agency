# Software Architecture Overview

This folder describes the active ARC-AGI-3 runtime architecture. The running
program is coordinated by orchestration, which owns environment interaction,
model calls, reward computation, online update scheduling, and SQLite
persistence.

## Active Runtime Shape

Active model roles:

- Change Summary converts observed frame transitions and animation bundles into
  compact ground-truth text.
- Memory regenerates a fresh free-form run memory document from the original
  first frame, current frame, and a sanitized action/change ledger that keeps
  only `turn_id`, prompt-facing `action`, and `change_summary` per row.
- World predicts change-summary-style text for candidate actions from the
  current frame, candidate action, and Memory. vLLM roles may also include the
  JSON output schema in the system instructions so replay training prompts see
  the same schema contract as live constrained inference.
- Goal predicts structured `goal`, `subgoals`, `steps_remaining`, and
  `confidence` from Memory.
- Interest scores the full candidate set for expected World learning progress
  and expected Goal delta after World predictions are available.
- Reward Judge scores World prediction text against Change Summary text.
- Agent X proposes coordinate candidates and selects the final action from
  World-evaluated and Interest-scored candidates.

World, Interest, and Agent X are trainable online LoRA roles in the v1 runtime.
World trains with image-aware supervised SFT; Interest and Agent X train with
GRPO after delayed World LP labels are available.
Memory, Goal, Change Summary, and Reward Judge are inference-only roles.

The durable state database keeps the existing `m_states` turn rows and adds
explicit v1 tables for turn ledgers, candidate predictions, judge scores, goal
predictions, rewards, replay samples, and LoRA update attempts. Older run
databases are intentionally incompatible and should be reset before running
this branch.

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
9. Regenerate Memory from sanitized action/change ledger rows and call Goal for
   next-turn state.
10. Persist `m_states` plus the v1 artifact tables.
11. When every trainable role has enough new replay samples beyond its rolling
    evaluation reserve, schedule a staged online LoRA update on the background
    worker: score old World rows, train World SFT, load and score the new World
    adapter, backfill executed Interest labels with per-sample LP, train/load
    Interest, rescore Agent replay candidates, then train/load Agent. The next
    orchestration poll activates World, Interest, and Agent together only after
    all roles staged successfully; any staged failure is persisted, cleaned up,
    and raised.
