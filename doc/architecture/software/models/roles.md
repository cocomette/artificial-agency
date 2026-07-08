# Model Roles

## Agent X

Agent X owns action choice. The v1 loop calls it in two stages:

1. propose distinct coordinate candidates for `ACTION6` up to the candidate cap
2. select one final candidate after World has predicted each candidate outcome
   and Interest has scored the candidate set

Orchestration always includes all valid simple non-coordinate actions before
asking for coordinate proposals. Agent X remains the final action owner; the
runtime does not argmax over Interest scores. Prompt guidance asks Agent X to
prefer reversible, low-risk probes early when uncertainty is high; no separate
reversibility reward bonus is added.

Repeated zero-change `ACTION6` attempts suppress only the exact coordinate in
prompt evidence. `ACTION6` remains available as an action class, including when
simple actions are also available; `ACTION6` without coordinates is not
class-suppressed. Recent action history includes reward components and proxy
learning-progress feedback from prior turns.

## Change Summary

The change summary role receives the previous observation, current observation,
chosen action, action glossary, and the deterministic changed-pixel percentage
for the model-visible cropped transition. It returns the observed-transition
ground truth text used by World and Reward Judge.

## Memory

Memory regenerates a fresh free-form run document every turn from the original
first frame, current frame, and a sanitized ledger. Memory is not appended
incrementally; the model receives `turn_id`, prompt-facing `action`,
`change_summary`, and concise reward feedback for each ledger row. Reward
feedback includes total reward, proxy `learning_progress`, prediction accuracy,
goal delta, progress bonus, resource cost, and short judge notes/error tags when
available. Candidate tables and internal debug metadata stay out of the Memory
input. A reset adds an explicit ledger marker instead of clearing prior run
knowledge.

## World

World predicts change-summary-style text for one candidate action from the
current frame, candidate action, and current Memory document. The observed
Change Summary is judged against the executed World prediction after the real
environment transition. That score becomes immediate proxy learning-progress
feedback for later prompts.

## Interest

Interest receives the current frame, Memory, Goal, candidate actions, World
predictions, and recent action history. It returns one value row per candidate:

- `candidate_index`
- expected World learning-progress proxy
- expected Goal delta
- confidence
- short notes

Orchestration computes the live blended score as:

`lp_weight * confidence * expected_learning_progress + goal_weight * expected_goal_delta`

The resulting value table is added to the Agent final-selection prompt and
persisted with candidate metadata. There is no delayed label backfill or policy
update in this branch.

## Goal

Goal reads Memory and returns structured `goal`, `subgoals`,
`steps_remaining`, and `confidence`. Agent X uses the latest stored Goal
prediction. Orchestration also calls Goal once after observing the next frame
and before Memory regeneration to compute reward-only Goal delta.

## Reward Judge

Reward Judge compares a World prediction with the observed Change Summary and
returns `score: 0..1`, short notes, and error tags. It is inference-only and is
used for current-turn World quality, proxy learning-progress, reward shaping,
and dashboard inspection.
