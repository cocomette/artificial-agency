# Model Roles

## Agent X

Agent X is dormant in the current runtime game loop. Its adapter code still
exists, but runtime bootstrap registers `orchestrator_agent=None`; controllable
actions are selected by updater P and wrapped by orchestration as decision
traces.

## Change Summary

The change summary role receives the previous observation, current observation,
chosen action, action glossary, previous change-summary element output, and
optional post-action animation frame bundle. The attached frames remain the
source of truth for the current visible change. It returns an
`elements` array plus a `change_detected` boolean indicating whether any visible
difference was detected across the attached image set. Each element has
`element_name`, `element_description`, and `element_mutation`; unchanged visible
elements keep an empty mutation. Orchestration stores the element list in action
history and derives the prompt-facing `change_summary` text as one bullet per
element, rendering the name, description, and either the mutation or an explicit
no-detected-changes note. The derived
summary is stored in action history
alongside the updater mode that selected the action, cumulative
`completed_levels`, and controllable `action_count`. Its attached frames are
cropped by `input_image_crop_arc_grid_edges`, defaulting to 4 source ARC-grid
cells per edge. Animation bundles keep the ordered frame array after exact
consecutive duplicate filtering, then resize all attached frames to fit the
change model's `animation_frame_budget_coefficient` configured-frame-area
budget. The coefficient defaults to `2` and values below `2` are clamped to
`2`. Optional `gaussian_blur_kernel_size` config blurs the final image copies
sent to the change-summary provider before optional independently sampled
zero-centered Gaussian RGB noise from `gaussian_noise_deviation`.

## World Model

The world-model role runs before the historizer. It receives the previous
world-model context from state metadata plus action history, allowed actions,
and the attached current frame for the latest transition. Previous frames and
animation bundles are not attached to world-model provider calls; their
transition evidence remains available through action history. The action
history includes cumulative `completed_levels` and controllable
`action_count`. The attached current frame is cropped by
`input_image_crop_arc_grid_edges`, defaulting to 4 source ARC-grid cells per
edge. It returns per-action effect summaries for the allowed action set, the
latest world description, and special-event memory for isolated or sporadic
feedback.

## Agent-Context Historizer

The historizer receives the fresh world-model output from that same turn plus
recent same-run updater strategy snapshots collected from state metadata. Each
snapshot contains the latest `probing_strategy` and `policy_strategy`
after an updater ran. The historizer call returns compact `probing_evolution`,
`policy_evolution`, and `strategy_summary` fields, and proposes whether probing
or policy should update next. Orchestration enforces the configured probing cap
before updater dispatch.

## Level Summary

The level-summary role runs when ARC reports that one or more levels have been
completed. It receives the completed level number and the ordered same-run
strategy snapshots persisted since the previous level summary. It returns a
compact `solution_method` that removes dead ends and keeps reusable same-game
method guidance. Orchestration persists this summary and passes the latest
same-run method to subsequent probing/policy updater calls.

## Updater P

Updater P has two active tasks:

- `agent_probing`: update `probing_strategy` and choose the next
  mechanics-learning action after controllable transitions.
- `agent_policy`: update `policy_strategy` and choose the next
  goal-pursuing action after controllable transitions.
- `general`: update agent general context at run end.

Each agent updater receives transition evidence, bounded action history with
`completed_levels` and `action_count`, allowed actions, and historizer output.
Their attached current frame is cropped by `input_image_crop_arc_grid_edges`,
and ACTION6 outputs carry a target description, cropped normalized target
bounding box, and target RGB color. The updater adapter deterministically
selects the closest-color pixel inside that box, preferring pixels closest to
the box center among equal color matches, then maps that point back to the full
ARC grid. Both mode-specific agent
updaters receive the current
`probing_strategy` and `policy_strategy` as previous game context.
Both receive `probing_evolution`, `policy_evolution`, and `strategy_summary`
from the historizer.
When available, both also
receive the latest previous-level
`solution_method`. Probing returns `probing_strategy` plus `next_actions`.
Policy returns `policy_strategy` plus `next_actions`. `next_actions` is an
ordered array whose length must equal the configured action window for the
active mode. Both agent updaters receive the fresh world-model context, but neither
persists or rewrites it.

`next_actions` is queued by orchestration. One queued action is submitted on
each controllable frame after previous-to-current transition modeling. The game
loop does not call Agent X to revise those actions.
