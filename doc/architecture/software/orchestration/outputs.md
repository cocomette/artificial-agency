# Orchestration Outputs

Orchestration emits outputs to the environment, memory, models, updates, and
runtime result boundary.

## Environment Outputs

- One final `ActionSpec` per real environment step.
- Optional reasoning summary derived from the agent trace.
- Reset requests when the ARC lifecycle requires reset.

## Memory Outputs

To persistent memory `M`:

- initial and current observations
- final real actions
- agent traces
- selected world and goal tool outputs
- committed post-decision world and goal predictions
- real next observations
- reward/update quantities
- context documents after updater output has been applied by orchestration

To experimental memory `E`:

- rolling-buffer world tool output observations
- rolling-buffer goal tool output observations
- source observation references for each tool input
- reference ids returned to `X` for later tool calls in the same experimental
  tree

Every successful `S` or `G` call in the experiment loop is persisted before its
result can be reused as a tool input. The input frame is not copied into `E`;
it remains a reference to `M` or to an earlier `E` output.

## Model Outputs

Orchestration calls model roles and receives their outputs, but it does not
rewrite model semantics. It records outputs and passes the relevant results to
the next target:

- tool results and reference ids are returned to `X`
- final action is sent to the environment
- committed post-decision predictions, trace, and transition data are passed to
  updater `P`
- update results are applied to live working contexts and persisted in `M`

## Runtime Outputs

At run end, orchestration returns a run result with stop reason, completed
levels, final lifecycle state, and useful record references.
