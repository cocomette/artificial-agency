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
- world predictions; dormant goal prediction fields remain unset in normal runtime
- real next observations
- transition timing
- score/progress metadata
- context documents after updater output has been applied by orchestration

To experimental memory `E`:

- Agent X tool outputs, when tools are configured
- source memory references for each tool input
- reference ids for later tool calls in the same experimental tree

## Model Outputs

Orchestration calls model roles and receives their outputs, but it does not
rewrite model semantics. It records outputs and passes the relevant results to
the next target:

- tool results and reference ids may be returned to `X` when tools are configured
- final action is sent to the environment
- world predictions, action history, and transition data are passed to updater `P`
- update results are applied to live working contexts and persisted in `M`

## Runtime Outputs

At run end, orchestration returns a run result with stop reason, completed
levels, final lifecycle state, and useful record references.
