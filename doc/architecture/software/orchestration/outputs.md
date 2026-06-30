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
- final real actions or synthetic `NONE` animation decisions
- agent traces
- transition change summaries and action history entries
- turn metrics and score/progress markers
- agent context after updater output has been applied by orchestration

To experimental memory `E`:

- no entries in the current vLLM-only runtime path
- the domain remains available for provider-neutral future tool outputs

## Model Outputs

Orchestration calls model roles and receives their outputs, but it does not
rewrite model semantics. It records outputs and passes the relevant results to
the next target:

- final action is sent to the environment
- change summaries and traces are passed to updater `P`
- update results are applied to live working contexts and persisted in `M`

## Runtime Outputs

At run end, orchestration returns a run result with stop reason, completed
levels, final lifecycle state, and useful record references.
