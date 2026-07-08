# Memory Domains

## `M` State Memory

`M` is the durable source of truth for completed frame turns. Each complete
row contains:

- game/run identifiers and frame indexes
- current observation payload
- chosen action
- agent context
- Agent X trace
- turn metrics
- metadata

Source rows may be prewritten before Agent X acts and completed after the
observed transition is resolved.

Additional v1 tables store the turn ledger, candidate World predictions,
Reward Judge scores, Goal predictions, separated reward components, and
model-input debug records. Within one run, game-over reset does not clear the
turn ledger used to regenerate Memory; orchestration appends an explicit reset
marker so Memory can preserve prior mechanics, failed attempts, and hypotheses.

## `E` Experimental Memory

`E` stores generic tool invocations and outputs for Agent X. It is pruned by
run/game/turn buffer settings and is not part of learned context hydration.
