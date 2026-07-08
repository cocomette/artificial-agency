# Model Inputs

## Agent X

Agent X proposal receives:

- current Memory
- latest Goal prediction
- current observation frame
- allowed current actions
- full action glossary for prompt semantics

Agent X final selection receives the same context plus the bounded candidate
list and World prediction text for each candidate.

## Change Summary

Change summary receives previous observation, current observation, chosen
action, and glossary actions. Provider adapters send separate previous/current
images plus prompt text.

## Memory

Memory receives the original first frame, current frame, and a sanitized
Memory ledger. Each Memory ledger row contains only `turn_id`, prompt-facing
`action`, and `change_summary`. Game-over reset does not replace the first
frame or clear prior ledger entries; orchestration adds an explicit reset
marker whose action/change summary are preserved in the sanitized rows.
Candidate predictions, judge scores, rewards, goals, and ledger metadata remain
internal/persisted orchestration artifacts and are not sent to Memory.

## World

World receives current frame, one candidate action, action glossary, and the
current Memory document.

## Goal

Goal normally receives the latest Memory document and run progress metadata.
For reward computation, orchestration makes one reward-only Goal call using the
previous Memory plus the next frame before regenerating Memory for next-turn
state.

## Reward Judge

Reward Judge receives the executed action, World prediction text, observed
Change Summary text, and optional current/next frames.
