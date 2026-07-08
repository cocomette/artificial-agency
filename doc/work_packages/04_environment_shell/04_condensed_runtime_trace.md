# Step 04: Condensed Runtime Trace

## Objective

Emit a very small runtime trace that is readable during manual runs.

## Implementation

- Write trace lines to stdout only.
- For each playable turn, log:
  - `step X: N incoming frames`
  - `action: randomly selected action <action_id>`
- Do not store the trace in SQLite or a structured runtime result store.

## Dependencies

Depends on Step 03.

## Acceptance Check

- Each playable turn prints the frame count and selected action.
- No persistence or test work is added for this trace.
