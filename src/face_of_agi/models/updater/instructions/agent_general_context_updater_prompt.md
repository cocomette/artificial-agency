# Agent model
Output format: return exactly one top-level `updated_context` field whose value
is the complete revised context string; do not return arrays,
or nested objects.
Example shape: {"updated_context":"<complete revised context text>"}

You are given 2 texts:
- 1 describing the agent strategy for a game
- 1 discribing how to act in general for unknown games.

You should update the general agent text in order to include strategies from this game, things that you can identify and generalize.
