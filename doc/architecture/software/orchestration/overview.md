# Orchestration Overview

Orchestration owns the game loop and coordinates environment, models, memory,
debug tracing, and runtime lifecycle.

Per controllable frame turn it:

1. builds a frame snapshot and optional source `M` row
2. reuses the latest Memory and Goal outputs
3. builds candidates from simple actions plus Agent X coordinate proposals
4. runs World on each candidate
5. runs Interest once on the full candidate set and World predictions
6. asks Agent X to select the final candidate from the World/Interest table
7. submits the action to the environment
8. summarizes the observed transition
9. judges the executed World prediction
10. calls Goal once with previous Memory plus the next frame for reward-only
   Goal delta, computes immediate reward, and appends the finalized ledger
   entry
11. regenerates Memory from sanitized action/change ledger rows and calls Goal
   again for next-turn state
12. persists the `M` row plus v1 artifact tables

Online LoRA updates are staged by a shared manager outside the frame-turn
critical path once enough complete trainable replay bundles are available
across games, or the configured max-wait elapses. A game whose samples enter
the active batch pauses before its next turn until the shared update is ready;
non-contributor games continue and adopt completed adapters at their own turn
boundary. In single-HF runtimes, model calls queue while the shared engine is
inside an exclusive trainer window. In same-GPU runtimes that suspend local
vLLM during trainer calls, all registered games pause at turn boundaries for
the update window because inference is globally unavailable while the trainer
owns the GPU. The background worker runs the complete staged pipeline: World
training, train/heldout old/new World scoring, adapter load, Interest label
backfill, Interest training/load, Agent candidate-table rescore, and Agent
training/load. Failed shared updates persist contributor context, unload staged
adapters, delete partial adapter directories, restore prior local adapter
names, and raise for all registered games.

Animation-unroll frames synthesize `NONE`; Agent X is skipped, but transition
summary, ledger append, Memory, and Goal still run for retained animation
keyframes.
