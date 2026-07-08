# Online LoRA Updates Overview

The active v1 update system is online LoRA training for World, Interest, and
Agent X. Memory, Goal, Change Summary, and Reward Judge are inference-only.

Orchestration writes replay samples during real turns and a shared online LoRA
manager batches complete `(run_id, game_id, turn_id)` replay bundles across
parallel games. A game whose samples enter the active batch pauses before its
next turn until the shared World, Interest, and Agent adapters have trained,
loaded into the active inference backend, and locally activated for that game.
Non-contributor games continue and adopt the new shared adapters at their next
turn boundary. In single-HF runtimes, model calls queue while the shared engine
is inside an exclusive trainer window. In same-GPU vLLM runtimes that suspend
the local server for trainer memory, all games pause at turn boundaries while
trainer calls run.

The update worker runs the complete staged pipeline off game threads: old World
scoring, image-aware World SFT, new World load and scoring, Interest label
backfill, Interest GRPO/load, Agent candidate-table rescore, and Agent
GRPO/load. SFT/GRPO trainer calls pass through a process-global gate configured
by `online_lora.max_concurrent_trainer_jobs`; scoring and adapter I/O stay
outside that gate. FP8 served models are inference-only for online LoRA; vLLM
FP8 configs train against a separately configured trainable non-FP8 base.
Single-HF configs instead load that non-FP8 base once with 4-bit BitsAndBytes
QLoRA and train adapters on the same resident model. Single-GPU vLLM debug
launchers can expose a restart command to the manager so trainer calls first
drain active vLLM provider calls, stop vLLM, train with the GPU free, release
trainer CUDA memory, restart vLLM, and reload any staged adapters needed for
the remaining update pipeline.

Adapters are shared and versioned. The manager writes
`adapter_root/shared/world/v001`, `adapter_root/shared/interest/v001`, and
`adapter_root/shared/agent/v001`, and exposes them to the active backend as
`shared_world_v001`, `shared_interest_v001`, and `shared_agent_v001`. vLLM
loads them through its runtime LoRA endpoints; HF keeps them loaded in the
resident PEFT model. Later versions continue from the previous shared adapter
path for that role. The shared adapter root must be empty when the run starts.

Online updates have no partial-success path. If trainer, evaluation, adapter
load, Interest backfill, Agent rescore, or local activation fails, the manager
persists failed update rows for contributor games with context, unloads any
newly staged adapters, deletes partial staged adapter directories,
restores previous local adapter names, and raises a fatal shared-learning
error. Local activation switches World, Interest, and Agent together for each
game only after every role has staged successfully.

## Trainable Roles

- `world`: train rows target Change Summary text as
  `{"predicted_change": ...}` supervised completions. The SFT trainer decodes
  the replay request image data and uses the model processor so training sees
  the same current-frame signal as inference. Train and heldout rows are scored
  before and after loading the new version to compute delayed learning
  progress.
- `interest`: train rows are exact live batch scoring requests. After World LP
  is measured, each executed Interest row receives the executed candidate
  index, realized delayed LP, realized Goal delta, and source update metadata.
  Unexecuted candidates remain unlabeled.
- `agent`: final-selection rows train after candidate score tables are
  rescored by the updated Interest adapter. Agent GRPO rewards generated
  actions by candidate-table advantage, not exact match to the historically
  executed action.

Successful rows record trained replay sample ids, rolling eval sample ids,
per-sample train LP, heldout old/new scores, aggregate heldout LP, maximum
trained replay sample id, sample count, versioned adapter name, and adapter
path.
