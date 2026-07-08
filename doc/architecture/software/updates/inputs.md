# Online LoRA Inputs

## Replay Samples

Replay samples contain:

- role: `world`, `interest`, or `agent`
- canonical vLLM request JSON under `prompt_json["request"]`
- schema-shaped target JSON under `completion_json["target"]`
- scalar reward; Agent rows start from the immediate proxy reward and are
  recomputed after delayed per-sample World LP is measured
- held-out flag
- metadata, including base model, request model, source turn, LP evaluation
  bundles for World rows, candidate score tables for Interest/Agent rows,
  Interest label components, and reward components for Agent rows

The stored request includes the live `messages`, inline image data URLs,
`response_format`, role phase, and candidate/source-turn metadata. World
samples target `{"predicted_change": "<observed Change Summary>"}` for
supervised image-aware World SFT. Agent samples target
`{"action": <executed action json>}` for traceability, but Agent GRPO reward
uses the persisted candidate score table after it is rescored by the staged
Interest adapter. Interest samples store the exact live batch scoring request
and receive executed-candidate LP/Goal labels after delayed per-sample World LP
is measured. Only executed World calls, Interest batch calls, and Agent
final-selection calls are trainable in v1.

For vLLM roles configured with `include_output_schema_in_instructions`, the
stored live `messages` also include a model-readable JSON schema in the system
instructions. This keeps replay prompts closer to schema-constrained live
inference.

`held_out_recent_samples` is a dynamic rolling reserve. The newest K eligible
samples for each role are excluded only from the current update and are written
into update metadata as eval sample ids. Reserved samples become trainable
later when newer samples push them out of the reserve. The persisted
`held_out` column remains available for manual or compatibility exclusion.

## Coordinator Config

`online_lora` supplies the served-model metadata, explicit trainable trainer
base (`trainer_base_model_path` or `trainer_base_model`), optional local-only
loading, trainer quantization, update interval, optional
`min_new_samples_per_update`, `max_update_wait_seconds`, adapter root, sample
caps, rolling reserve size, signed LP clipping bounds, GRPO generation count,
completion budget, learning rate, trainer cache controls, and LoRA target
modules. Adapter names and paths are derived from the shared role and version,
not from game id. FP8 bases are invalid for HF trainer loading; FP8 inference
configs must point the trainer at a non-FP8 base. `train_batch_size` and
`train_epochs` determine derived Interest/Agent GRPO `max_steps` as
`ceil(sample_count / train_batch_size) * train_epochs`; `max_update_steps`
optionally caps that derived value. `max_concurrent_trainer_jobs` bounds
process-global concurrent trainer calls while allowing scoring and vLLM adapter
I/O to overlap. When `min_new_samples_per_update` is omitted, it defaults to
`update_interval_turns`.

The shared batcher selects complete turn bundles across games. It trains when
at least `min_new_samples_per_update` train bundles are ready, or after
`max_update_wait_seconds` once at least one train bundle exists beyond that
game's rolling evaluation reserve.
