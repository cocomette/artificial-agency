# Runtime Config

Runtime YAML config is loaded by `face_of_agi.environment.config`.

## Required Runtime Keys

- `max_actions_per_level`
- one of `game_index`, `game_indices`, `game_ids`, or `game_selection` for most
  normal runs
- `models`

## Active Model Shape

The active v1 roles can run either through the existing `vllm` backend or the
opt-in `hf_transformers` backend. `models.shared_vlm` supplies shared defaults
for roles that use the same backend.

```yaml
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  change:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  memory:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  world:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  goal:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  interest:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  reward_judge:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8

# Single-process HF debug configs use the same role slots with:
#   backend: hf_transformers
#   model: Qwen/Qwen3.6-35B-A3B
#   quantization: bnb_4bit
#   local_files_only: true

online_lora:
  enabled: true
  base_model: Qwen/Qwen3.6-35B-A3B-FP8
  trainer_base_model: Qwen/Qwen3.6-35B-A3B
  trainer_local_files_only: true
  trainer_quantization: bnb_4bit
  trainer_device_map: cuda:0
  update_interval_turns: 4
  min_new_samples_per_update: 4
  max_update_wait_seconds: 30.0
  train_batch_size: 1
  train_epochs: 1
  max_update_steps: null
  max_concurrent_trainer_jobs: 1
  trainer_cache_enabled: false
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

`models.agent`, `models.change`, `models.memory`, `models.world`,
`models.goal`, `models.interest`, and `models.reward_judge` are required active
slots. `models.historizer` and `models.updater` are removed runtime keys and
fail config loading.

The vLLM backend serves an OpenAI-compatible chat endpoint. vLLM server startup
must enable LoRA and runtime adapter updates; the committed launcher helpers
add `--enable-lora`, `--max-loras`, and
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.

The HF/Transformers backend keeps one trainable VLM and processor warm inside
the runtime process. All v1 roles call that shared engine. Online LoRA trains
adapters on the same model under an exclusive trainer window; queued
generation waits behind training and resumes without vLLM restarts. HF JSON
outputs rely on prompt-visible schemas and the existing repair loop, not
server-side constrained decoding.

`online_lora` controls bounded shared async updates for trainable roles
`world`, `interest`, and `agent`. `world` uses image-aware supervised
transition SFT. `interest` and `agent` use image-aware GRPO through the same
VLM processor. Adapter names are derived from role and version as
`shared_<role>_vNNN`; the config does not accept stable adapter-name fields.
Adapter files are stored below `adapter_root/shared/<role>/vNNN`, and the
shared adapter root must be empty at run start. FP8 model repos are inference
only for online LoRA; vLLM FP8 configs must set `trainer_base_model_path` or
`trainer_base_model` to a trainable non-FP8 base. HF debug configs use the
non-FP8 base for both inference and training with `quantization: bnb_4bit`.
Modal requires the non-FP8 base to be pre-downloaded; Kaggle requires the
matching input dataset to be mounted.

`min_new_samples_per_update` controls the shared train-bundle gate; when
omitted, it defaults to `update_interval_turns`. `max_update_wait_seconds`
flushes a smaller available batch once at least one game has train bundles
beyond its rolling reserve. `train_batch_size` and `train_epochs` derive GRPO
`max_steps` for Interest and Agent from the selected replay batch, and
`max_update_steps` optionally caps that value.
`max_concurrent_trainer_jobs` bounds process-global concurrent SFT/GRPO trainer
calls; scoring, label backfill, and adapter load/unload remain outside that
trainer gate. `trainer_cache_enabled` keeps one HF VLM base and processor warm
for vLLM deployments with separate trainer memory. Same-GPU FP8 vLLM debug
runtimes disable the warm trainer cache and may suspend/restart local vLLM
around trainer calls because the 35B trainable QLoRA base cannot coexist with
the FP8 vLLM server. Single-HF debug runtimes do not restart a server; they
queue generation while the shared engine trains.
`trainer_quantization`
supports `none` and `bnb_4bit`; `bnb_4bit` loads the trainer base through
BitsAndBytes and prepares it for k-bit PEFT training before adapters are added
or loaded. `trainer_device_map: cuda:0` forces the trainable base onto the GPU;
debug FP8 configs pair that with a controlled vLLM suspension path instead of
allowing trainable modules to spill to CPU or disk.

The environment config also exposes zero-default resource-cost fields:
`reward_action_penalty`, `reward_trace_seconds_penalty`,
`reward_input_token_penalty_per_1k`, and
`reward_output_token_penalty_per_1k`. Active vLLM configs set these
explicitly to `0.0`.

## Animation Keyframes

`animation_keyframe_pixel_threshold` controls how many changed raw frame
cells/pixels are needed before an intermediate animation frame becomes a
retained frame turn. It defaults to `8`; set it to `0` to keep every
non-duplicate frame. The final frame in each environment bundle is always
retained.

## Memory Compatibility

The current schema requires a database reset. Local SQLite run databases
created by older runtime shapes should be reset instead of migrated.
