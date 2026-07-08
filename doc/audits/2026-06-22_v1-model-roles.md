# Audit: V1 Model Roles And Online Training

## Sources

- User-provided architecture note in the 2026-06-22 chat.
- Latest commit: `130417af` (`full implementation`).
- Architecture docs:
  - `doc/architecture/software/overview.md`
  - `doc/architecture/software/models/roles.md`
  - `doc/architecture/software/updates/overview.md`
  - `doc/architecture/software/updates/outputs.md`
- Runtime code:
  - `src/face_of_agi/orchestration/game_loop/state_machine.py`
  - `src/face_of_agi/orchestration/game_loop/v1_roles.py`
  - `src/face_of_agi/orchestration/online_lora.py`
  - `src/face_of_agi/models/change/adapter.py`
  - `src/face_of_agi/models/memory/adapter.py`
  - `src/face_of_agi/models/world/adapter.py`
  - `src/face_of_agi/models/goal/adapter.py`
  - `src/face_of_agi/models/reward_judge/adapter.py`
  - `src/face_of_agi/models/orchestrator_agent/providers/vllm.py`
  - `src/face_of_agi/memory/sqlite.py`
- Verification run:
  - `uv run pytest tests/suites/test_v1_runtime_roles.py tests/suites/test_online_lora.py tests/suites/test_sqlite_memory.py tests/suites/test_runtime_smoke.py`
  - Result: `17 passed`.

## Findings

- **High: World and Agent X default to stable LoRA adapter names before an adapter is loaded.**
  `VLLMWorldConfig.lora_adapter_name` defaults to `world-lora`, and `VLLMOrchestratorAgentConfig.lora_adapter_name` defaults to `agent-lora` (`src/face_of_agi/models/world/config.py:14`, `src/face_of_agi/models/orchestrator_agent/config.py:93`). Both vLLM callers use `config.lora_adapter_name or config.model` as the request model (`src/face_of_agi/models/vllm_roles.py:86`, `src/face_of_agi/models/orchestrator_agent/providers/vllm.py:397`). The coordinator only loads those adapter names after a successful scheduled update (`src/face_of_agi/orchestration/online_lora.py:147`). Unless vLLM has preloaded placeholder adapters, first-turn World or Agent calls can fail instead of falling back to the base model.

- **High: Online LoRA samples do not match live inference inputs or output contracts.**
  The live World prompt includes the current frame image, candidate index, action glossary, Memory, system instructions, and JSON schema (`src/face_of_agi/models/world/adapter.py:48`, `src/face_of_agi/models/world/adapter.py:66`). The live Agent selection prompt includes the current frame image, candidate list, World predictions, Memory, Goal, system instructions, and final-action schema (`src/face_of_agi/models/orchestrator_agent/providers/vllm.py:335`, `src/face_of_agi/models/orchestrator_agent/providers/vllm.py:380`). Replay samples instead store simplified text only: World gets Memory plus an action dict, and Agent gets Memory, goal text, steps, and prediction lines (`src/face_of_agi/orchestration/game_loop/v1_roles.py:668`, `src/face_of_agi/orchestration/game_loop/v1_roles.py:698`, `src/face_of_agi/orchestration/game_loop/v1_roles.py:709`). Training then converts `completion_json["target"]` with `str(...)` and rewards completions by text similarity plus recorded reward (`src/face_of_agi/orchestration/online_lora.py:239`, `src/face_of_agi/orchestration/online_lora.py:281`). This trains a different task distribution than the one the adapters serve, and it does not directly train the JSON shapes expected at inference.

- **Medium: The documented held-out quality signal is not implemented.**
  The docs say the default LP signal is a held-out recent replay quality delta measured by pre/post Reward Judge scores (`doc/architecture/software/updates/outputs.md:16`). The config exposes `held_out_recent_samples` (`src/face_of_agi/environment/config.py:58`), and replay rows mark every fifth turn as held out (`src/face_of_agi/orchestration/game_loop/v1_roles.py:679`). `_train_grpo_adapter` still reads the newest `max_train_samples` without filtering held-out rows and performs no pre/post Reward Judge evaluation (`src/face_of_agi/orchestration/online_lora.py:221`). The actual reward is `0.5 * recorded_reward + 0.5 * SequenceMatcher(target)` (`src/face_of_agi/orchestration/online_lora.py:287`).

- **Medium: Real turns regenerate Memory and Goal twice, and reward may disagree with persisted Goal.**
  For controllable turns, the loop first regenerates Memory/Goal after appending a provisional ledger entry without reward, computes reward from that first `next_goal`, then replaces the ledger entry with reward and regenerates Memory/Goal again (`src/face_of_agi/orchestration/game_loop/v1_roles.py:273`, `src/face_of_agi/orchestration/game_loop/v1_roles.py:284`, `src/face_of_agi/orchestration/game_loop/v1_roles.py:301`). The final persisted `goal_after` can differ from the `current_goal` used for `reward.goal_delta`, and each real turn pays for two Memory plus two Goal calls.

- **Medium: Adapter reload failures are not persisted as failed LoRA attempts.**
  Trainer failures inside `_run_update` are persisted as `failed` (`src/face_of_agi/orchestration/online_lora.py:120`). Reload failures from `_load_lora_adapter` occur later in `poll` or `shutdown` after `future.result()` succeeds, but those paths do not catch the exception and write a failed status (`src/face_of_agi/orchestration/online_lora.py:50`, `src/face_of_agi/orchestration/online_lora.py:89`). This weakens the doc claim that failed attempts are persisted and surfaced.

- **Low: Candidate capping can contradict the “all simple actions” architecture text.**
  The docs say orchestration includes all valid simple non-coordinate actions before asking for coordinate proposals (`doc/architecture/software/models/roles.md:10`). `_candidate_actions` truncates simple actions with `simple_actions[:max_candidates]` (`src/face_of_agi/orchestration/game_loop/v1_roles.py:374`). The default cap is large enough for normal ARC simple actions, but a lower config can silently omit simple actions.

## Gaps

- Change Summary is implemented as an inference-only transition summarizer. It uses deterministic model-visible changed-pixel evidence, skips the model when frames are identical, and overrides impossible “no change” answers when pixels changed (`src/face_of_agi/orchestration/game_loop/actions/steps.py:386`, `src/face_of_agi/orchestration/game_loop/actions/steps.py:442`). This is a good v1 ground-truth source, but it is only as reliable as the VLM summary and Reward Judge agreement.

- Memory matches the proposed design closely: it receives first/current frames plus the full ledger and rewrites a free-form document rather than appending incrementally (`src/face_of_agi/models/memory/adapter.py:42`, `src/face_of_agi/models/memory/adapter.py:68`). The main downside is cost and possible summary drift, especially because it is called every retained animation turn and twice on real turns.

- World matches the intended input shape at inference: current frame, candidate action, action glossary, and Memory (`src/face_of_agi/models/world/adapter.py:66`). Its online target is the observed Change Summary, but the current trainer does not replay the same visual/schema prompt used online.

- Goal reads Memory plus the current frame and predicts `goal`, `subgoals`, `steps_remaining`, and `confidence` (`src/face_of_agi/models/goal/adapter.py:42`). It is inference-only, which is consistent with v1, but the reward path trusts step deltas from an untrained model and does not calibrate or guard against oscillating estimates.

- Agent X is implemented as the proposed two-stage pipeline: runtime inserts simple actions, Agent proposes coordinate candidates, World predicts each candidate, and Agent selects one candidate (`src/face_of_agi/orchestration/game_loop/v1_roles.py:132`, `src/face_of_agi/orchestration/game_loop/v1_roles.py:145`, `src/face_of_agi/orchestration/game_loop/v1_roles.py:160`). The current training sample only targets the executed action and does not implement counterfactual unexecuted-candidate rewards.

- Reward Judge is inference-only and compares executed World prediction against Change Summary with optional previous/current frames (`src/face_of_agi/models/reward_judge/adapter.py:42`). This is useful as a normalized World score, but it introduces another VLM dependency into the reward path.

## Suggested Follow-Up

- Make World and Agent request the base model until their stable LoRA adapters have actually been loaded, or explicitly preload valid initial adapters before the first turn.
- Persist replay samples from the exact captured vLLM request payloads, including messages, images or image references, response schema identity, and expected JSON completion shape.
- Decide whether online training is GRPO over free-form completions or supervised/GRPO over role JSON. Then make replay targets match the live role output schema.
- Implement the held-out split described in `updates/outputs.md`, or rewrite the docs to describe the current SequenceMatcher-plus-recorded-reward signal.
- Regenerate Memory/Goal once per real turn, or make the reward calculation use the same final Goal prediction that is persisted.
- Add focused tests for first-turn vLLM model selection without preloaded adapters, replay prompt fidelity, held-out exclusion, reload failure persistence, and candidate caps below the simple-action count.
