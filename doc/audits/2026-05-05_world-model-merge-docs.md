# Audit: world-model merge docs

## Sources

- `README.md` conflict between `orchestration-loop` and `origin/world-model`.
- `doc/architecture/description.md`, moved from `doc/prompts/description.md`.
- `doc/architecture/techstack.md`.
- `doc/architecture/software/orchestration/game_loop/overview.md`.
- `doc/architecture/software/orchestration/game_loop/state_machine.md`.
- `doc/architecture/software/models/roles.md`.
- `src/face_of_agi/orchestration/game_loop/state_machine.py`.
- `src/face_of_agi/models/tools/world/adapter.py`.
- `src/face_of_agi/models/tools/world/config.py`.
- `pyproject.toml`.

## Findings

- `README.md` should keep the `orchestration-loop` wording for main execution
  ownership. The current branch describes a "frame-unrolled orchestration loop"
  that calls `X` on every frame turn, synthetic `NONE` on non-controllable
  frames, runtime delegation into orchestration, and trace output per frame
  turn. This matches the source implementation in
  `src/face_of_agi/orchestration/game_loop/state_machine.py` and the target
  architecture in
  `doc/architecture/software/orchestration/game_loop/overview.md` and
  `doc/architecture/software/orchestration/game_loop/state_machine.md`.

- The incoming `README.md` wording that says "a minimal orchestration shell
  that picks one random valid `GameAction`" and "a single-game runtime loop"
  should not be used as-is. It conflicts with the current ownership rule that
  random selection lives behind the default `X` model role and that runtime is
  only bootstrap/delegation, not the game-loop owner.

- The incoming `README.md` world-model content should be kept, but separated
  from loop ownership. The claims about a concrete Hugging Face Diffusers
  backend using `Qwen/Qwen-Image-Edit`, explicit registry wiring, first-call
  model weight download, device selection, and the manual E2E script are
  consistent with `src/face_of_agi/models/tools/world/adapter.py`,
  `src/face_of_agi/models/tools/world/config.py`, `scripts/world_model_e2e.py`,
  and `pyproject.toml`.

- `doc/architecture/description.md` is useful as original vision/context, not
  as the current execution spec. It correctly preserves the high-level idea
  that `X` can chain world/goal tool calls, tool calls use references into
  state or experimental memory, `E` is step-local, `M` is the durable run
  record, and `K`/`L` role context documents are part of the learning plan.
  These points align with `doc/architecture/system_architecture.md` and
  `doc/architecture/software/models/roles.md`.

- `doc/architecture/description.md` should not be allowed to override the
  frame-turn game-loop docs. Phrases like "run the game 1 step forward" and
  "experimental memory is used only during the agent's tool calling" are too
  coarse for the current orchestration-loop branch, where one environment
  response may be unrolled into several frame turns, `X` is called on
  animation frames too, the updater boundary runs after every frame decision,
  and only final controllable frames submit real actions.

- The tech stack update is architecturally compatible. Adding Diffusers,
  Torch, Transformers, Accelerate, Safetensors, SentencePiece, and Protobuf in
  `pyproject.toml` supports a role-specific world-tool backend and does not
  violate the model adapter rule in `doc/architecture/software/models/roles.md`.
  `doc/architecture/techstack.md` also states that orchestration still depends
  on the provider-neutral world-tool contract rather than on Diffusers.

- `doc/architecture/techstack.md` has minor drift unrelated to the incoming
  model implementation: it still references `doc/architecture/arch.md`, which
  is not present in the current tree. The current architecture source appears
  to be `doc/architecture/system_architecture.md` plus
  `doc/architecture/software/`.

- The new test surface currently fails in this workspace when run with
  `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_model_layout.py
  tests/test_world_tool_adapter.py`. Three world-tool adapter tests fail with
  `ModuleNotFoundError: No module named 'torch'`. This may be an unsynced
  local environment, but there is also a design smell: `WorldToolAdapter`
  imports `torch` before using an injected fake pipeline, so unit tests for
  prompt composition and adapter wrapping require the heavy runtime stack.

## Gaps

- `README.md` needs a merged version that combines orchestration-loop ownership
  with the incoming world-model backend notes. The merge should keep the
  current branch's frame-turn language and append the world-model backend as a
  model-layer capability.

- `README.md` should avoid "starter shell picks random action" language. A
  better direction is: the runtime shell starts the run, orchestration owns the
  loop, and the default `X` adapter selects from the action space provided by
  orchestration.

- `doc/architecture/description.md` needs a short framing header if kept under
  `doc/architecture/`: it should say this is high-level project vision /
  original model concept, while the normative software spec lives under
  `doc/architecture/system_architecture.md` and `doc/architecture/software/`.

- `doc/architecture/description.md` likely needs copy editing before it is used
  as architecture context. It currently has typos and old phrasing such as
  "the game loop describes on step", "plut current", and an unfinished final
  sentence.

- `doc/architecture/techstack.md` should eventually name the new concrete
  world-model stack in the "Chosen Stack" or "Development Direction" lists, not
  only in the model-layer prose. It should also replace references to the
  missing `arch.md`.

- The README setup section now says `uv sync --group dev`, while the manual
  world-model check says `uv run --locked python scripts/world_model_e2e.py`.
  That is probably fine after the lockfile update, but the docs should be clear
  that this path installs large ML dependencies and may download model weights.

## Suggested Follow-Up

- Resolve the README conflict by taking the current branch's orchestration-loop
  bullets and adding only the incoming world-model backend bullet and setup/E2E
  paragraphs.

- Keep `doc/architecture/description.md` as non-normative model vision, or
  rename/frame it as such. Do not use it to replace the current game-loop docs.

- Keep the tech stack update, but follow up with a small documentation cleanup
  that replaces `arch.md` references and lists the concrete world-model ML
  stack in the chosen stack.

- Decide whether unit tests should pass without `torch` when a fake pipeline is
  injected. If yes, defer torch import/generator/inference-mode handling until
  the real pipeline path or provide a lightweight test seam in the adapter.
