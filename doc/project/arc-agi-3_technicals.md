# ARC-AGI-3 Development Setup, Rules, and Runtime Constraints

_Last checked: 2026-05-03_

## Purpose

This document focuses on the technical constraints for building and submitting an ARC-AGI-3 agent, especially the development environment, Kaggle runtime limits, offline execution rules, competition-mode behavior, and action-interface constraints.

It intentionally excludes the general challenge overview and prize summary except where they affect implementation.

---

## 1. Submission Environment

ARC-AGI-3 submissions are evaluated through the official Kaggle competition.

The Kaggle submission environment has the following hard limits:

| Constraint | Limit / Rule |
|---|---|
| CPU notebook runtime | `<= 6 hours` |
| GPU notebook runtime | `<= 6 hours` |
| Internet access | Disabled during evaluation |
| External data | Freely and publicly available external data is allowed |
| Submission platform | Kaggle competition notebook / submission flow |

### Engineering implication

The submitted agent must be fully self-contained at evaluation time.

Do not rely on:

- Hosted LLM APIs.
- Remote inference endpoints.
- Online databases.
- Runtime package downloads.
- Web scraping.
- Any service requiring internet access.

All required code, model weights, assets, metadata, heuristics, and lookup tables must be packaged with the submission or available through allowed Kaggle inputs.

---

## 2. Code and Open-Source Requirements

Prize-eligible submissions must be reproducible and open source.

Rules that matter for implementation:

- All submitter-authored code and methods must be open sourced.
- The submitter-authored parts must use a permissive public-domain-style license such as `CC0` or `MIT-0`.
- Third-party code must be under a license that allows public sharing, such as `Apache-2.0` or `GPLv3`.
- The solution must be open sourced before receiving official private evaluation scores.
- The submitted method should be reproducible from the released code.

### Engineering implication

Keep the project structured so that it can be published cleanly:

```text
arc_agi_agent/
  agent/
    policy.py
    planner.py
    perception.py
    memory.py
  models/
    README.md
  configs/
  notebooks/
  LICENSE
  README.md
  requirements.txt or pyproject.toml
```

Avoid hidden dependencies, private services, or manual steps that cannot be reproduced by reviewers.

---

## 3. Recommended Development Stack

The official development interface is the ARC-AGI Toolkit, an open-source Python SDK for ARC-AGI-3 environments.

Typical local installation:

```bash
pip install arc-agi
```

or, with `uv`:

```bash
uv add arc-agi
```

The toolkit lets agents interact with ARC-AGI-3 environments through a consistent API, locally or online.

### Local development vs. competition evaluation

| Mode | Use case | Notes |
|---|---|---|
| Local/offline development | Fast iteration, debugging, repeated experiments | Best for agent development and profiling |
| Online/API development | Scorecards, replays, leaderboard visibility | Requires API access and has request limits |
| Kaggle evaluation | Official submission scoring | Forced into Competition Mode, no internet |

### Engineering implication

Build your agent so the core logic is independent from the environment backend:

```text
observation -> perception -> state/memory update -> action selection -> env.step(action)
```

This makes it easier to test locally and then run unchanged in Kaggle evaluation.

---

## 4. Competition Mode Constraints

Kaggle evaluation is forced into ARC-AGI-3 Competition Mode.

Important restrictions:

| Constraint | Meaning |
|---|---|
| API-only interaction | Environments must be interacted with through the provided API/toolkit interface |
| Full environment scoring | Scoring is against all available environments, even if the agent skips some |
| Level resets only | Game resets are not allowed; game resets become level resets |
| Single `make` call | The agent can call `make` only once per environment |
| Single scorecard | Only one scorecard can be opened |
| No in-flight scorecard reads | `get_scorecard` does not return scoring for an active scorecard |

### Engineering implication

Your evaluation loop should assume a one-pass run.

Avoid designs that require:

- Recreating the same environment multiple times.
- Opening multiple scorecards.
- Checking partial scorecard results mid-run.
- Resetting the full game to probe alternatives.
- Running a calibration pass before the real pass.

A robust loop should look conceptually like:

```python
for env_id in all_envs:
    env = arcade.make(env_id)  # only once per environment
    obs = env.reset()

    while not done and within_budget:
        action, data = agent.act(obs)
        obs = env.step(action, data=data)
```

---

## 5. Runtime Budget Constraints

The headline Kaggle limit is simple:

```text
CPU notebook runtime <= 6 hours
GPU notebook runtime <= 6 hours
```

This is a wall-clock notebook runtime constraint, not an action-count constraint.

However, ARC-AGI-3 scoring is based on environment action efficiency, so there are two separate budgets:

| Budget type | What it limits | Why it matters |
|---|---|---|
| Kaggle runtime | Total compute time | Submission must finish inside the notebook runtime limit |
| Environment actions | Number of actions sent to the game | Directly affects RHAE score |

Internal computation does not count as an environment action, but it still consumes Kaggle runtime.

### Engineering implication

You can spend compute on search, planning, perception, and simulation, but only up to the 6-hour notebook cap.

Tradeoff:

- More internal planning may reduce action count and improve score.
- Too much internal planning may exceed runtime or reduce coverage across environments.

Practical design target:

```text
maximize solved levels per environment
minimize real env.step(...) calls
keep per-environment compute bounded
finish all environments within 6 hours
```

---

## 6. Internet and API Constraints

Internet is disabled during Kaggle evaluation.

That means the following are not usable at evaluation time:

- OpenAI / Anthropic / Google / xAI / hosted LLM APIs.
- External vision model APIs.
- Remote vector databases.
- Remote file storage.
- Package installation from PyPI or GitHub during execution.
- Online documentation lookup.

### Allowed in principle

Freely and publicly available external data is allowed, but it must be available to the notebook without internet during evaluation.

Examples of safer approaches:

- Include external data as a Kaggle Dataset input if allowed.
- Package small static assets with the submission.
- Vendor required source code into the repository if license-compatible.
- Precompute tables or heuristics and include them as files.

### Engineering implication

Before submission, test with internet disabled.

A practical local check:

```bash
# run in a clean environment with no network access if possible
python run_agent.py
```

The agent should not attempt any network calls during execution.

---

## 7. Action Interface Constraints

ARC-AGI-3 exposes a standardized action interface.

| Action | Description |
|---|---|
| `RESET` | Initialize or restart the current game/level state |
| `ACTION1` | Simple action, usually semantically mapped to up |
| `ACTION2` | Simple action, usually semantically mapped to down |
| `ACTION3` | Simple action, usually semantically mapped to left |
| `ACTION4` | Simple action, usually semantically mapped to right |
| `ACTION5` | Game-specific simple action: interact, select, rotate, attach/detach, execute, etc. |
| `ACTION6` | Coordinate action requiring `x,y` coordinates in the `0-63` range |
| `ACTION7` | Undo, when supported |

For `ACTION6`, the agent must provide coordinates:

```python
obs = env.step(
    GameAction.ACTION6,
    data={"x": 32, "y": 32},
)
```

The environment may reveal that `ACTION6` is available, but it does not necessarily reveal which coordinates are useful or active.

### Game-over constraint

When a game reaches a game-over state:

- Only `RESET` is valid.
- Sending another action can return a `400 Bad Request`.

### Engineering implication

The agent should maintain action validity checks:

```python
if obs.state_is_game_over:
    return GameAction.RESET, None

if GameAction.ACTION6 in obs.available_actions:
    # choose x,y deliberately, not randomly unless exploring
    return GameAction.ACTION6, {"x": x, "y": y}
```

For coordinate actions, avoid brute-forcing all `64 * 64 = 4096` coordinates unless the search is carefully budgeted, because every real click/action affects the action-efficiency score.

---

## 8. Scoring-Relevant Technical Constraints

ARC-AGI-3 uses Relative Human Action Efficiency, or RHAE.

The score depends on:

- Completing levels.
- Using few environment actions relative to human baselines.

Per-level score:

```text
level_score = (human_baseline_actions / ai_actions) ^ 2
```

Only environment interactions count as actions.

Do not count these as scoring actions:

- Internal reasoning.
- Search over internal state graphs.
- Model inference.
- Memory updates.
- Planning.
- Offline simulation.

But they still consume notebook runtime.

### Engineering implication

The agent should separate:

```text
internal search budget != environment action budget
```

A strong architecture should track both:

```python
runtime_budget_remaining
real_actions_used
per_level_action_budget
visited_states
tested_state_action_pairs
```

---

## 9. Practical Agent Architecture Constraints

A Kaggle-compatible ARC-AGI-3 agent should be designed around the following constraints:

### Must have

- Offline execution.
- Deterministic or reproducible behavior where possible.
- Bounded runtime per environment.
- Bounded action exploration.
- Graceful handling of failed actions and game-over states.
- Clear state/memory tracking.
- No external API calls.

### Should have

- Frame differencing.
- Object/component extraction from `64x64` visual observations.
- State hashing for visited-state detection.
- Action-effect modeling.
- Coordinate-action proposal strategy for `ACTION6`.
- Per-environment timeout handling.
- Logging that does not require internet.

### Avoid

- Blind random action spam.
- Exhaustive coordinate clicking without pruning.
- Multi-pass assumptions that violate Competition Mode.
- Any dependency installed at runtime from the internet.
- Any call to hosted LLMs or remote vision systems.

---

## 10. Submission Readiness Checklist

Before submitting, verify:

- [ ] The notebook runs with internet disabled.
- [ ] The notebook finishes within 6 hours on CPU or GPU, depending on the selected runtime.
- [ ] All dependencies are included or available in the Kaggle environment.
- [ ] No remote API calls are made.
- [ ] The agent handles Competition Mode restrictions.
- [ ] The agent calls `make` only once per environment.
- [ ] The agent does not rely on reading in-flight scorecard results.
- [ ] The agent handles game-over states by using `RESET` only.
- [ ] `ACTION6` always includes valid `x,y` coordinates in the `0-63` range.
- [ ] Real environment actions are tracked separately from internal reasoning steps.
- [ ] The solution can be open sourced under acceptable licenses.
- [ ] Third-party code licenses are compatible with public release.
- [ ] The run is reproducible from the published code and assets.

---

## 11. Key Engineering Takeaway

Treat ARC-AGI-3 as a constrained offline agent benchmark, not as an API orchestration task.

The core implementation problem is:

```text
Build a self-contained agent that can infer game mechanics and goals from interaction,
while minimizing real environment actions and staying within a 6-hour Kaggle runtime.
```

The strongest engineering focus should be:

1. Efficient exploration.
2. State abstraction.
3. Action-effect modeling.
4. Coordinate-action pruning.
5. One-pass robust evaluation.
6. Offline reproducibility.

---

## Sources

- Kaggle ARC Prize 2026 ARC-AGI-3 overview and rules: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- ARC Prize 2026 rules: https://arcprize.org/competitions/2026
- ARC-AGI-3 competition page: https://arcprize.org/competitions/2026/arc-agi-3
- ARC-AGI-3 docs: https://docs.arcprize.org/
- Competition Mode docs: https://docs.arcprize.org/toolkit/competition_mode
- Actions docs: https://docs.arcprize.org/actions
- Toolkit overview: https://docs.arcprize.org/toolkit/overview
