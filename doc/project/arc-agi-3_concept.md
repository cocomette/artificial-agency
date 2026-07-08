# ARC-AGI-3 Challenge Summary

## Goal

The ARC-AGI-3 challenge is about building an AI agent that can interact with novel, instruction-free game-like environments.

The agent must be able to:

- Probe unfamiliar environments.
- Infer what matters from interaction.
- Discover goals without explicit instructions.
- Learn environment dynamics.
- Plan and execute efficient action sequences.
- Adapt based on feedback.

A perfect score means the agent completes every game or level while matching or beating human action efficiency.

## What the Benchmark Tests

ARC-AGI-3 evaluates whether an agent can perform general intelligence-like behavior in interactive environments.

The main capabilities tested are:

### 1. Probing

The agent must actively gather information by interacting with the environment.

### 2. World Modeling

The agent must build an internal model of how the environment behaves.

### 3. Goal Inference

The agent must infer what the goal is without being directly told.

### 4. Planning and Execution

The agent must choose actions that move it toward the inferred goal efficiently.

## Scoring

The challenge uses a scoring method called **Relative Human Action Efficiency**, or **RHAE**.

The score depends on two main factors:

- **Completion:** whether the agent completes the level.
- **Efficiency:** how many actions the agent uses compared with a human baseline.

Only environment interactions count as actions. Internal computation, reasoning, search, planning, and retries do not count as actions.

The per-level score is calculated as:

```text
level_score = (human_baseline_actions / ai_actions) ^ 2
```

The score is capped above human performance, so being faster than humans helps but cannot dominate the full score.

Later levels are weighted more heavily. If the agent fails or skips later levels, the maximum possible score for that game is capped.

## Action Interface

Each game exposes a standardized action interface.

| Action | Meaning |
|---|---|
| `RESET` | Restart the current game or level. |
| `ACTION1` | Simple action, usually mapped to up. |
| `ACTION2` | Simple action, usually mapped to down. |
| `ACTION3` | Simple action, usually mapped to left. |
| `ACTION4` | Simple action, usually mapped to right. |
| `ACTION5` | Game-specific interaction such as select, rotate, execute, etc. |
| `ACTION6` | Complex action requiring `x,y` coordinates in the range `0–63`. |
| `ACTION7` | Undo, for games that support it. |

Each game specifies which actions are currently available.

For `ACTION6`, the metadata only indicates that a coordinate-based action is available. It does not reveal which coordinates are valid or active.

When a game is over, only `RESET` is valid. Other actions are rejected.

## Code and Programming Rules

Submissions must go through the official Kaggle competition.

The most important programming constraints are:

- No internet access is available during evaluation.
- API-based systems such as hosted LLM calls cannot be used at evaluation time.
- Prize-eligible solutions must be reproducible.
- Prize-eligible solutions must be open source.
- Code and methods authored by the submitter must be released under a permissive public-domain-style license, such as CC0 or MIT-0.
- Third-party code must allow public sharing.
- Participants must open source their solution before receiving official private evaluation scores.

## Competition Mode Constraints

Kaggle evaluation uses Competition Mode.

Important constraints include:

- Environments must be interacted with through the provided API.
- Scoring is performed against all available environments, even if the agent chooses not to interact with some.
- Only level resets are allowed.
- Game resets are not allowed and are treated as level resets.
- The agent can call `make` only once per environment.
- Only one scorecard can be opened.
- In-flight scorecard results cannot be retrieved.

## General Constraints

The challenge is not only about solving known games. The agent must generalize to hidden or novel environments.

The main practical constraints are:

- The system must work offline during evaluation.
- It must avoid reliance on external APIs or internet services.
- It must be reproducible from the submitted code.
- It must be efficient in terms of environment actions.
- It must handle uncertainty and learn from interaction.
- It must infer goals without natural language instructions.
- It must operate under a limited standardized action space.
- It must perform well across many environments, not just hand-picked ones.

## Practical Engineering Takeaway

ARC-AGI-3 should be approached as a general game-playing and reasoning-agent challenge.

The agent needs to combine:

- Probing strategy.
- State representation.
- World modeling.
- Goal inference.
- Planning.
- Action efficiency.
- Robustness across unknown environments.

The main difficulty is not writing a solver for one game, but building a system that can discover and solve many unfamiliar environments under strict offline, reproducibility, and action-efficiency constraints.