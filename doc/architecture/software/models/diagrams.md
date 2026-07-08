# Model Diagrams

## Role Layout

```mermaid
flowchart TB
    Registry["ModelRegistry"]

    X["models/orchestrator_agent\nX"]
    S["models/world\nS"]
    G["models/goal\nG"]
    D["models/description\nshared S/G capability"]
    P["models/updater\nP"]

    XAdapter["X adapter"]
    SAdapter["S adapter"]
    GAdapter["G adapter"]
    PAdapter["P adapter"]

    XBackend["LLM / agent SDK /\ntool-calling model"]
    SBackend["VLM / world predictor /\nneural model"]
    GBackend["VLM or LLM /\ngoal reasoner"]
    PBackend["LLM updater /\nLoRA updater"]

    Registry --> X --> XAdapter --> XBackend
    Registry --> S --> SAdapter --> D --> SBackend
    Registry --> G --> GAdapter --> D --> GBackend
    Registry --> P --> PAdapter --> PBackend
```

Backends are role-specific. Two roles may share a provider as an
implementation choice, but that is not an architectural dependency.

## Committed Prediction Roles

```mermaid
flowchart LR
    X["Orchestrator Agent X"] -->|"final action"| Orch["Orchestration"]
    Orch -->|"world prediction after decision"| S["World Model S"]
    Orch -->|"goal prediction after decision"| G["Goal Model G"]
    S -->|"committed prediction"| Orch
    G -->|"committed prediction"| Orch
    Orch -->|"predictions for updater and M"| P["Updater / Memory"]
```
