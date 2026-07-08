# Model Diagrams

## Role Layout

```mermaid
flowchart TB
    Registry["ModelRegistry"]

    X["models/orchestrator_agent\nX"]
    S["models/tools/world\nS"]
    G["models/tools/goal\nG"]
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
    Registry --> S --> SAdapter --> SBackend
    Registry --> G --> GAdapter --> GBackend
    Registry --> P --> PAdapter --> PBackend
```

Backends are role-specific. Two roles may share a provider as an
implementation choice, but that is not an architectural dependency.

## Tool Roles

```mermaid
flowchart LR
    X["Orchestrator Agent X"] -->|"requests ToolCall"| Orch["Orchestration"]
    Orch -->|"world call"| S["World Tool S"]
    Orch -->|"goal call"| G["Goal Tool G"]
    S -->|"ToolResult"| Orch
    G -->|"ToolResult"| Orch
    Orch -->|"result in active context"| X
```
