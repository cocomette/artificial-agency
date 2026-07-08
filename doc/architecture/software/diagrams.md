# Software Architecture Diagrams

These diagrams describe the target architecture. They intentionally show
orchestration as the central owner of execution, persistence, model routing,
and environment communication.

## High-Level Block Diagram

```mermaid
flowchart TB
    Runtime["runtime\nstartup, config, dependency assembly"]
    Orchestration["orchestration\nmain loop and side-effect owner"]
    Environment["environment\nARC-AGI adapter"]
    Memory["memory\nSQLite M and E domains"]
    Models["models\nprovider-neutral role adapters"]
    Agent["orchestrator agent X"]
    World["world prediction model S"]
    Goal["goal prediction model G"]
    Updater["updater P"]
    Contracts["shared_contracts\ntyped boundaries"]
    ARC["ARC-AGI framework"]
    SQLite["SQLite database"]

    Runtime --> Orchestration
    Orchestration --> Environment
    Environment --> ARC
    Orchestration <-->|read/write refs and records| Memory
    Memory --> SQLite
    Orchestration --> Models
    Models --> Agent
    Models --> World
    Models -. dormant .-> Goal
    Models --> Updater
    Orchestration -. uses .-> Contracts
    Environment -. uses .-> Contracts
    Memory -. uses .-> Contracts
    Models -. uses .-> Contracts
```

## Main Execution Loop

```mermaid
sequenceDiagram
    autonumber
    participant Runtime
    participant Orch as Orchestration
    participant Env as Environment Adapter
    participant ARC as ARC-AGI Framework
    participant Mem as SQLite Memory
    participant X as Orchestrator Agent X
    participant P as Updater P

    Runtime->>Orch: assemble dependencies and start run
    Orch->>Env: select game and reset
    Env->>ARC: reset()
    ARC-->>Env: initial frame bundle and metadata
    Env-->>Orch: Observation O_i,0 and EnvironmentInfo
    Orch->>Mem: persist initial observation in M

    loop each real environment step t
        Orch->>Env: read current info and valid actions
        Env-->>Orch: action space and lifecycle state
        Orch->>Mem: read relevant M and initialize/read E_i,t
        Orch->>X: decide(agent context, observations, action space)
        X-->>Orch: final action A_i,t and trace T_i,t
        Orch->>Orch: run world prediction for A_i,t
        Orch->>Env: step(A_i,t, reasoning summary)
        Env->>ARC: step(action, data, reasoning)
        ARC-->>Env: next frame bundle and metadata
        Env-->>Orch: Observation O_i,t+1 and EnvironmentInfo
        Orch->>P: update game-specific contexts from transition, trace, and predictions
        P-->>Orch: L_i,t+1 context documents
        Orch->>Orch: apply contexts to working ContextDocuments
        Orch->>Mem: persist transition, trace, world prediction, timing, score delta, and contexts in M
        Orch->>Mem: clear or expire E_i,t
    end
```

## Active Context Flow

```mermaid
sequenceDiagram
    autonumber
    participant Orch as Orchestration
    participant X as Orchestrator Agent X
    participant S as World Model S
    participant P as Updater P
    participant M as Persistent Memory M

    Orch->>X: decision request with agent context
    X-->>Orch: final action and trace
    Orch->>S: predict(C^S, final action, current observation)
    S-->>Orch: world prediction
    Orch->>P: update contexts from trace, world prediction, and observed transition
    P-->>Orch: revised C^S, C^X
    Orch->>M: commit action, trace, world prediction, transition, and contexts
```
