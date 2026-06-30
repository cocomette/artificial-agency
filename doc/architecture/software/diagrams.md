# Software Architecture Diagrams

These diagrams describe the current runtime architecture. Orchestration is the
central owner of execution, persistence, model calls, and environment
communication.

## High-Level Block Diagram

```mermaid
flowchart TB
    Runtime["runtime\nstartup, config, dependency assembly"]
    Orchestration["orchestration\nmain loop and side-effect owner"]
    Environment["environment\nARC-AGI adapter"]
    Memory["memory\nSQLite M and E domains"]
    Models["models\nprovider-neutral role adapters"]
    Agent["orchestrator agent X"]
    Change["change summary"]
    Historizer["agent context historizer"]
    Updater["updater P"]
    Contracts["shared_contracts\ntyped boundaries"]
    ARC["ARC-AGI framework"]
    SQLite["SQLite database"]
    VLLM["vLLM\nOpenAI-compatible API"]

    Runtime --> Orchestration
    Orchestration --> Environment
    Environment --> ARC
    Orchestration <-->|read/write refs and records| Memory
    Memory --> SQLite
    Orchestration --> Models
    Models --> Agent --> VLLM
    Models --> Change --> VLLM
    Models --> Historizer --> VLLM
    Models --> Updater --> VLLM
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
    participant X as Agent X
    participant C as Change Summary
    participant H as Historizer
    participant P as Updater P

    Runtime->>Orch: assemble dependencies and start run
    Orch->>Env: select game and reset
    Env->>ARC: reset()
    ARC-->>Env: initial frame bundle and metadata
    Env-->>Orch: Observation and EnvironmentInfo
    Orch->>Mem: persist initial frame state in M

    loop each frame turn
        Orch->>Env: read current info and valid actions
        Env-->>Orch: action space and lifecycle state
        Orch->>Mem: prewrite/load current M state
        alt controllable final frame
            Orch->>X: decide(text observations, context, action space)
            X-->>Orch: final action and trace
            Orch->>Env: step(final action)
            Env->>ARC: step(action, data, reasoning)
            ARC-->>Env: next frame bundle and metadata
            Env-->>Orch: next Observation and EnvironmentInfo
        else animation frame
            Orch->>Orch: synthesize NONE decision
        end
        Orch->>C: summarize observed transition
        C-->>Orch: change summary
        Orch->>H: summarize recent agent context history
        H-->>Orch: history summary
        Orch->>P: update context from transition and trace
        P-->>Orch: updated agent context
        Orch->>Mem: persist trace, metrics, summary, and context in M
    end
```
