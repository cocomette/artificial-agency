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
    World["world model tool S"]
    Goal["goal model tool G"]
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
    Models --> Goal
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
        Orch->>X: decide(context, first observation, current observation, action space, tools)
        X-->>Orch: final action A_i,t and trace T_i,t
        Orch->>Orch: run committed S/G predictions for A_i,t
        Orch->>Env: step(A_i,t, reasoning summary)
        Env->>ARC: step(action, data, reasoning)
        ARC-->>Env: next frame bundle and metadata
        Env-->>Orch: Observation O_i,t+1 and EnvironmentInfo
        Orch->>P: update game-specific contexts from transition, trace, and predictions
        P-->>Orch: L_i,t+1 context documents
        Orch->>Orch: apply contexts to working ContextDocuments
        Orch->>Mem: persist transition, trace, predictions, update quantities, and contexts in M
        Orch->>Mem: clear or expire E_i,t
    end
```

## Agent-As-Tools Flow

```mermaid
sequenceDiagram
    autonumber
    participant Orch as Orchestration
    participant X as Orchestrator Agent X
    participant Router as Tool Routing
    participant S as World Tool Agent S
    participant G as Goal Tool Agent G
    participant E as Experimental Memory E
    participant M as Persistent Memory M

    Orch->>X: decision request with callable world and goal tools
    X->>Orch: request world tool call(action, source_ref)
    Orch->>M: resolve exact persisted state frame when needed
    Orch->>E: resolve exact persisted prediction when needed
    Orch->>Router: route world call
    Router->>S: predict(C^S, action, resolved observation)
    S-->>Router: ToolResult O_hat^S
    Router-->>Orch: world tool result
    Orch->>E: persist temporary world result and ref id
    Orch-->>X: return world result and ref id in active context

    X->>Orch: request goal tool call(prior_prediction_ref)
    Orch->>E: resolve exact persisted world prediction
    Orch->>Router: route goal call
    Router->>G: predict(C^G, resolved observation)
    G-->>Router: ToolResult O_hat^G
    Router-->>Orch: goal tool result
    Orch->>E: persist temporary goal result and ref id
    Orch-->>X: return goal result and ref id in active context

    X-->>Orch: final action and full trace
    Orch->>M: commit selected action, trace, and selected tool artifacts
```
