# Software Diagrams

```mermaid
flowchart LR
    Runtime["runtime shell/config"]
    Orch["orchestration"]
    Env["environment adapter"]
    Models["active models"]
    Memory["SQLite memory"]
    Debug["debug tracing/dashboard"]

    Runtime --> Orch
    Orch <--> Env
    Orch <--> Models
    Orch <--> Memory
    Orch --> Debug
```

```mermaid
sequenceDiagram
    participant Env as Environment
    participant X as Agent X
    participant C as Change Summary
    participant H as Historizer
    participant P as Updater P
    participant M as Memory
    Env->>X: current frame/action space via orchestration
    X-->>Env: selected action via orchestration
    Env-->>C: observed transition via orchestration
    M-->>H: recent agent contexts via orchestration
    C-->>P: transition evidence via orchestration
    H-->>P: context history summary
    P-->>M: updated selected agent context field via orchestration
```
