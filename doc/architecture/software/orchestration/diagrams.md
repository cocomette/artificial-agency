# Orchestration Diagrams

## Ownership Around One Step

```mermaid
flowchart LR
    Env["Environment Adapter"] -->|Observation and action space| Orch["Orchestration"]
    Orch -->|Decision request with S/G contexts| X["Orchestrator Agent X"]
    X -->|Final action and trace| Orch
    Orch -->|Prediction request| S["World Model S"]
    S -->|World prediction| Orch
    Orch -->|Prediction request| G["Goal Model G"]
    G -->|Goal prediction| Orch
    Orch -->|ActionSpec| Env
    Orch <-->|Resolve state refs and commit transition| M["Persistent Memory M"]
    Orch -->|Trace and outcome| P["Updater P"]
    P -->|Updated context| Orch
    Orch -->|Context records| M
```

## Persistence Gate

```mermaid
flowchart TB
    S["World Model S"] -->|predicted_description| Orch["Orchestration"]
    G["Goal Model G"] -->|predicted_description| Orch
    X["Agent X"] -->|trace and action| Orch
    P["Updater P"] -->|context updates| Orch
    Orch <-->|committed run history and state refs| M["M: m_states"]
```
