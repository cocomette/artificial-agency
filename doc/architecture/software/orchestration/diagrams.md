# Orchestration Diagrams

```mermaid
flowchart TD
    Env[EnvironmentAdapter] --> Orch[GameLoopStateMachine]
    Agent[OnlineLearnerAgent] <--> Orch
    Orch --> M[StateMemory]
    Orch --> Debug[DebugBus]
```
