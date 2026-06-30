# Game Loop Diagrams

## State-Machine Block Diagram

```mermaid
flowchart TD
    Start([START_RUN]) --> Load[LOAD_FRAME_BUFFER]
    Load --> Enter[ENTER_FRAME_TURN]
    Enter --> Control{Controllable final frame?}
    Control -- yes --> Build[BUILD_X_INPUT]
    Build --> CallX[CALL_X]
    CallX --> Resolve[RESOLVE_NEXT_SNAPSHOT]
    Control -- no --> Synthetic[SYNTHESIZE_NONE]
    Synthetic --> Resolve

    Resolve --> Change[SUMMARIZE_CHANGE]
    Change --> Hist[SUMMARIZE_AGENT_CONTEXT_HISTORY]
    Hist --> Update[RUN_UPDATER]
    Update --> Persist[PERSIST_TURN]
    Persist --> MoreFrames{More frames in buffer?}

    MoreFrames -- yes --> Enter
    MoreFrames -- no --> Lifecycle{Terminal lifecycle?}
    Lifecycle -- no --> Load
    Lifecycle -- GAME_WIN --> Win([GAME_WIN])
    Lifecycle -- GAME_OVER --> Reset([GAME_OVER_RESET])
    Lifecycle -- ACTION_LIMIT --> Limit([ACTION_LIMIT_REACHED])
    Lifecycle -- ERROR --> Error([ERROR])
    Reset --> Start
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Env as Environment Adapter
    participant Orch as Orchestration
    participant M as State Memory M
    participant X as Agent X
    participant C as Change Summary
    participant H as Historizer
    participant P as Updater P

    Orch->>Env: reset or step(real action)
    Env-->>Orch: EnvironmentObservationBundle(frames, info, actions)
    Orch->>Orch: load FrameUnrollBuffer

    loop each FrameTurn in buffer
        Orch->>M: prewrite or load current real frame

        alt non-final frame
            Orch->>Orch: synthesize DecisionResult(final_action=NONE)
            Orch->>Orch: skip environment step
            Orch->>Orch: actual next = next buffered frame
        else final frame
            Orch->>X: decide(text observations, context, action_space)
            X-->>Orch: DecisionResult(final_action=real action)
            Orch->>Env: step(real action)
            Env-->>Orch: next EnvironmentObservationBundle
            Orch->>Orch: actual next = first new frame
        end

        Orch->>C: summarize transition text
        C-->>Orch: ChangeSummaryResult
        Orch->>H: summarize recent context history
        H-->>Orch: AgentContextHistorySummary
        Orch->>P: update agent context
        P-->>Orch: update result
        Orch->>M: persist trace, summary, metrics, context
    end
```
