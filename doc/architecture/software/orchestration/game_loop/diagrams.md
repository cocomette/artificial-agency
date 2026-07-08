# Game Loop Diagrams

## State-Machine Block Diagram

```mermaid
flowchart TD
    Start([START_RUN]) --> Load[LOAD_FRAME_BUFFER]
    Load --> Enter[ENTER_FRAME_TURN]
    Enter --> Control{Controllable final frame?}
    Control -- yes --> Build[BUILD_X_INPUT]
    Build --> CallX[CALL_X]
    CallX --> ValidateReal[Require real environment action]
    Control -- no --> Synthetic[SYNTHESIZE_NONE]
    Synthetic --> PredictBuffered["RUN_WORLD_PREDICTION<br/>S with NONE"]

    PredictBuffered --> SkipEnv[Skip environment step]
    SkipEnv --> UpdateBuffered["RUN_UPDATER<br/>actual = next buffered frame"]

    ValidateReal --> Predict["RUN_WORLD_PREDICTION<br/>S from current frame"]
    Predict --> StepEnv["Call environment.step(action)"]
    StepEnv --> NewBundle[Receive next EnvironmentObservationBundle]
    NewBundle --> UpdateFresh["RUN_UPDATER<br/>actual = first new frame"]

    UpdateBuffered --> Persist[PERSIST_TURN]
    UpdateFresh --> Persist
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
    participant X as Orchestrator Agent X
    participant P as Updater P

    Orch->>Env: reset or step(real action)
    Env-->>Orch: EnvironmentObservationBundle(frames, info, actions)
    Orch->>Orch: load FrameUnrollBuffer

    loop each FrameTurn in buffer
        Orch->>M: persist or load current real frame

        alt non-final frame
            Orch->>Orch: synthesize DecisionResult(final_action=NONE)
            Orch->>Orch: run world prediction
            Orch->>Orch: skip environment step
            Orch->>P: compare current frame with next buffered frame
        else final frame
            Orch->>X: decide(agent context, frame, action_space)
            X-->>Orch: DecisionResult(final_action=real action)
            Orch->>Orch: run world prediction
            Orch->>Env: step(real action)
            Env-->>Orch: next EnvironmentObservationBundle
            Orch->>P: compare description predictions and final frame with first new frame
        end

        P-->>Orch: update result
        Orch->>M: persist trace, transition, update refs
    end
```
