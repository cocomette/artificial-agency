# Step 02: Adapter Signal Passthrough

## Objective

Keep ARC-AGI framework interaction behind one thin adapter layer.

## Implementation

- Add game selection through `arc_agi.Arcade.make(...)`.
- Keep raw ARC actions unchanged by exposing `arcengine.GameAction`.
- Surface lifecycle state through `arcengine.GameState`.
- Normalize incoming `arcengine.FrameDataRaw` into the local `Observation`
  contract only as much as needed for the starter shell.

## Parallelism

This step can run in parallel with Step 01.

## Acceptance Check

- The adapter is the only direct environment integration point.
- Real ARC toolkit actions, state enums, and frame data are available above the
  adapter.
