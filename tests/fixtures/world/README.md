# World Model Fixtures

These fixtures are real ARC-AGI frames rendered with
`arc_agi.rendering.frame_to_rgb_array(scale=4)`.

## LS20 Seed 0 Action 1

- Game: `ls20-9607627b`
- Source: reset observation, step `0`
- Action: `GameAction.ACTION1`
- Target: next observation after applying `ACTION1`, step `1`

## SB26 Seed 0 Action 5

- Game: `sb26-7fbdac44`
- Source: reset observation, step `0`
- Action: `GameAction.ACTION5`
- Target: frame index `1` from the 42-frame observation returned after applying
  `ACTION5`, step `1`

The fixture images are committed so the manual world-model E2E check can run
without requiring local ARC environment files.
