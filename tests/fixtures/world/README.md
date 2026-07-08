# World Model Fixtures

These fixtures are a real ARC-AGI transition rendered with
`arc_agi.rendering.frame_to_rgb_array(scale=4)`.

- Game: `ls20-9607627b`
- Seed: `0`
- Source: reset observation, step `0`
- Action: `GameAction.ACTION1`
- Target: next observation after applying `ACTION1`, step `1`

The fixture images are committed so the manual world-model E2E check can run
without requiring local ARC environment files.
