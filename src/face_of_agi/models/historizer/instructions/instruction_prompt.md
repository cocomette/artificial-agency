You summarize how the agent's game-specific context evolved across prior
updater outputs.

The input contains complete prior agent game contexts ordered oldest-to-newest.
Each context has the fields `goals`, `game_mechanics`, `policy`, `history`,
and `extras`.

Return exactly one JSON object:

{"field_evolution":{"goals":"","game_mechanics":"","policy":"","history":"","extras":""}}

The `field_evolution` object must contain exactly these five string fields:
`goals`, `game_mechanics`, `policy`, `history`, `extras`.

Use enough detail to explain trend-level evolution without turning the response
into a chronological log.

For each field, summarize what changed, what stayed stable, what became more
or less certain, and what old assumptions were replaced. If the field did not
meaningfully change, say that directly.
