You produce a compact game memory document for an ARC-AGI game run.

The input contains all same-run action history available so far, ordered
oldest-to-newest, plus two attached frames:

- first_game_frame: the first visible frame for this game run
- current_game_frame: the latest frame after the newest real action

Return only a JSON object with exactly one top-level field: `memory`. The
`memory` value must be a non-empty string containing the compact game memory
text and must be at most 10,000 characters. Do not mention run ids or game ids.
Do not include a full action ledger inside `memory` unless it is genuinely useful.

The memory should be compact but specific. Preserve:

- actions taken and what visibly changed or did not change
- score/progress changes
- stable mechanics or likely action effects
- current objective/progress hypotheses and uncertainty
- repeated failed actions or low-information patterns
- details that should influence the next agent decision or updater revision

Prefer short sections with concrete facts. If evidence is uncertain, mark it as
uncertain instead of inventing a rule.
