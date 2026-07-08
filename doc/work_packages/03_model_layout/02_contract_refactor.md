# Step 02: Contract Refactor

## Objective

Keep shared runtime data contracts global while moving model-specific Protocols
into their owning model folders.

## Implementation

- Remove world, goal, agent, and updater Protocols from `face_of_agi.contracts`.
- Define role-specific Protocols in the matching model packages.
- Update exports so callers can import model roles from `face_of_agi.models`.
- Keep shared dataclasses such as observations, actions, traces, and context
  documents in `face_of_agi.contracts`.

## Dependencies

Depends on Step 01.

## Acceptance Check

- No model role Protocol is defined in `face_of_agi.contracts`.
