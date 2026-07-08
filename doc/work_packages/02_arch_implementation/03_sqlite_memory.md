# Step 03: SQLite Memory

## Objective

Add the first SQLite-backed memory foundation without committing to the final
storage schema.

## Implementation

- Add a small `SQLiteDatabase` wrapper.
- Initialize separate generic tables for durable state memory `M` and
  experimental memory `E`.
- Implement generic write/list helpers for each memory domain.
- Store payloads as JSON text for now.
- Keep tests on temporary database paths only.

## Dependencies

Depends on Step 02.

## Acceptance Check

- Tests can initialize a temporary database and write/list records in both
  memory domains.
