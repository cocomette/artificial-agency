# Shared Contracts Overview

The shared contracts module owns typed data boundaries used between modules.
Contracts should stay small and provider-neutral.

Shared contracts are not a business-logic layer. They define the shape of data
that environment, orchestration, memory, models, runtime, and updates exchange.

## Contract Goals

- Keep module boundaries explicit.
- Avoid provider-specific model types in orchestration.
- Avoid ARC toolkit leakage outside the environment boundary except where the
  project intentionally preserves ARC-native values.
- Give memory stable record payloads without freezing a final database schema
  too early.
