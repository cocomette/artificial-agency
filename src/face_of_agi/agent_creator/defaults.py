"""Default dynamic updater roles used to bootstrap the creator store."""

from __future__ import annotations

from pathlib import Path

from face_of_agi.agent_creator.contracts import AgentRoleDefinition

UPDATER_INSTRUCTION_DIR = (
    Path(__file__).parents[1] / "models" / "updater" / "instructions"
)
DEFAULT_GENERAL_AGENT_SYSTEM_PROMPT_PATH = (
    UPDATER_INSTRUCTION_DIR / "agent_role_context_updater_prompt.md"
)


def default_agent_roles() -> tuple[AgentRoleDefinition, ...]:
    """Return the initial probing and policy role definitions."""

    return (
        AgentRoleDefinition(
            role="probing",
            meta_description=(
                "Tests uncertain mechanics, explores under-tested actions, and "
                "helps recover from stuck or oscillating behavior."
            ),
            role_instructions=_guidance_after_marker(
                UPDATER_INSTRUCTION_DIR / "agent_probing_context_updater_prompt.md"
            ),
        ),
        AgentRoleDefinition(
            role="policy",
            meta_description=(
                "Pursues the current objective, updates goal hypotheses, and "
                "chooses actions that should progress toward solving the game."
            ),
            role_instructions=_guidance_after_marker(
                UPDATER_INSTRUCTION_DIR / "agent_policy_context_updater_prompt.md"
            ),
        ),
    )


def default_general_agent_system_prompt() -> str:
    """Load the shared generic agent-updater system prompt."""

    return DEFAULT_GENERAL_AGENT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _guidance_after_marker(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    marker = "## Guidance"
    if marker not in text:
        raise RuntimeError(f"default role prompt is missing {marker}: {path}")
    return (marker + text.split(marker, 1)[1]).strip()
