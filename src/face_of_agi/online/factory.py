"""Runtime assembly helpers for the online learner."""

from __future__ import annotations

from face_of_agi.environment.config import AgentRuntimeConfig
from face_of_agi.online.agent import OnlineLearnerAgent
from face_of_agi.online.backbone import DeterministicBackbone, TransformersBackbone


def build_online_agent(config: AgentRuntimeConfig) -> OnlineLearnerAgent:
    """Build the configured online learner agent."""

    backend = config.backbone.backend
    if backend == "deterministic":
        backbone = DeterministicBackbone(feature_dim=config.online.hidden_dim)
    elif backend == "transformers":
        backbone = TransformersBackbone(
            config.backbone,
            feature_dim=config.online.hidden_dim,
        )
    else:
        raise ValueError(f"unknown agent.backbone.backend: {backend}")
    return OnlineLearnerAgent(config=config, backbone=backbone)
