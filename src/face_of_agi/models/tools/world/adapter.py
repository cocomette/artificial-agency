"""Framework adapter for world model tool S."""

from __future__ import annotations

from face_of_agi.models.tools.world.providers.huggingface import (
    HuggingFaceWorldToolAdapter,
)


class WorldToolAdapter(HuggingFaceWorldToolAdapter):
    """Default world tool adapter backed by Hugging Face/Diffusers."""
