"""Game memory model role."""

from face_of_agi.models.memory.adapter import (
    DisabledGameMemoryAdapter,
    GameMemoryAdapter,
    GameMemoryOutputError,
    build_game_memory_repair_prompt,
    build_game_memory_prompt,
    game_memory_images,
    load_game_memory_instructions,
    parse_game_memory_output,
)
from face_of_agi.models.memory.config import (
    GameMemoryConfig,
    OllamaGameMemoryConfig,
    OpenAIGameMemoryConfig,
    VLLMGameMemoryConfig,
)
from face_of_agi.models.memory.contracts import (
    GAME_MEMORY_MAX_CHARS,
    GameMemoryDocument,
    GameMemoryInput,
    GameMemoryModel,
    PromptGameMemoryImage,
    PromptGameMemoryProvider,
    PromptGameMemoryProviderResponse,
    PromptGameMemoryRequest,
    game_memory_json_schema,
    openai_game_memory_text_format,
)

__all__ = [
    "DisabledGameMemoryAdapter",
    "GameMemoryAdapter",
    "GameMemoryConfig",
    "GameMemoryDocument",
    "GameMemoryInput",
    "GAME_MEMORY_MAX_CHARS",
    "GameMemoryModel",
    "GameMemoryOutputError",
    "OllamaGameMemoryConfig",
    "OpenAIGameMemoryConfig",
    "PromptGameMemoryImage",
    "PromptGameMemoryProvider",
    "PromptGameMemoryProviderResponse",
    "PromptGameMemoryRequest",
    "VLLMGameMemoryConfig",
    "build_game_memory_repair_prompt",
    "build_game_memory_prompt",
    "game_memory_images",
    "game_memory_json_schema",
    "load_game_memory_instructions",
    "openai_game_memory_text_format",
    "parse_game_memory_output",
]
