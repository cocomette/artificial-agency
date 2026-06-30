"""Tests for model-facing ARC image input helpers."""

from __future__ import annotations

from face_of_agi.contracts import Observation
from face_of_agi.models.image_inputs import (
    image_crop_size,
    observation_to_cropped_image,
    vllm_text_image_content,
)
from face_of_agi.models.observation_text import ObservationTextConfig


def _grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]


def test_observation_image_crop_matches_observation_text_crop() -> None:
    frame = _grid()
    frame[2][2] = 8
    config = ObservationTextConfig(crop_cells=2)

    image = observation_to_cropped_image(
        Observation(id="obs-1", step=0, frame=frame),
        observation_text_config=config,
        frame_scale=3,
        size=None,
    )

    assert image.size == (180, 180)
    assert image_crop_size(config, frame_scale=3, input_image_size=None) == (180, 180)


def test_vllm_text_image_content_uses_png_data_urls() -> None:
    image = observation_to_cropped_image(
        Observation(id="obs-1", step=0, frame=_grid()),
        observation_text_config=ObservationTextConfig(crop_cells=3),
        frame_scale=4,
        size="32x32",
    )

    content = vllm_text_image_content("prompt text", (image,), detail="auto")

    assert content[0] == {"type": "text", "text": "prompt text"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["detail"] == "auto"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
