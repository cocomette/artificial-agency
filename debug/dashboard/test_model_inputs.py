"""Dashboard-local validation for model input annotation helpers."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from debug.dashboard.model_inputs import prediction_overlay


def test_prediction_overlay_draws_bbox_2d_arrays_from_provider_output() -> None:
    record = _world_record(
        response_text='[{"bbox_2d": [10, 5, 50, 25], "description": "target"}]',
        image_size=(100, 50),
        coordinate_space="pixel",
    )

    overlay = prediction_overlay(record, display_size=(100, 50))

    assert overlay.drawn_count == 1
    assert overlay.area_count == 1
    assert overlay.warnings == ()
    assert overlay.image is not None
    assert overlay.image.getpixel((10, 5)) == (0, 255, 0)


def test_prediction_overlay_draws_openai_items_with_normalized_bbox_2d() -> None:
    record = _world_record(
        response_text=(
            '{"items": [{"bbox_2d": [100, 200, 300, 600], '
            '"description": "target"}]}'
        ),
        image_size=(100, 50),
        coordinate_space="normalized_1000",
    )

    overlay = prediction_overlay(record, display_size=(100, 50))

    assert overlay.drawn_count == 1
    assert overlay.area_count == 1
    assert overlay.warnings == ()
    assert overlay.image is not None
    assert overlay.image.getpixel((10, 10)) == (0, 255, 0)


def test_prediction_overlay_draws_predicted_description_wrapper() -> None:
    record = _world_record(
        response_text=(
            '{"predicted_description": [{"bbox_2d": [10, 5, 50, 25], '
            '"description": "target"}]}'
        ),
        image_size=(100, 50),
        coordinate_space="pixel",
    )

    overlay = prediction_overlay(record, display_size=(100, 50))

    assert overlay.drawn_count == 1
    assert overlay.area_count == 1
    assert overlay.warnings == ()


def _world_record(
    *,
    response_text: str,
    image_size: tuple[int, int],
    coordinate_space: str,
) -> dict[str, object]:
    return {
        "request": {
            "input": [
                {
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": _tiny_png_data_url(size=image_size),
                        }
                    ],
                }
            ]
        },
        "metadata": {
            "response_output_text": response_text,
            "visual_coordinate_space": coordinate_space,
        },
    }


def _tiny_png_data_url(*, size: tuple[int, int]) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
