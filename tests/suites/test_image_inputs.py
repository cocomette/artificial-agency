"""Tests for shared model image input helpers."""

from PIL import Image
import pytest

from face_of_agi.models.image_inputs import (
    cumulative_changed_pixel_masks,
    draw_scaled_cumulative_mask_edges,
)


def test_cumulative_changed_pixel_masks_accumulate_raw_pair_diffs() -> None:
    first = Image.new("RGB", (6, 6), color=(0, 0, 0))
    middle = first.copy()
    middle.putpixel((1, 1), (255, 0, 0))
    final = middle.copy()
    final.putpixel((4, 4), (0, 0, 255))

    masks = cumulative_changed_pixel_masks(first, middle, final)

    assert [_mask_points(mask) for mask in masks] == [
        {(1, 1)},
        {(1, 1)},
        {(1, 1), (4, 4)},
    ]


def test_cumulative_changed_pixel_masks_do_not_accumulate_dilation() -> None:
    first = Image.new("RGB", (6, 6), color=(0, 0, 0))
    middle = first.copy()
    middle.putpixel((2, 2), (255, 0, 0))
    final = middle.copy()
    final.putpixel((4, 2), (0, 0, 255))

    masks = cumulative_changed_pixel_masks(first, middle, final)

    assert [_mask_points(mask) for mask in masks] == [
        {(2, 2)},
        {(2, 2)},
        {(2, 2), (4, 2)},
    ]


def test_cumulative_changed_pixel_masks_require_same_sized_images() -> None:
    with pytest.raises(ValueError, match="same-sized images"):
        cumulative_changed_pixel_masks(
            Image.new("RGB", (4, 4)),
            Image.new("RGB", (5, 4)),
        )


def test_draw_scaled_cumulative_mask_edges_draws_first_mask_on_first_frame() -> None:
    source_images = (
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
    )
    target_images = (
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
        Image.new("RGB", (5, 5), color=(255, 0, 0)),
    )

    previous_image, current_image = draw_scaled_cumulative_mask_edges(
        source_images=source_images,
        target_images=target_images,
        frame_masks=(
            _mask_image((5, 5), {(2, 2)}),
            _mask_image((5, 5), {(2, 2)}),
        ),
        dilation_kernel_size=1,
        line_width=1,
    )

    assert previous_image.getpixel((2, 2)) == (255, 0, 255)
    assert current_image.getpixel((2, 2)) == (255, 0, 255)
    assert previous_image.getpixel((1, 1)) == (0, 0, 0)
    assert current_image.getpixel((1, 1)) == (255, 0, 0)


def test_draw_scaled_cumulative_mask_edges_dilates_only_while_drawing() -> None:
    source_images = (
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
    )
    target_images = (
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
    )

    previous_image, current_image = draw_scaled_cumulative_mask_edges(
        source_images=source_images,
        target_images=target_images,
        frame_masks=(
            _mask_image((5, 5), {(2, 2)}),
            _mask_image((5, 5), {(2, 2)}),
        ),
        dilation_kernel_size=3,
        line_width=1,
    )

    for image in (previous_image, current_image):
        assert image.getpixel((1, 1)) == (255, 0, 255)
        assert image.getpixel((3, 3)) == (255, 0, 255)
        assert image.getpixel((2, 2)) == (0, 0, 0)


def test_draw_scaled_cumulative_mask_edges_extracts_edges_after_scaling() -> None:
    source_images = (
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
        Image.new("RGB", (5, 5), color=(0, 0, 0)),
    )
    target_images = (
        Image.new("RGB", (25, 25), color=(0, 0, 0)),
        Image.new("RGB", (25, 25), color=(0, 0, 0)),
    )

    previous_image, current_image = draw_scaled_cumulative_mask_edges(
        source_images=source_images,
        target_images=target_images,
        frame_masks=(
            _mask_image((5, 5), {(2, 2)}),
            _mask_image((5, 5), {(2, 2)}),
        ),
        dilation_kernel_size=1,
        line_width=1,
    )

    for image in (previous_image, current_image):
        assert image.getpixel((10, 10)) == (255, 0, 255)
        assert image.getpixel((14, 14)) == (255, 0, 255)
        assert image.getpixel((12, 12)) == (0, 0, 0)


def _mask_image(
    size: tuple[int, int],
    points: set[tuple[int, int]],
) -> Image.Image:
    image = Image.new("L", size, 0)
    for point in points:
        image.putpixel(point, 255)
    return image


def _mask_points(mask: Image.Image) -> set[tuple[int, int]]:
    pixels = mask.load()
    width, height = mask.size
    return {
        (x, y)
        for y in range(height)
        for x in range(width)
        if pixels[x, y] != 0
    }
