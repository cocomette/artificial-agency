"""Visualization config helpers for the starter ARC shell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

from arcengine import FrameDataRaw

ArcRenderer = Callable[[int, FrameDataRaw], None]


@dataclass(slots=True)
class VisualizationSettings:
    """Resolved visualization settings for one ARC environment wrapper."""

    render_mode: str | None
    renderer: ArcRenderer | None


def resolve_visualization(
    *,
    enabled: bool,
    render_mode: str | None,
) -> VisualizationSettings:
    """Resolve the ARC toolkit visualization strategy for the starter shell."""

    if not enabled:
        return VisualizationSettings(render_mode=None, renderer=None)

    selected_mode = (render_mode or "human").strip()
    _prepare_matplotlib_backend(selected_mode)

    if selected_mode == "human":
        return VisualizationSettings(
            render_mode=None,
            renderer=PersistentHumanRenderer(),
        )

    if selected_mode in {"terminal", "terminal-fast"}:
        return VisualizationSettings(render_mode=selected_mode, renderer=None)

    raise ValueError(
        "visualization render_mode must be one of: "
        "'human', 'terminal', 'terminal-fast'"
    )


def _prepare_matplotlib_backend(render_mode: str) -> None:
    """Set a sensible matplotlib backend before ARC imports pyplot."""

    if render_mode != "human":
        return

    if os.environ.get("MPLBACKEND"):
        return

    os.environ["MPLBACKEND"] = "TkAgg"


class PersistentHumanRenderer:
    """Keep one matplotlib window open and update it in place.

    ARC's built-in `render_mode="human"` opens a figure for one response and
    closes it right away. For the starter shell we keep the last frame visible
    until the next response arrives, which makes the game much easier to watch.
    """

    def __init__(self, *, scale: int = 4, frame_delay_seconds: float = 0.1) -> None:
        self.scale = scale
        self.frame_delay_seconds = frame_delay_seconds
        self._figure = None
        self._axis = None
        self._image = None

    def __call__(self, steps: int, frame_data: FrameDataRaw) -> None:
        """Render every outgoing frame and keep the last one on screen."""

        frames = tuple(frame_data.frame)
        if not frames:
            return

        self._ensure_window()

        from arc_agi.rendering import frame_to_rgb_array
        import matplotlib.pyplot as plt

        for frame in frames:
            image = frame_to_rgb_array(
                steps=steps,
                frame=frame,
                scale=self.scale,
            )

            if self._image is None:
                self._image = self._axis.imshow(image, interpolation="nearest")
                self._figure.tight_layout()
                plt.show(block=False)
            else:
                self._image.set_data(image)

            self._axis.set_title(
                f"ARC-AGI-3 Environment | step {steps} | {frame_data.state.name}"
            )
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            plt.pause(self.frame_delay_seconds)

    def _ensure_window(self) -> None:
        """Create the persistent matplotlib window on first use or after close."""

        import matplotlib.pyplot as plt

        if self._figure is not None and plt.fignum_exists(self._figure.number):
            return

        plt.ion()
        self._figure, self._axis = plt.subplots(figsize=(8, 8))
        self._axis.axis("off")
        self._image = None
