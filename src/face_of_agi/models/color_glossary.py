"""Prompt-facing ARC symbol color glossary."""

from __future__ import annotations

ARC_COLOR_NAMES: dict[int, str] = {
    0: "white",
    1: "light gray",
    2: "gray",
    3: "dark gray",
    4: "charcoal",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "light cyan",
    11: "yellow",
    12: "orange",
    13: "dark red",
    14: "green",
    15: "purple",
}


def arc_color_glossary_text() -> str:
    """Render the canonical ARC symbol-to-color glossary for prompts."""

    symbols = _toolkit_palette_symbols()
    lines = ["## ARC color glossary", ""]
    lines.extend(
        f"- symbol {symbol:X}: {ARC_COLOR_NAMES[symbol]}"
        for symbol in symbols
    )
    return "\n".join(lines)


def append_arc_color_glossary(instructions: str) -> str:
    """Append the canonical ARC color glossary to role instructions."""

    return instructions.strip() + "\n\n" + arc_color_glossary_text()


def _toolkit_palette_symbols() -> tuple[int, ...]:
    """Return ARC palette symbols from the installed renderer."""

    try:
        from arc_agi.rendering import COLOR_MAP
    except Exception:
        return tuple(sorted(ARC_COLOR_NAMES))
    expected = set(range(16))
    actual = set(COLOR_MAP)
    if not expected.issubset(actual):
        missing = ", ".join(f"{symbol:X}" for symbol in sorted(expected - actual))
        raise ValueError(f"ARC renderer COLOR_MAP is missing symbols: {missing}")
    return tuple(symbol for symbol in sorted(COLOR_MAP) if symbol in ARC_COLOR_NAMES)
