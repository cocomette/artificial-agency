"""Helpers for deriving action hints from local ARC game source."""

from __future__ import annotations

import ast
from pathlib import Path


def resolve_cheat_action_context_game_dir(
    *,
    environments_dir: str | Path,
    game_id: str,
) -> Path:
    """Infer the local game source directory for an ARC game id."""

    if "-" not in game_id:
        raise RuntimeError(f"cannot infer local game directory from game id: {game_id}")
    game_name, game_hash = game_id.split("-", 1)
    return Path(environments_dir) / game_name / game_hash


def load_cheat_action_context(game_dir: str | Path) -> str:
    """Generate keyboard action mappings from the local game implementation."""

    source_path = _game_source_path(Path(game_dir))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    axis_vars = _extract_player_axis_vars(tree)
    action_deltas = _extract_action_deltas(tree, axis_vars)
    return "\n".join(_format_action_keyboard_lines(action_deltas))


def _game_source_path(game_dir: Path) -> Path:
    """Return the single Python game file from a local game directory."""

    game_files = sorted(path for path in game_dir.glob("*.py") if path.is_file())
    if len(game_files) != 1:
        raise RuntimeError(
            f"expected exactly one Python game file in {game_dir}, found {game_files}"
        )
    return game_files[0]


def _extract_player_axis_vars(tree: ast.AST) -> dict[str, str]:
    """Find the obfuscated dx/dy variable names used for player movement."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not node.targets or not isinstance(node.targets[0], ast.Tuple):
            continue
        if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2:
            continue
        x_var = _movement_var_from_expr(node.value.elts[0], size_attr="gisrhqpee")
        y_var = _movement_var_from_expr(node.value.elts[1], size_attr="tbwnoxqgc")
        if x_var and y_var:
            return {"x": x_var, "y": y_var}
    raise RuntimeError("could not extract player movement axis variables")


def _movement_var_from_expr(expr: ast.AST, *, size_attr: str) -> str | None:
    """Return the movement variable multiplied by one player dimension."""

    for node in ast.walk(expr):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        if _contains_self_attr(node.left, size_attr):
            return _first_non_self_name(node.right)
        if _contains_self_attr(node.right, size_attr):
            return _first_non_self_name(node.left)
    return None


def _first_non_self_name(expr: ast.AST) -> str | None:
    """Return the first plain variable name from an expression."""

    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id != "self":
            return node.id
    return None


def _contains_self_attr(expr: ast.AST, attr: str) -> bool:
    """Return whether an expression references self.<attr>."""

    for node in ast.walk(expr):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == attr
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return True
    return False


def _extract_action_deltas(
    tree: ast.AST,
    axis_vars: dict[str, str],
) -> dict[str, tuple[int, int]]:
    """Extract GameAction branch deltas from the game's step method."""

    action_deltas: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        action_name = _game_action_name(node.test)
        if action_name is None:
            continue
        assignments = _constant_assignments(node.body)
        dx = int(assignments.get(axis_vars["x"], 0))
        dy = int(assignments.get(axis_vars["y"], 0))
        action_deltas[action_name] = (dx, dy)
    if not action_deltas:
        raise RuntimeError("could not extract GameAction movement branches")
    return dict(sorted(action_deltas.items()))


def _game_action_name(test: ast.AST) -> str | None:
    """Return ACTIONn from a GameAction comparison expression."""

    if not isinstance(test, ast.Compare):
        return None
    for comparator in test.comparators:
        if (
            isinstance(comparator, ast.Attribute)
            and comparator.attr.startswith("ACTION")
            and isinstance(comparator.value, ast.Name)
            and comparator.value.id == "GameAction"
        ):
            return comparator.attr
    return None


def _constant_assignments(nodes: list[ast.stmt]) -> dict[str, int | bool]:
    """Return simple name = constant assignments from a branch body."""

    assignments: dict[str, int | bool] = {}
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        value = _literal_int_or_bool(node.value)
        if value is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    return assignments


def _literal_int_or_bool(node: ast.AST) -> int | bool | None:
    """Return simple integer/bool literals, including negative integers."""

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def _format_action_keyboard_lines(
    action_deltas: dict[str, tuple[int, int]],
) -> list[str]:
    """Format extracted action deltas as keyboard buttons for the model prompt."""

    return [
        f"{action_name}: {_keyboard_button_text(dx, dy)}"
        for action_name, (dx, dy) in action_deltas.items()
    ]


def _keyboard_button_text(dx: int, dy: int) -> str:
    """Describe an ARC movement delta as a keyboard button."""

    buttons = {
        (0, -1): "up arrow",
        (0, 1): "down arrow",
        (-1, 0): "left arrow",
        (1, 0): "right arrow",
        (0, 0): "no-op",
    }
    return buttons.get((dx, dy), f"movement key for delta ({dx}, {dy})")
