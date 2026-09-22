"""Shared evaluator for translated branching-logic expressions.

The branching-logic translator intentionally produces Python expressions. This
module preserves that architecture while ensuring active consumers use the
same compile cache and deliberately small global namespace. It is not a
general-purpose sandbox; it prevents translated metadata from receiving
Python's default builtins such as ``open`` and ``__import__``.
"""

from __future__ import annotations

import ast
from types import CodeType
from typing import Any


_COMPILED_BRANCHING_LOGIC: dict[str, CodeType] = {}

# ``getattr`` is needed by the hand-maintained SCID diagnosis conditions. The
# other functions are emitted by TransformBranchingLogic. Supplying an empty
# ``__builtins__`` mapping is essential: otherwise eval inserts the process-wide
# builtins automatically.
_BRANCHING_LOGIC_GLOBALS = {
    "__builtins__": {},
    "bool": bool,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "str": str,
}


class BranchingLogicValidationError(ValueError):
    """A translated expression is outside the supported legacy subset."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)
_ALLOWED_FUNCTION_NAMES = {"bool", "float", "getattr", "hasattr", "str"}


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def validate_branching_logic(
        expression: str, *, allow_legacy_self: bool = True) -> ast.Expression:
    """Validate the Python emitted by the existing regex translator.

    This does not translate or reinterpret REDCap syntax. It only verifies that
    the resulting Python is the narrow expression language actually emitted by
    this project. In particular, bare event/field names and arbitrary function
    calls are rejected before they can reach ``eval``.
    """

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise BranchingLogicValidationError(
            "syntax_error", f"{exc.msg} at line {exc.lineno}, column {exc.offset}"
        ) from exc

    allowed_names = {
        "curr_row", "instance", "bool", "float", "getattr", "hasattr", "str"
    }
    if allow_legacy_self:
        allowed_names.add("self")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise BranchingLogicValidationError(
                "unsupported_syntax", type(node).__name__)

        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise BranchingLogicValidationError(
                "unbound_name", node.id)

        if isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if path is None or any(part.startswith("_") for part in path[1:]):
                raise BranchingLogicValidationError(
                    "unsafe_attribute", ast.unparse(node))
            if path[0] == "curr_row" and len(path) == 2:
                continue
            checker_roots = {"instance"}
            if allow_legacy_self:
                checker_roots.add("self")
            if path[0] in checker_roots and path[1:] in {
                    ("missing_code_list",),
                    ("utils",),
                    ("utils", "can_be_float"),
                    ("utils", "missing_code_list"),
            }:
                continue
            raise BranchingLogicValidationError(
                "unsafe_attribute", ".".join(path))

        if isinstance(node, ast.Call):
            if node.keywords:
                raise BranchingLogicValidationError(
                    "unsupported_call", "keyword arguments are not allowed")
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_FUNCTION_NAMES:
                    raise BranchingLogicValidationError(
                        "unsupported_call", node.func.id)
                if node.func.id in {"getattr", "hasattr"}:
                    if (len(node.args) < 2
                            or not isinstance(node.args[0], ast.Name)
                            or node.args[0].id != "curr_row"
                            or not isinstance(node.args[1], ast.Constant)
                            or not isinstance(node.args[1].value, str)
                            or node.args[1].value.startswith("_")):
                        raise BranchingLogicValidationError(
                            "unsafe_attribute_lookup", ast.unparse(node))
            elif _attribute_path(node.func) not in {
                    ("instance", "utils", "can_be_float"),
                    ("self", "utils", "can_be_float"),
            }:
                raise BranchingLogicValidationError(
                    "unsupported_call", ast.unparse(node.func))

    return tree


def compile_branching_logic(expression: str) -> CodeType:
    """Compile *expression* once and return its cached code object."""

    cached = _COMPILED_BRANCHING_LOGIC.get(expression)
    if cached is None:
        tree = validate_branching_logic(expression)
        cached = compile(tree, "<branching_logic>", "eval")
        _COMPILED_BRANCHING_LOGIC[expression] = cached
    return cached


def evaluate_branching_logic(
        expression: str | CodeType, *, curr_row: Any, instance: Any) -> bool:
    """Evaluate translated branching logic with explicit row/checker locals.

    Both ``instance`` and ``self`` refer to the checker because generated
    expressions use the former while legacy SCID conditions use the latter.
    """

    code = (
        compile_branching_logic(expression)
        if isinstance(expression, str)
        else expression
    )
    return bool(eval(  # noqa: S307 - restricted globals by design
        code,
        _BRANCHING_LOGIC_GLOBALS,
        {"curr_row": curr_row, "instance": instance, "self": instance},
    ))
