"""Translate REDCap calculated-field expressions into validated Python.

This module is intentionally separate from ``transform_branching_logic``.
Calculated fields contain nested functions, arithmetic, quoted date literals,
and longitudinal ``[event][field]`` references; ordered regular-expression
substitutions cannot preserve those semantics reliably.

The core translator is configuration-independent and can be imported in tests.
The command-line program reads a REDCap data dictionary and writes:

* ``converted_calculated_fields.json`` -- complete conversion artifact;
* ``converted_calculated_fields.csv`` -- human-readable audit table;
* ``excluded_calculated_field_vars.json`` -- explicit failed conversions.

Generated expressions call only the allowlisted ``redcap_*`` helpers defined in
this module. Raw data-dictionary text is never passed to ``eval``.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence, Union

import pandas as pd


VARIABLE_COLUMN = "Variable / Field Name"
FORM_COLUMN = "Form Name"
FIELD_TYPE_COLUMN = "Field Type"
CALCULATION_COLUMN = "Choices, Calculations, OR Slider Labels"
BRANCHING_LOGIC_COLUMN = "Branching Logic (Show field only if...)"
IDENTIFIER_COLUMN = "Identifier?"

SCHEMA_VERSION = 2
MAX_EXPRESSION_LENGTH = 100_000
MAX_TOKENS = 100_000
MAX_PARSE_DEPTH = 512
MAX_NUMERIC_LITERAL_LENGTH = 256
MAX_RUNTIME_NUMERIC_TEXT_LENGTH = 1_000
MAX_POWER_ABS_EXPONENT = 1_000
MAX_RESULT_INTEGER_DIGITS = 100

# REDCap coded-missing values must behave exactly like an empty field when a
# calculated expression reads imported data. This is the study Rule-H policy
# shared with C:\REDCAP_WORK\simulate_calcs.R: matching is exact text, so
# two-decimal numeric exports and every minute-precision datetime spelling are
# listed explicitly. Seconds-precision, whitespace-padded, leading-zero, and
# scientific-notation variants remain ordinary values. Formula literals keep
# their meaning while an expression is evaluated; the normalized root result
# is checked against the same policy before it can feed a dependent target.
_CALCULATION_MISSING_DATE_TEXT = (
    '1909-09-09', '1903-03-03', '1901-01-01',
)
_CALCULATION_MISSING_DATETIME_TEXT = tuple(
    f'{missing_date} {hour:02d}:{minute:02d}'
    for missing_date in _CALCULATION_MISSING_DATE_TEXT
    for hour in range(24)
    for minute in range(60)
)
CALCULATION_MISSING_CODES = (
    '-3', '-9', -3, -9, -3.0, -9.0,
    '-3.0', '-9.0', '-3.00', '-9.00',
    '-99', -99, -99.0, '-99.0', '-99.00',
    999, 999.0, '999', '999.0', '999.00',
    *_CALCULATION_MISSING_DATE_TEXT,
    *_CALCULATION_MISSING_DATETIME_TEXT,
)
_CALCULATION_MISSING_CODE_TEXT = frozenset(
    value for value in CALCULATION_MISSING_CODES
    if isinstance(value, str))
_CALCULATION_MISSING_CODE_NUMBERS = frozenset(
    value for value in CALCULATION_MISSING_CODES
    if not isinstance(value, str))

SUPPORTED_FUNCTION_ARITY = {
    "if": (3, 3),
    "sum": (1, None),
    "round": (1, 2),
    "datediff": (3, 5),
    "left": (2, 2),
    "mid": (3, 3),
    "right": (2, 2),
    "and": (1, None),
    "or": (1, None),
    "not": (1, 1),
}


class CalculationTranslationError(ValueError):
    """Base error for a calculation that cannot be translated safely."""

    code = "invalid_calculation"

    def __init__(self, message: str, position: int | None = None):
        self.position = position
        suffix = "" if position is None else f" at character {position}"
        super().__init__(f"{message}{suffix}")


class UnsupportedCalculationError(CalculationTranslationError):
    """Well-formed REDCap syntax that this translator does not support."""

    code = "unsupported_calculation"


class CalculationSchemaError(ValueError):
    """The data dictionary itself does not satisfy the input contract."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    position: int
    raw: str


@dataclass(frozen=True)
class LiteralNode:
    value: Any


@dataclass(frozen=True)
class NumberNode:
    """An exact REDCap numeric literal retained in its source spelling."""

    raw: str


@dataclass(frozen=True)
class FieldNode:
    variable: str
    event: str | None = None
    choice: str | None = None
    instance: int | None = None

    @property
    def display(self) -> str:
        field = self.variable
        if self.choice is not None:
            field += f"({self.choice})"
        display = f"[{self.event}][{field}]" if self.event else f"[{field}]"
        if self.instance is not None:
            display += f"[{self.instance}]"
        return display


@dataclass(frozen=True)
class UnaryNode:
    operator: str
    operand: Any


@dataclass(frozen=True)
class BinaryNode:
    operator: str
    left: Any
    right: Any


@dataclass(frozen=True)
class CallNode:
    function: str
    arguments: tuple[Any, ...]


# typing.Union rather than the PEP 604 "A | B" spelling: this assignment runs
# at import time, and the pipe syntax makes the whole module unimportable on
# the Python 3.9 interpreters that pair with the server's pandas 1.4.x.
Node = Union[NumberNode, LiteralNode, FieldNode, UnaryNode, BinaryNode, CallNode]


class CalculationTokenizer:
    """Turn one REDCap calculation into a strict token stream."""

    _number_re = re.compile(
        r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
    _identifier_re = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
    _two_char_operators = frozenset({"<>", "!=", "<=", ">=", "=="})
    _one_char_tokens = {
        "+": "OP", "-": "OP", "*": "OP", "/": "OP",
        "%": "OP", "^": "OP", "=": "OP", "<": "OP", ">": "OP",
        "(": "LPAREN", ")": "RPAREN", ",": "COMMA",
    }

    def tokenize(self, expression: str) -> list[Token]:
        if not isinstance(expression, str):
            raise CalculationTranslationError(
                "Calculation must be a string", 0)
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise UnsupportedCalculationError(
                f"Calculation exceeds {MAX_EXPRESSION_LENGTH:,} characters",
                MAX_EXPRESSION_LENGTH)

        tokens: list[Token] = []
        index = 0
        while index < len(expression):
            char = expression[index]
            if char.isspace():
                index += 1
                continue

            # REDCap permits end-of-line calculation comments starting with
            # ``//`` or ``#``. Only whitespace-delimited markers count: a
            # mid-token ``//`` (e.g. the ``[a]//[b]`` typo) must fail closed
            # as two division operators rather than silently deleting the
            # rest of the expression.
            if (expression.startswith("//", index)
                    or char == "#") and (
                    index == 0 or expression[index - 1].isspace()):
                skip = 2 if expression.startswith("//", index) else 1
                newline = expression.find("\n", index + skip)
                index = len(expression) if newline < 0 else newline + 1
                continue

            if char == "[":
                end = expression.find("]", index + 1)
                if end < 0:
                    raise CalculationTranslationError(
                        "Unclosed field reference", index)
                content = expression[index + 1:end].strip()
                if not content:
                    raise CalculationTranslationError(
                        "Empty field reference", index)
                tokens.append(Token(
                    "FIELD", content, index, expression[index:end + 1]))
                index = end + 1
                self._check_token_limit(tokens, index)
                continue

            if char in {"'", '"'}:
                value, end = self._read_string(expression, index)
                tokens.append(Token(
                    "STRING", value, index, expression[index:end]))
                index = end
                self._check_token_limit(tokens, index)
                continue

            two = expression[index:index + 2]
            if two in self._two_char_operators:
                tokens.append(Token("OP", two, index, two))
                index += 2
                self._check_token_limit(tokens, index)
                continue

            number = self._number_re.match(expression, index)
            if number is not None:
                raw = number.group(0)
                if len(raw) > MAX_NUMERIC_LITERAL_LENGTH:
                    raise UnsupportedCalculationError(
                        "Numeric literal exceeds "
                        f"{MAX_NUMERIC_LITERAL_LENGTH} characters", index)
                try:
                    value = Decimal(raw)
                    if not value.is_finite():
                        raise ValueError("non-finite number")
                except (InvalidOperation, ValueError) as exc:
                    raise UnsupportedCalculationError(
                        f"Unsupported numeric literal {raw!r}", index) from exc
                # Store source text, not a binary float. The compiler routes it
                # through an allowlisted Decimal-backed runtime helper.
                tokens.append(Token("NUMBER", raw, index, raw))
                index = number.end()
                self._check_token_limit(tokens, index)
                continue

            identifier = self._identifier_re.match(expression, index)
            if identifier is not None:
                raw = identifier.group(0)
                tokens.append(Token(
                    "IDENT", raw.casefold(), index, raw))
                index = identifier.end()
                self._check_token_limit(tokens, index)
                continue

            token_kind = self._one_char_tokens.get(char)
            if token_kind is not None:
                tokens.append(Token(token_kind, char, index, char))
                index += 1
                self._check_token_limit(tokens, index)
                continue

            raise UnsupportedCalculationError(
                f"Unsupported character {char!r}", index)

        tokens.append(Token("EOF", None, len(expression), ""))
        return tokens

    @staticmethod
    def _read_string(expression: str, start: int) -> tuple[str, int]:
        quote = expression[start]
        value: list[str] = []
        index = start + 1
        while index < len(expression):
            char = expression[index]
            if char == "\\":
                if index + 1 >= len(expression):
                    raise CalculationTranslationError(
                        "String ends with an escape character", index)
                value.append(expression[index + 1])
                index += 2
                continue
            if char == quote:
                # REDCap exports occasionally represent a literal quote by
                # doubling it, matching CSV/SQL conventions.
                if index + 1 < len(expression) and expression[index + 1] == quote:
                    value.append(quote)
                    index += 2
                    continue
                return "".join(value), index + 1
            value.append(char)
            index += 1
        raise CalculationTranslationError("Unclosed string literal", start)

    @staticmethod
    def _check_token_limit(tokens: Sequence[Token], position: int) -> None:
        if len(tokens) > MAX_TOKENS:
            raise UnsupportedCalculationError(
                f"Calculation exceeds {MAX_TOKENS:,} tokens", position)


class CalculationParser:
    """Pratt parser for the REDCap calculation grammar used by this project."""

    _precedence = {
        "or": 10,
        "and": 20,
        "=": 30, "==": 30, "<>": 30, "!=": 30,
        "<": 30, "<=": 30, ">": 30, ">=": 30,
        "+": 40, "-": 40,
        "*": 50, "/": 50, "%": 50,
        "^": 60,
    }
    _comparison_operators = frozenset(
        {"=", "==", "<>", "!=", "<", "<=", ">", ">="})

    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = CalculationTokenizer().tokenize(expression)
        self.index = 0
        self.depth = 0

    def parse(self) -> Node:
        if self._current().kind == "EOF":
            raise CalculationTranslationError("Calculation is blank", 0)
        result = self._parse_expression(0)
        trailing = self._current()
        if trailing.kind != "EOF":
            raise CalculationTranslationError(
                f"Unexpected trailing token {trailing.raw!r}",
                trailing.position)
        return result

    def _parse_expression(self, minimum_precedence: int) -> Node:
        self._enter_depth()
        try:
            left = self._parse_unary()
            comparison_seen = False
            while True:
                token = self._current()
                operator = self._binary_operator(token)
                if operator is None:
                    break
                precedence = self._precedence[operator]
                if precedence < minimum_precedence:
                    break
                if operator in self._comparison_operators:
                    if comparison_seen:
                        raise UnsupportedCalculationError(
                            "Chained comparisons are not supported",
                            token.position)
                    comparison_seen = True
                self._advance()
                # Power is right-associative; all other operators are left.
                right_minimum = precedence if operator == "^" else precedence + 1
                right = self._parse_expression(right_minimum)
                left = BinaryNode(operator, left, right)
            return left
        finally:
            self.depth -= 1

    def _parse_unary(self) -> Node:
        token = self._current()
        if token.kind == "OP" and token.value in {"+", "-"}:
            self._advance()
            # Exponentiation binds more tightly than a leading sign.
            return UnaryNode(
                token.value,
                self._parse_expression(self._precedence["^"]))
        # ``not`` is intentionally NOT a loose unary operator here: parsed
        # that way, ``not(X) + Y`` absorbed the ``+ Y`` into the negated
        # argument. It reaches _parse_call through _parse_primary instead,
        # so ``not(...)`` gets exact function-call semantics and a bare
        # ``not`` without parentheses fails closed.
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        token = self._current()
        if token.kind == "NUMBER":
            self._advance()
            return NumberNode(token.raw)

        if token.kind == "STRING":
            self._advance()
            return LiteralNode(token.value)

        if token.kind == "IDENT":
            self._advance()
            if token.value == "true":
                return LiteralNode(True)
            if token.value == "false":
                return LiteralNode(False)
            if self._current().kind != "LPAREN":
                raise UnsupportedCalculationError(
                    f"Unsupported bare identifier {token.raw!r}",
                    token.position)
            return self._parse_call(token)

        if token.kind == "FIELD":
            return self._parse_field_reference()

        if token.kind == "LPAREN":
            self._advance()
            expression = self._parse_expression(0)
            self._expect("RPAREN", "Expected ')' to close expression")
            return expression

        raise CalculationTranslationError(
            f"Expected a value, found {token.raw or 'end of input'!r}",
            token.position)

    def _parse_call(self, name_token: Token) -> Node:
        function = name_token.value
        if function not in SUPPORTED_FUNCTION_ARITY:
            raise UnsupportedCalculationError(
                f"Unsupported REDCap function {name_token.raw!r}",
                name_token.position)
        self._expect("LPAREN", "Expected '(' after function name")
        arguments: list[Node] = []
        if self._current().kind != "RPAREN":
            while True:
                arguments.append(self._parse_expression(0))
                if self._current().kind != "COMMA":
                    break
                self._advance()
        self._expect("RPAREN", f"Expected ')' after {function} arguments")

        minimum, maximum = SUPPORTED_FUNCTION_ARITY[function]
        if len(arguments) < minimum or (
                maximum is not None and len(arguments) > maximum):
            expected = (
                f"at least {minimum}" if maximum is None
                else str(minimum) if minimum == maximum
                else f"{minimum} to {maximum}")
            raise UnsupportedCalculationError(
                f"{function}() expects {expected} argument(s), got "
                f"{len(arguments)}", name_token.position)
        return CallNode(function, tuple(arguments))

    def _parse_field_reference(self) -> FieldNode:
        groups = [self._advance()]
        while self._current().kind == "FIELD" and len(groups) < 3:
            groups.append(self._advance())
        if self._current().kind == "FIELD":
            raise UnsupportedCalculationError(
                "References with more than three adjacent bracket groups "
                "are not supported", self._current().position)

        first = groups[0]
        event: str | None = None
        instance: int | None = None
        field_token = first
        if len(groups) == 2:
            if self._is_instance_token(groups[1]):
                instance = self._parse_instance(groups[1])
            else:
                event = first.value
                field_token = groups[1]
        elif len(groups) == 3:
            event = first.value
            field_token = groups[1]
            instance = self._parse_instance(groups[2])

        variable, choice = self._split_field_choice(field_token)
        if event is not None and not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]*", event):
            raise UnsupportedCalculationError(
                f"Unsupported event reference {event!r}", first.position)
        return FieldNode(
            variable=variable, event=event, choice=choice,
            instance=instance)

    @staticmethod
    def _is_instance_token(token: Token) -> bool:
        return bool(re.fullmatch(r"\d+", token.value))

    @staticmethod
    def _parse_instance(token: Token) -> int:
        if not CalculationParser._is_instance_token(token):
            raise UnsupportedCalculationError(
                f"Unsupported repeating-instance reference {token.raw!r}",
                token.position)
        instance = int(token.value)
        if instance < 1:
            raise UnsupportedCalculationError(
                "Repeating-instance numbers must be positive",
                token.position)
        return instance

    @staticmethod
    def _split_field_choice(token: Token) -> tuple[str, str | None]:
        match = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9_]*)(?:\(([^()]+)\))?", token.value)
        if match is None:
            raise UnsupportedCalculationError(
                f"Unsupported field reference {token.raw!r}", token.position)
        variable = match.group(1)
        choice = match.group(2)
        if choice is not None:
            choice = choice.strip()
            if not choice or not re.fullmatch(r"[A-Za-z0-9_.-]+", choice):
                raise UnsupportedCalculationError(
                    f"Unsupported checkbox choice {choice!r}", token.position)
        return variable, choice

    def _binary_operator(self, token: Token) -> str | None:
        if token.kind == "OP" and token.value in self._precedence:
            return token.value
        if token.kind == "IDENT" and token.value in {"and", "or"}:
            return token.value
        return None

    def _current(self) -> Token:
        return self.tokens[self.index]

    def _advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _expect(self, kind: str, message: str) -> Token:
        token = self._current()
        if token.kind != kind:
            raise CalculationTranslationError(message, token.position)
        return self._advance()

    def _enter_depth(self) -> None:
        self.depth += 1
        if self.depth > MAX_PARSE_DEPTH:
            raise UnsupportedCalculationError(
                f"Calculation nesting exceeds {MAX_PARSE_DEPTH}",
                self._current().position)


def iter_field_references(node: Node) -> Iterable[FieldNode]:
    """Yield field references from the parsed tree in source order."""
    if isinstance(node, FieldNode):
        yield node
    elif isinstance(node, UnaryNode):
        yield from iter_field_references(node.operand)
    elif isinstance(node, BinaryNode):
        yield from iter_field_references(node.left)
        yield from iter_field_references(node.right)
    elif isinstance(node, CallNode):
        for argument in node.arguments:
            yield from iter_field_references(argument)


def iter_function_names(node: Node) -> Iterable[str]:
    """Yield REDCap function names used by a parsed calculation."""
    if isinstance(node, UnaryNode):
        yield from iter_function_names(node.operand)
    elif isinstance(node, BinaryNode):
        yield from iter_function_names(node.left)
        yield from iter_function_names(node.right)
    elif isinstance(node, CallNode):
        yield node.function
        for argument in node.arguments:
            yield from iter_function_names(argument)


_BINARY_HELPERS = {
    "+": "redcap_add",
    "-": "redcap_subtract",
    "*": "redcap_multiply",
    "/": "redcap_divide",
    "%": "redcap_modulo",
    "^": "redcap_power",
    "=": "redcap_equal",
    "==": "redcap_equal",
    "<>": "redcap_not_equal",
    "!=": "redcap_not_equal",
    "<": "redcap_less_than",
    "<=": "redcap_less_equal",
    ">": "redcap_greater_than",
    ">=": "redcap_greater_equal",
}


def compile_calculation(node: Node) -> str:
    """Compile a trusted calculation AST into an allowlisted Python string."""
    if isinstance(node, NumberNode):
        return f"redcap_number({node.raw!r})"
    if isinstance(node, LiteralNode):
        return repr(node.value)
    if isinstance(node, FieldNode):
        variable = node.variable
        if node.choice is not None:
            # REDCap export column names lowercase the choice code and
            # replace every non-alphanumeric character with an underscore
            # (e.g. [field(-1)] exports as field____1), so the raw code
            # would look up a column that never exists.
            normalized_choice = re.sub(r"[^a-z0-9]", "_", node.choice.casefold())
            variable = f"{variable}___{normalized_choice}"
        arguments = (
            f"curr_row, {variable!r}, {node.event!r}, event_data")
        if node.instance is not None:
            arguments += f", {node.instance!r}"
        return f"redcap_value({arguments})"
    if isinstance(node, UnaryNode):
        operand = compile_calculation(node.operand)
        if node.operator == "+":
            return f"redcap_positive({operand})"
        if node.operator == "-":
            return f"redcap_negative({operand})"
        if node.operator == "not":
            return f"(not redcap_truthy({operand}))"
        raise AssertionError(f"Unexpected unary operator: {node.operator}")
    if isinstance(node, BinaryNode):
        left = compile_calculation(node.left)
        right = compile_calculation(node.right)
        if node.operator == "and":
            return (
                f"(redcap_truthy({left}) and redcap_truthy({right}))")
        if node.operator == "or":
            return (
                f"(redcap_truthy({left}) or redcap_truthy({right}))")
        helper = _BINARY_HELPERS[node.operator]
        return f"{helper}({left}, {right})"
    if isinstance(node, CallNode):
        arguments = [compile_calculation(arg) for arg in node.arguments]
        if node.function == "if":
            condition, when_true, when_false = arguments
            # Python's conditional expression is lazy, matching REDCap if().
            return (
                f"({when_true} if redcap_truthy({condition}) "
                f"else {when_false})")
        if node.function == "not":
            return f"(not redcap_truthy({arguments[0]}))"
        if node.function in {"and", "or"}:
            operator = f" {node.function} "
            return "(" + operator.join(
                f"redcap_truthy({value})" for value in arguments) + ")"
        helper = {
            "sum": "redcap_sum",
            "round": "redcap_round",
            "datediff": "redcap_datediff",
            "left": "redcap_left",
            "mid": "redcap_mid",
            "right": "redcap_right",
        }[node.function]
        return f"{helper}({', '.join(arguments)})"
    raise TypeError(f"Unsupported AST node: {type(node).__name__}")


_SAFE_CALL_NAMES = frozenset({
    "redcap_number", "redcap_value", "redcap_truthy",
    "redcap_positive", "redcap_negative",
    "redcap_add", "redcap_subtract", "redcap_multiply", "redcap_divide",
    "redcap_modulo", "redcap_power", "redcap_equal", "redcap_not_equal",
    "redcap_less_than", "redcap_less_equal", "redcap_greater_than",
    "redcap_greater_equal", "redcap_sum", "redcap_round",
    "redcap_datediff", "redcap_left", "redcap_mid", "redcap_right",
})
_SAFE_NAME_LOADS = _SAFE_CALL_NAMES | frozenset({"curr_row", "event_data"})
_SAFE_AST_NODES = (
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, ast.IfExp,
    ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
)

_SAFE_CALL_ARITY: Mapping[str, tuple[int, int | None]] = {
    "redcap_number": (1, 1),
    "redcap_value": (4, 5),
    "redcap_truthy": (1, 1),
    "redcap_positive": (1, 1),
    "redcap_negative": (1, 1),
    "redcap_add": (2, 2),
    "redcap_subtract": (2, 2),
    "redcap_multiply": (2, 2),
    "redcap_divide": (2, 2),
    "redcap_modulo": (2, 2),
    "redcap_power": (2, 2),
    "redcap_equal": (2, 2),
    "redcap_not_equal": (2, 2),
    "redcap_less_than": (2, 2),
    "redcap_less_equal": (2, 2),
    "redcap_greater_than": (2, 2),
    "redcap_greater_equal": (2, 2),
    "redcap_sum": (1, None),
    "redcap_round": (1, 2),
    "redcap_datediff": (3, 5),
    "redcap_left": (2, 2),
    "redcap_mid": (3, 3),
    "redcap_right": (2, 2),
}


def validate_generated_python(expression: str) -> ast.Expression:
    """Validate generated source against a deliberately tiny Python subset."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise CalculationTranslationError(
            f"Generated Python did not parse: {exc}") from exc
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            raise CalculationTranslationError(
                f"Generated Python contains forbidden AST node "
                f"{type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in _SAFE_NAME_LOADS:
                raise CalculationTranslationError(
                    f"Generated Python contains forbidden name {node.id!r}")
            parent = parents.get(node)
            if node.id in _SAFE_CALL_NAMES:
                if not (isinstance(parent, ast.Call)
                        and parent.func is node):
                    raise CalculationTranslationError(
                        f"Generated Python uses helper {node.id!r} "
                        "without calling it")
            elif not _is_valid_runtime_name_use(node, parent):
                raise CalculationTranslationError(
                    f"Generated Python uses runtime value {node.id!r} "
                    "outside redcap_value()")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise CalculationTranslationError(
                    "Generated Python calls a non-name expression")
            if node.func.id not in _SAFE_CALL_NAMES:
                raise CalculationTranslationError(
                    f"Generated Python calls forbidden helper "
                    f"{node.func.id!r}")
            if node.keywords:
                raise CalculationTranslationError(
                    "Generated Python contains keyword arguments")
            minimum, maximum = _SAFE_CALL_ARITY[node.func.id]
            if len(node.args) < minimum or (
                    maximum is not None and len(node.args) > maximum):
                raise CalculationTranslationError(
                    f"Generated Python calls {node.func.id}() with an "
                    "invalid number of arguments")
            if node.func.id == "redcap_value":
                _validate_redcap_value_call(node)
            elif node.func.id == "redcap_number":
                _validate_redcap_number_call(node)
    return tree


def _is_valid_runtime_name_use(
        node: ast.Name, parent: ast.AST | None) -> bool:
    if not isinstance(parent, ast.Call):
        return False
    if not isinstance(parent.func, ast.Name) or parent.func.id != "redcap_value":
        return False
    expected_index = 0 if node.id == "curr_row" else 3
    return (len(parent.args) > expected_index
            and parent.args[expected_index] is node)


def _validate_redcap_number_call(node: ast.Call) -> None:
    argument = node.args[0]
    if not (isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and CalculationTokenizer._number_re.fullmatch(argument.value)
            and len(argument.value) <= MAX_NUMERIC_LITERAL_LENGTH):
        raise CalculationTranslationError(
            "Generated Python contains an invalid numeric literal")


def _validate_redcap_value_call(node: ast.Call) -> None:
    row, variable, event, event_data, *instance = node.args
    if not (isinstance(row, ast.Name) and row.id == "curr_row"):
        raise CalculationTranslationError(
            "redcap_value() must receive curr_row as its first argument")
    if not (isinstance(variable, ast.Constant)
            and isinstance(variable.value, str)
            and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]*(?:___[A-Za-z0-9_.-]+)?",
                variable.value)):
        raise CalculationTranslationError(
            "redcap_value() contains an invalid variable argument")
    if not (isinstance(event, ast.Constant)
            and (event.value is None or (
                isinstance(event.value, str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", event.value)))):
        raise CalculationTranslationError(
            "redcap_value() contains an invalid event argument")
    if not (isinstance(event_data, ast.Name)
            and event_data.id == "event_data"):
        raise CalculationTranslationError(
            "redcap_value() must receive event_data as its fourth argument")
    if instance and not (
            isinstance(instance[0], ast.Constant)
            and isinstance(instance[0].value, int)
            and not isinstance(instance[0].value, bool)
            and instance[0].value >= 1):
        raise CalculationTranslationError(
            "redcap_value() contains an invalid repeating instance")


def translate_redcap_calculation(expression: str) -> tuple[str, Node]:
    """Parse one REDCap expression and return validated Python plus its AST."""
    try:
        node = CalculationParser(expression).parse()
        converted = compile_calculation(node)
        validate_generated_python(converted)
        return converted, node
    except RecursionError as exc:
        raise UnsupportedCalculationError(
            "Calculation nesting exceeds the safe Python recursion limit") from exc


# ---------------------------------------------------------------------------
# Runtime helpers for generated expressions

_MISSING = object()


def _container_value(container: Any, key: Any) -> Any:
    if container is None:
        return _MISSING
    if isinstance(container, Mapping):
        return container.get(key, _MISSING)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, _MISSING)
        except (KeyError, TypeError, ValueError):
            pass
    # pandas Series and similar labeled rows expose field names through
    # __getitem__. Prefer that path to attributes such as ``size`` or ``name``.
    if not isinstance(key, tuple) and hasattr(container, "__getitem__"):
        try:
            return container[key]
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    return getattr(container, str(key), _MISSING)


def _present_value(value: Any) -> Any:
    return normalize_calculation_input(value)


def redcap_number(raw: str) -> Decimal:
    """Construct an exact numeric literal already validated by the compiler."""
    value = Decimal(raw)
    if not value.is_finite():
        raise ValueError("REDCap numeric literals must be finite")
    return value


def redcap_value(curr_row: Any, variable: str, event: str | None = None,
                 event_data: Any = None, instance: int | None = None) -> Any:
    """Resolve current-row or explicit-event data without dynamic attributes.

    ``event_data`` may be nested as ``{event: row}``, keyed by
    ``(event, variable)``, or use the pipeline's historical concatenated
    ``event + variable`` column convention. An explicit event never silently
    falls back to the current row.
    """
    if event is None and instance is None:
        value = _container_value(curr_row, variable)
        return _present_value(value)

    # Explicit instances are never allowed to fall back to the current row.
    # Accept common tuple-keyed and nested representations so callers do not
    # have to reshape longitudinal/repeating REDCap exports into one layout.
    if instance is not None:
        tuple_keys = (
            ((event, variable, instance), (event, instance, variable))
            if event is not None
            else ((variable, instance), (instance, variable)))
        for key in tuple_keys:
            value = _container_value(event_data, key)
            if value is not _MISSING:
                return _present_value(value)

        scope = (
            _container_value(event_data, event)
            if event is not None else event_data)
        if scope is not _MISSING:
            instance_row = _container_value(scope, instance)
            if instance_row is _MISSING:
                instance_row = _container_value(scope, str(instance))
            if instance_row is not _MISSING:
                value = _container_value(instance_row, variable)
                if value is not _MISSING:
                    return _present_value(value)
            variable_instances = _container_value(scope, variable)
            if variable_instances is not _MISSING:
                value = _container_value(variable_instances, instance)
                if value is _MISSING:
                    value = _container_value(
                        variable_instances, str(instance))
                if value is not _MISSING:
                    return _present_value(value)

        prefix = f"[{event}][{variable}]" if event else f"[{variable}]"
        flat_keys = (
            f"{prefix}[{instance}]",
            f"{event}__{variable}__{instance}" if event
            else f"{variable}__{instance}",
        )
        for key in flat_keys:
            value = _container_value(event_data, key)
            if value is not _MISSING:
                return _present_value(value)
            value = _container_value(curr_row, key)
            if value is not _MISSING:
                return _present_value(value)
        return ""

    pair_value = _container_value(event_data, (event, variable))
    if pair_value is not _MISSING:
        return _present_value(pair_value)

    event_row = _container_value(event_data, event)
    if event_row is not _MISSING:
        value = _container_value(event_row, variable)
        if value is not _MISSING:
            return _present_value(value)

    for key in (
            f"[{event}][{variable}]", f"{event}__{variable}",
            f"{event}{variable}"):
        value = _container_value(event_data, key)
        if value is not _MISSING:
            return _present_value(value)
        value = _container_value(curr_row, key)
        if value is not _MISSING:
            return _present_value(value)
    return ""


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def is_calculation_missing_code(value: Any) -> bool:
    """Whether one imported scalar is an exact configured missing code."""
    if isinstance(value, (datetime, date)):
        return str(value) in _CALCULATION_MISSING_CODE_TEXT
    if isinstance(value, str):
        return value in _CALCULATION_MISSING_CODE_TEXT
    # bool is an int subclass, but True/False are answers, not missing codes.
    if isinstance(value, bool):
        return False
    try:
        return bool(value in _CALCULATION_MISSING_CODE_NUMBERS)
    except (TypeError, ValueError):
        return False


def normalize_calculation_input(value: Any) -> Any:
    """Present absent, null, or study-policy missing data as ``''``.

    Field lookups use this before evaluation. Formula literals retain their
    value within the expression (so ``999 + 1`` remains 1000), while the
    evaluator reapplies the same policy to the normalized root result before
    that result can be compared or passed to a dependent calculation.
    """
    if (value is _MISSING or _is_missing_scalar(value)
            or is_calculation_missing_code(value)):
        return ""
    return value


def _raw_text(value: Any) -> str:
    if _is_missing_scalar(value):
        return ""
    return str(value)


def _text(value: Any) -> str:
    return _raw_text(value).strip()


def _decimal(value: Any, *, blank_as_zero: bool) -> Decimal | None:
    if isinstance(value, bool):
        return Decimal(1 if value else 0)
    text = _text(value)
    if text == "":
        return Decimal(0) if blank_as_zero else None
    if len(text) > MAX_RUNTIME_NUMERIC_TEXT_LENGTH:
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return number


def _number_result(value: Decimal | None) -> int | float | str:
    if value is None or not value.is_finite():
        return ""
    integral = value.to_integral_value()
    if value == integral:
        # Keep astronomically large integrals in their compact Decimal text
        # form: materializing e.g. 9E+200000 as an int takes unbounded CPU,
        # and any downstream str() of the result trips Python 3.11+'s
        # 4300-digit integer-string limit inside the consumer.
        if integral.adjusted() >= MAX_RESULT_INTEGER_DIGITS:
            return str(value)
        return int(integral)
    # Preserve values whose decimal spelling cannot round-trip through a
    # binary float. Simple values such as 0.1 remain convenient floats, while
    # high-precision results remain exact strings for subsequent helpers and
    # export comparisons.
    try:
        candidate = float(value)
        if Decimal(str(candidate)) == value:
            return candidate
    except (OverflowError, ValueError):
        pass
    return str(value)


def redcap_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw_text = _raw_text(value)
    if raw_text == "":
        return False
    number = _decimal(value, blank_as_zero=False)
    if number is not None:
        return number != 0
    return raw_text.casefold() not in {"false", "no"}


def redcap_positive(value: Any) -> int | float | str:
    return _number_result(_decimal(value, blank_as_zero=False))


def redcap_negative(value: Any) -> int | float | str:
    number = _decimal(value, blank_as_zero=False)
    return _number_result(None if number is None else -number)


def _numeric_binary(left: Any, right: Any, operation) -> int | float | str:
    # Direct REDCap arithmetic propagates blank operands. The special sum()
    # function is intentionally different and skips blank/non-numeric values.
    left_number = _decimal(left, blank_as_zero=False)
    right_number = _decimal(right, blank_as_zero=False)
    if left_number is None or right_number is None:
        return ""
    try:
        return _number_result(operation(left_number, right_number))
    except (ArithmeticError, InvalidOperation, OverflowError, ValueError):
        return ""


def redcap_add(left: Any, right: Any) -> int | float | str:
    return _numeric_binary(left, right, lambda a, b: a + b)


def redcap_subtract(left: Any, right: Any) -> int | float | str:
    return _numeric_binary(left, right, lambda a, b: a - b)


def redcap_multiply(left: Any, right: Any) -> int | float | str:
    return _numeric_binary(left, right, lambda a, b: a * b)


def redcap_divide(left: Any, right: Any) -> int | float | str:
    return _numeric_binary(left, right, lambda a, b: a / b)


def redcap_modulo(left: Any, right: Any) -> int | float | str:
    return _numeric_binary(left, right, lambda a, b: a % b)


def redcap_power(left: Any, right: Any) -> int | float | str:
    exponent = _decimal(right, blank_as_zero=False)
    if exponent is None or abs(exponent) > MAX_POWER_ABS_EXPONENT:
        return ""
    return _numeric_binary(left, exponent, lambda a, b: a ** b)


def _comparison_values(left: Any, right: Any) -> tuple[Any, Any, bool]:
    left_number = _decimal(left, blank_as_zero=False)
    right_number = _decimal(right, blank_as_zero=False)
    if left_number is not None and right_number is not None:
        return left_number, right_number, True
    return _raw_text(left), _raw_text(right), False


def redcap_equal(left: Any, right: Any) -> bool:
    left_value, right_value, _ = _comparison_values(left, right)
    return left_value == right_value


def redcap_not_equal(left: Any, right: Any) -> bool:
    return not redcap_equal(left, right)


def _ordered_comparison(left: Any, right: Any, comparison) -> bool:
    left_value, right_value, numeric = _comparison_values(left, right)
    if not numeric and (left_value == "" or right_value == ""):
        return False
    try:
        return bool(comparison(left_value, right_value))
    except (TypeError, ValueError):
        return False


def redcap_less_than(left: Any, right: Any) -> bool:
    return _ordered_comparison(left, right, lambda a, b: a < b)


def redcap_less_equal(left: Any, right: Any) -> bool:
    return _ordered_comparison(left, right, lambda a, b: a <= b)


def redcap_greater_than(left: Any, right: Any) -> bool:
    return _ordered_comparison(left, right, lambda a, b: a > b)


def redcap_greater_equal(left: Any, right: Any) -> bool:
    return _ordered_comparison(left, right, lambda a, b: a >= b)


def redcap_sum(*values: Any) -> int | float | str:
    total = Decimal(0)
    for value in values:
        number = _decimal(value, blank_as_zero=False)
        if number is not None:
            total += number
    return _number_result(total)


def redcap_round(value: Any, places: Any = 0) -> int | float | str:
    number = _decimal(value, blank_as_zero=False)
    decimal_places = _decimal(places, blank_as_zero=False)
    if number is None or decimal_places is None:
        return ""
    if decimal_places != decimal_places.to_integral_value():
        return ""
    count = int(decimal_places)
    if count < -100 or count > 100:
        return ""
    quantum = Decimal(1).scaleb(-count)
    try:
        return _number_result(number.quantize(quantum, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return ""


def _parse_redcap_datetime(value: Any, date_format: str) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    # REDCap resolves the dynamic keywords 'today' and 'now' at evaluation
    # time; without this branch the canonical age calculation
    # datediff([dob],'today','y') silently evaluated to blank.
    keyword = text.casefold()
    if keyword == "today":
        current = datetime.now()
        return datetime(current.year, current.month, current.day)
    if keyword == "now":
        return datetime.now()
    normalized = date_format.strip().casefold()
    formats = {
        "ymd": ("%Y-%m-%d",),
        "mdy": ("%m-%d-%Y", "%m/%d/%Y"),
        "dmy": ("%d-%m-%Y", "%d/%m/%Y"),
        "ymd h:m": ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"),
        "ymd h:m:s": (
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"),
        "h:m": ("%H:%M",),
        "h:m:s": ("%H:%M:%S",),
    }.get(normalized)
    if formats is None:
        return None
    for pattern in formats:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def redcap_datediff(start: Any, end: Any, units: Any,
                    date_format: Any = "ymd", signed: Any = False
                    ) -> int | float | str:
    """Evaluate the observed REDCap datediff signatures.

    REDCap treats the unsigned form as an absolute difference. Unit ``M`` is
    months while lower-case ``m`` is minutes. Year/month conversion follows the
    conventional mean Gregorian lengths and should be golden-tested against a
    live REDCap export before this evaluator is used for production QC.
    """
    unit = _text(units)
    fourth = _raw_text(date_format)
    if (isinstance(date_format, bool)
            or fourth.strip().casefold() in {"true", "false", "1", "0"}):
        # Modern REDCap also permits the four-argument form where the fourth
        # argument is returnSignedValue rather than a date format. Numeric
        # 1/0 flags count too: no date format is a bare digit, and treating
        # them as a format made the whole calculation silently blank.
        signed = date_format
        fmt = "ymd"
    else:
        fmt = fourth.strip() or "ymd"
    start_dt = _parse_redcap_datetime(start, fmt)
    end_dt = _parse_redcap_datetime(end, fmt)
    if start_dt is None or end_dt is None:
        return ""
    seconds = (end_dt - start_dt).total_seconds()
    if not redcap_truthy(signed):
        seconds = abs(seconds)
    divisors = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "M": 86400 * 30.436875,
        "y": 86400 * 365.2425,
    }
    divisor = divisors.get(unit)
    if divisor is None:
        return ""
    return _number_result(Decimal(str(seconds)) / Decimal(str(divisor)))


def redcap_left(value: Any, length: Any) -> str:
    count = _decimal(length, blank_as_zero=False)
    if count is None or count != count.to_integral_value() or count < 0:
        return ""
    return _raw_text(value)[:int(count)]


def redcap_right(value: Any, length: Any) -> str:
    count = _decimal(length, blank_as_zero=False)
    if count is None or count != count.to_integral_value() or count < 0:
        return ""
    count_int = int(count)
    return "" if count_int == 0 else _raw_text(value)[-count_int:]


def redcap_mid(value: Any, start: Any, length: Any) -> str:
    start_number = _decimal(start, blank_as_zero=False)
    count = _decimal(length, blank_as_zero=False)
    if (start_number is None or count is None
            or start_number != start_number.to_integral_value()
            or count != count.to_integral_value()
            or start_number < 1 or count < 0):
        return ""
    begin = int(start_number) - 1  # REDCap positions are one-based.
    return _raw_text(value)[begin:begin + int(count)]


_SAFE_EVAL_GLOBALS = {
    "__builtins__": {},
    **{
        name: globals()[name]
        for name in _SAFE_CALL_NAMES
    },
}


@dataclass(frozen=True)
class CalculationEvaluation:
    value: Any
    status: str
    error: str = ""


def evaluate_converted_calculation(
        expression: str, curr_row: Any,
        event_data: Any = None, *,
        blank_result_as_zero: bool = False) -> CalculationEvaluation:
    """Safely evaluate translator-generated Python and report failures.

    ``blank_result_as_zero`` is an explicit persistence compatibility mode.
    Some locally observed recalculation exports store an exact root ``''`` as
    numeric zero, while institutional REDCap guidance documents blank output.
    The default therefore preserves blank until a study-version golden export
    establishes that zero coercion is required.

    The study missing-code policy is then applied to the normalized root
    result so a configured sentinel cannot feed a dependent calculation.
    """
    try:
        code = _compile_validated_calculation(expression)
        value = eval(  # noqa: S307 -- validated AST + no builtins by design
            code, _SAFE_EVAL_GLOBALS,
            {"curr_row": curr_row, "event_data": event_data})
        # REDCap calculated fields persist logical results as numeric 1/0.
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, Decimal):
            value = _number_result(value)
        elif blank_result_as_zero and value == "":
            value = 0
        value = normalize_calculation_input(value)
        return CalculationEvaluation(value=value, status="ok")
    except Exception as exc:
        return CalculationEvaluation(
            value="", status="error",
            error=f"{type(exc).__name__}: {exc}")


@lru_cache(maxsize=4_096)
def _compile_validated_calculation(expression: str):
    """Validate and compile once per generated expression."""
    tree = validate_generated_python(expression)
    return compile(tree, "<converted REDCap calculation>", "eval")


# ---------------------------------------------------------------------------
# Data-dictionary conversion and dependency analysis


class TransformCalculatedFields:
    """Convert every ``Field Type == calc`` row in a REDCap dictionary."""

    def __init__(self, data_dictionary_df: pd.DataFrame):
        self.data_dictionary_df = data_dictionary_df.copy()
        self._validate_schema()
        self.all_converted_calculations: dict[str, dict[str, Any]] = {}
        self.excluded_conversions: dict[str, dict[str, Any]] = {}
        self.evaluation_order: list[str] = []
        self._converted = False

    def __call__(self) -> dict[str, dict[str, Any]]:
        return self.convert_all_calculated_fields()

    def _validate_schema(self) -> None:
        required = {
            VARIABLE_COLUMN, FORM_COLUMN, FIELD_TYPE_COLUMN,
            CALCULATION_COLUMN,
        }
        missing = sorted(required.difference(self.data_dictionary_df.columns))
        if missing:
            raise CalculationSchemaError(
                "Data dictionary is missing required column(s): "
                + ", ".join(missing))

        variables = self.data_dictionary_df[VARIABLE_COLUMN].astype(str)
        blank = variables.str.strip().eq("")
        if blank.any():
            rows = [
                position + 2 for position, is_blank in enumerate(blank)
                if is_blank][:10]
            raise CalculationSchemaError(
                f"Variable names are blank at CSV row(s): {rows}")
        surrounding_whitespace = variables.ne(variables.str.strip())
        if surrounding_whitespace.any():
            rows = [
                position + 2 for position, has_whitespace
                in enumerate(surrounding_whitespace) if has_whitespace][:10]
            raise CalculationSchemaError(
                "Variable names contain leading/trailing whitespace at CSV "
                f"row(s): {rows}")
        invalid_names = variables[~variables.str.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*")].unique().tolist()
        if invalid_names:
            raise CalculationSchemaError(
                "Invalid REDCap variable name(s): "
                + ", ".join(invalid_names[:20]))
        normalized = variables.str.casefold()
        duplicates = variables[
            normalized.duplicated(keep=False)].unique().tolist()
        if duplicates:
            raise CalculationSchemaError(
                "Duplicate variable name(s): " + ", ".join(duplicates[:20]))

    def convert_all_calculated_fields(self) -> dict[str, dict[str, Any]]:
        if self._converted:
            return self.all_converted_calculations

        frame = self.data_dictionary_df
        all_fields = set(frame[VARIABLE_COLUMN].astype(str))
        metadata_by_field = {
            str(row[VARIABLE_COLUMN]): row
            for _, row in frame.iterrows()
        }
        calc_frame = frame[
            frame[FIELD_TYPE_COLUMN].astype(str).str.strip().str.casefold()
            == "calc"
        ]
        calc_source_rows = [
            position + 2 for position, is_calc in enumerate(
                frame[FIELD_TYPE_COLUMN].astype(str).str.strip().str.casefold()
                == "calc") if is_calc]
        calc_fields = calc_frame[VARIABLE_COLUMN].astype(str).tolist()
        calc_field_set = set(calc_fields)
        identifier_fields = set()
        if IDENTIFIER_COLUMN in frame.columns:
            identifier_fields = set(frame.loc[
                frame[IDENTIFIER_COLUMN].astype(str).str.strip().str.casefold()
                == "y", VARIABLE_COLUMN].astype(str))

        for source_row, (_, row) in zip(
                calc_source_rows, calc_frame.iterrows()):
            variable = str(row[VARIABLE_COLUMN]).strip()
            original = str(row[CALCULATION_COLUMN])
            entry: dict[str, Any] = {
                "variable": variable,
                "form": str(row[FORM_COLUMN]),
                "source_row": source_row,
                "original_calculation": original,
                "converted_calculation": "",
                "status": "invalid",
                "error_code": "",
                "error": "",
                "references": [],
                "referenced_variables": [],
                "same_event_calc_dependencies": [],
                "cross_event_dependencies": [],
                "direct_identifier_dependencies": [],
                "transitive_identifier_dependencies": [],
                "functions": [],
                "branching_logic": (
                    str(row[BRANCHING_LOGIC_COLUMN])
                    if BRANCHING_LOGIC_COLUMN in calc_frame.columns else ""),
                "evaluation_order": None,
            }
            try:
                converted, node = translate_redcap_calculation(original)
                references = self._reference_metadata(
                    node, metadata_by_field, calc_field_set)
                unknown = sorted({
                    reference["variable"] for reference in references
                    if reference["variable"] not in all_fields
                })
                if unknown:
                    raise CalculationTranslationError(
                        "Unknown referenced variable(s): " + ", ".join(unknown))

                entry["converted_calculation"] = converted
                entry["status"] = "converted"
                entry["references"] = references
                entry["referenced_variables"] = list(dict.fromkeys(
                    reference["variable"] for reference in references))
                entry["same_event_calc_dependencies"] = list(dict.fromkeys(
                    reference["variable"] for reference in references
                    if (reference["event"] is None
                        and reference["variable"] in calc_field_set)))
                entry["cross_event_dependencies"] = [
                    reference for reference in references
                    if reference["event"] is not None]
                entry["direct_identifier_dependencies"] = sorted({
                    reference["variable"] for reference in references
                    if reference["variable"] in identifier_fields})
                entry["functions"] = list(dict.fromkeys(
                    iter_function_names(node)))
            except CalculationTranslationError as exc:
                entry["status"] = (
                    "unsupported"
                    if isinstance(exc, UnsupportedCalculationError)
                    else "invalid")
                entry["error_code"] = exc.code
                entry["error"] = str(exc)

            self.all_converted_calculations[variable] = entry

        self._apply_dependency_order(calc_fields)
        self._apply_identifier_propagation(calc_fields, identifier_fields)
        self.excluded_conversions = {
            variable: {
                "variable": variable,
                "form": entry["form"],
                "source_row": entry["source_row"],
                "original_calculation": entry["original_calculation"],
                "status": entry["status"],
                "error_code": entry["error_code"],
                "error": entry["error"],
            }
            for variable, entry in self.all_converted_calculations.items()
            if entry["status"] != "converted"
        }
        self._converted = True
        return self.all_converted_calculations

    @staticmethod
    def _reference_metadata(
            node: Node, metadata_by_field: Mapping[str, pd.Series],
            calc_field_set: set[str]) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        seen: set[tuple[str, str | None, str | None, int | None]] = set()
        for reference in iter_field_references(node):
            key = (
                reference.variable, reference.event, reference.choice,
                reference.instance)
            if key in seen:
                continue
            seen.add(key)
            source = metadata_by_field.get(reference.variable)
            source_type = ""
            source_form = ""
            if source is not None:
                source_type = str(source.get(FIELD_TYPE_COLUMN, ""))
                source_form = str(source.get(FORM_COLUMN, ""))
            if reference.choice is not None and source is not None:
                if source_type.strip().casefold() != "checkbox":
                    raise CalculationTranslationError(
                        f"Choice reference {reference.display} targets "
                        f"non-checkbox field {reference.variable!r}")
                choices = TransformCalculatedFields._checkbox_choice_codes(
                    source.get(CALCULATION_COLUMN, ""))
                if reference.choice not in choices:
                    raise CalculationTranslationError(
                        f"Choice {reference.choice!r} is not declared for "
                        f"checkbox field {reference.variable!r}")
            references.append({
                "variable": reference.variable,
                "event": reference.event,
                "choice": reference.choice,
                "instance": reference.instance,
                "kind": (
                    "checkbox" if reference.choice is not None
                    else "calc" if reference.variable in calc_field_set
                    else "field"),
                "source_field_type": source_type,
                "source_form": source_form,
                "display": reference.display,
            })
        return references

    @staticmethod
    def _checkbox_choice_codes(raw_choices: Any) -> set[str]:
        codes: set[str] = set()
        for choice in str(raw_choices).split("|"):
            code, separator, _ = choice.partition(",")
            normalized = code.strip()
            if separator and normalized:
                codes.add(normalized)
        return codes

    def _apply_dependency_order(self, calc_fields: Sequence[str]) -> None:
        field_position = {field: index for index, field in enumerate(calc_fields)}
        self._block_failed_same_event_dependencies(calc_fields)

        active = [
            field for field in calc_fields
            if self.all_converted_calculations[field]["status"] == "converted"]
        dependencies = self._active_dependencies(active)
        order, residue = self._topological_order(
            active, dependencies, field_position)

        if residue:
            cycle_members = self._find_cycle_members(residue, dependencies)
            cycle_text = ", ".join(sorted(
                cycle_members, key=field_position.__getitem__)[:20])
            if len(cycle_members) > 20:
                cycle_text += f", ... ({len(cycle_members)} fields total)"
            for field in cycle_members:
                entry = self.all_converted_calculations[field]
                entry["status"] = "invalid"
                entry["converted_calculation"] = ""
                entry["error_code"] = "calculation_dependency_cycle"
                entry["error"] = (
                    "Same-event calculated-field dependency cycle includes: "
                    + cycle_text)

            # Fields downstream of a cycle are blocked, but are not themselves
            # mislabeled as cycle members.
            self._block_failed_same_event_dependencies(calc_fields)
            active = [
                field for field in calc_fields
                if self.all_converted_calculations[field]["status"]
                == "converted"]
            dependencies = self._active_dependencies(active)
            order, residue = self._topological_order(
                active, dependencies, field_position)
            if residue:
                # Defensive fail-closed path; _find_cycle_members should have
                # removed every cyclic component.
                for field in residue:
                    entry = self.all_converted_calculations[field]
                    entry["status"] = "invalid"
                    entry["converted_calculation"] = ""
                    entry["error_code"] = "calculation_dependency_cycle"
                    entry["error"] = (
                        "Unresolved same-event calculation dependency cycle")
                order = [field for field in order if field not in residue]

        self.evaluation_order = order
        for index, field in enumerate(self.evaluation_order):
            self.all_converted_calculations[field]["evaluation_order"] = index

    def _block_failed_same_event_dependencies(
            self, calc_fields: Sequence[str]) -> None:
        position = {field: index for index, field in enumerate(calc_fields)}
        reverse: dict[str, list[str]] = defaultdict(list)
        for field in calc_fields:
            for dependency in self.all_converted_calculations[field][
                    "same_event_calc_dependencies"]:
                reverse[dependency].append(field)
        for dependents in reverse.values():
            dependents.sort(key=position.__getitem__)

        queue = deque(
            field for field in calc_fields
            if self.all_converted_calculations[field]["status"] != "converted")
        while queue:
            failed_dependency = queue.popleft()
            for field in reverse.get(failed_dependency, ()):
                entry = self.all_converted_calculations[field]
                if entry["status"] != "converted":
                    continue
                failed = [
                    dependency
                    for dependency in entry["same_event_calc_dependencies"]
                    if self.all_converted_calculations[dependency]["status"]
                    != "converted"]
                if not failed:
                    continue
                entry["status"] = "invalid"
                entry["converted_calculation"] = ""
                entry["error_code"] = "blocked_calculation_dependency"
                entry["error"] = (
                    "Cannot safely recalculate because same-event "
                    "calculated dependency conversion failed: "
                    + ", ".join(failed))
                queue.append(field)

    def _active_dependencies(
            self, active: Sequence[str]) -> dict[str, set[str]]:
        active_set = set(active)
        return {
            field: set(self.all_converted_calculations[field][
                "same_event_calc_dependencies"]).intersection(active_set)
            for field in active
        }

    @staticmethod
    def _topological_order(
            fields: Sequence[str], dependencies: Mapping[str, set[str]],
            field_position: Mapping[str, int]
            ) -> tuple[list[str], set[str]]:
        reverse: dict[str, set[str]] = defaultdict(set)
        indegree = {field: len(values) for field, values in dependencies.items()}
        for field, values in dependencies.items():
            for dependency in values:
                reverse[dependency].add(field)

        ready = deque(field for field in fields if indegree[field] == 0)
        order: list[str] = []
        while ready:
            field = ready.popleft()
            order.append(field)
            newly_ready: list[str] = []
            for dependent in reverse.get(field, ()):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    newly_ready.append(dependent)
            for dependent in sorted(
                    newly_ready, key=field_position.__getitem__):
                ready.append(dependent)
        residue = {field for field, count in indegree.items() if count > 0}
        return order, residue

    @staticmethod
    def _find_cycle_members(
            residue: set[str], dependencies: Mapping[str, set[str]]) -> set[str]:
        """Return nodes on cycles, excluding merely downstream residue."""
        graph = {
            field: tuple(sorted(
                dependencies.get(field, set()).intersection(residue)))
            for field in residue
        }
        reverse: dict[str, list[str]] = defaultdict(list)
        for field, values in graph.items():
            for dependency in values:
                reverse[dependency].append(field)

        # Iterative Kosaraju avoids both the previous O(V*(V+E)) walk and
        # Python recursion limits on long dependency chains.
        visited: set[str] = set()
        finish_order: list[str] = []
        for start in sorted(residue):
            if start in visited:
                continue
            visited.add(start)
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                field, next_index = stack[-1]
                neighbors = graph[field]
                if next_index < len(neighbors):
                    neighbor = neighbors[next_index]
                    stack[-1] = (field, next_index + 1)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, 0))
                else:
                    stack.pop()
                    finish_order.append(field)

        cycle_members: set[str] = set()
        assigned: set[str] = set()
        for start in reversed(finish_order):
            if start in assigned:
                continue
            component: set[str] = set()
            stack = [start]
            assigned.add(start)
            while stack:
                current = stack.pop()
                component.add(current)
                for neighbor in reverse.get(current, ()):
                    if neighbor not in assigned:
                        assigned.add(neighbor)
                        stack.append(neighbor)
            if len(component) > 1:
                cycle_members.update(component)
            else:
                only = next(iter(component))
                if only in graph[only]:
                    cycle_members.add(only)
        return cycle_members

    def _apply_identifier_propagation(
            self, calc_fields: Sequence[str], identifier_fields: set[str]
            ) -> None:
        # Include cross-event calc dependencies for provenance propagation,
        # but not for same-event evaluation ordering.
        all_calc_dependencies: dict[str, set[str]] = {}
        for field in calc_fields:
            entry = self.all_converted_calculations[field]
            dependencies = set(entry["same_event_calc_dependencies"])
            dependencies.update(
                reference["variable"]
                for reference in entry["cross_event_dependencies"]
                if reference["kind"] == "calc")
            all_calc_dependencies[field] = dependencies

        propagated = {
            field: set(self.all_converted_calculations[field][
                "direct_identifier_dependencies"])
            for field in calc_fields
        }
        changed = True
        while changed:
            changed = False
            for field in calc_fields:
                expanded = set(propagated[field])
                for dependency in all_calc_dependencies[field]:
                    expanded.update(propagated.get(dependency, ()))
                    if dependency in identifier_fields:
                        expanded.add(dependency)
                if expanded != propagated[field]:
                    propagated[field] = expanded
                    changed = True
        for field in calc_fields:
            direct = set(self.all_converted_calculations[field][
                "direct_identifier_dependencies"])
            self.all_converted_calculations[field][
                "transitive_identifier_dependencies"] = sorted(
                    propagated[field] - direct)

    def build_artifact(
            self, source_path: Path | None = None,
            source_sha256: str | None = None) -> dict[str, Any]:
        calculations = self.convert_all_calculated_fields()
        converted_count = sum(
            entry["status"] == "converted" for entry in calculations.values())
        artifact: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_data_dictionary": (
                str(source_path.resolve()) if source_path is not None else ""),
            "source_sha256": (
                source_sha256
                if source_sha256 is not None
                else _sha256_file(source_path)
                if source_path is not None else ""),
            "summary": {
                "calculated_field_count": len(calculations),
                "converted_count": converted_count,
                "excluded_count": len(calculations) - converted_count,
                "same_event_evaluation_order_count": len(
                    self.evaluation_order),
            },
            "runtime_evaluator": {
                "status": "experimental",
                "warning": (
                    "Translation and documented blank arithmetic are "
                    "validated, but persisted root-blank coercion and "
                    "datediff behavior must be golden-tested against the "
                    "study's deployed REDCap version before evaluator output "
                    "is used for QC. The same applies to comparison typing "
                    "(blank=0 and blank<n are False here but True in "
                    "REDCap's JS client; '1'='1.0' is True here), "
                    "Decimal-exact arithmetic vs IEEE floats "
                    "(0.1+0.2=0.3 is True here, False in JS), round() "
                    "tie-breaking, fractional modulo, the datediff month "
                    "constant (30.436875 vs the documented 30.44), and the "
                    "truthiness of the strings 'no'/'false'."),
                "blank_result_as_zero_default": False,
                "blank_result_as_zero_note": (
                    "Pass blank_result_as_zero=True only when a trusted "
                    "study-version export proves exact root blanks persist "
                    "as numeric zero."),
            },
            "supported_functions": sorted(SUPPORTED_FUNCTION_ARITY),
            "same_event_evaluation_order": self.evaluation_order,
            "calculations": calculations,
        }
        return artifact


CalculatedFieldTranslator = TransformCalculatedFields


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_csv_write(calculations: Mapping[str, Mapping[str, Any]],
                      path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for entry in calculations.values():
        row = dict(entry)
        for key in (
                "references", "referenced_variables",
                "same_event_calc_dependencies", "cross_event_dependencies",
                "direct_identifier_dependencies",
                "transitive_identifier_dependencies", "functions"):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_artifact_set(
        artifact: Mapping[str, Any],
        calculations: Mapping[str, Mapping[str, Any]],
        diagnostics: Mapping[str, Any],
        json_path: Path, csv_path: Path, diagnostics_path: Path) -> None:
    """Stage all outputs, then commit them as one rollback-safe snapshot.

    No operating system offers a single atomic rename for three files. This
    transaction writes and validates every staged file first, commits the JSON
    artifact (the consumer-facing manifest) last, and restores the prior set
    if any rename fails.
    """
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    destinations = (csv_path, diagnostics_path, json_path)
    staged = {
        destination: destination.with_name(
            f".{destination.name}.{os.getpid()}.{token}.stage")
        for destination in destinations
    }
    backups = {
        destination: destination.with_name(
            f".{destination.name}.{os.getpid()}.{token}.backup")
        for destination in destinations
    }
    touched: list[Path] = []
    committed_successfully = False

    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_csv_write(calculations, staged[csv_path])
        _atomic_json_write(diagnostics, staged[diagnostics_path])
        _atomic_json_write(artifact, staged[json_path])

        for destination in destinations:
            backup = backups[destination]
            if destination.exists():
                os.replace(destination, backup)
            touched.append(destination)
            os.replace(staged[destination], destination)
        committed_successfully = True
    except Exception:
        # Restore in reverse commit order. Cleanup is deliberately best effort
        # so the original write error remains the reported failure.
        for destination in reversed(touched):
            backup = backups[destination]
            try:
                destination.unlink(missing_ok=True)
                if backup.exists():
                    os.replace(backup, destination)
            except OSError:
                pass
        raise
    finally:
        cleanup_paths = list(staged.values())
        if committed_successfully:
            cleanup_paths.extend(backups.values())
        for path in cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _discover_data_dictionary() -> Path:
    roots = list(dict.fromkeys(
        (Path.cwd().resolve(), Path(__file__).resolve().parents[2])))
    checked: list[Path] = []
    for root in roots:
        config_path = root / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            dependencies = Path(config["paths"]["dependencies_path"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if not dependencies.is_absolute():
            dependencies = (config_path.parent / dependencies).resolve()
        dictionary_dir = dependencies / "data_dictionary"
        checked.append(dictionary_dir)
        matches = sorted(dictionary_dir.glob("*current_data_dictionary*.csv"))
        exact = dictionary_dir / "current_data_dictionary.csv"
        if exact.is_file():
            return exact.resolve()
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            raise FileNotFoundError(
                "Multiple current data dictionaries were found; pass one "
                "explicitly with --data-dictionary: "
                + ", ".join(str(path.resolve()) for path in matches))
    locations = ", ".join(str(path) for path in checked) or "<none>"
    raise FileNotFoundError(
        "Could not discover a current REDCap data dictionary. Pass it with "
        f"--data-dictionary. Checked: {locations}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate REDCap calculated fields into validated Python and "
            "write conversion/dependency diagnostics."))
    parser.add_argument(
        "--data-dictionary",
        help=(
            "REDCap data-dictionary CSV. If omitted, config.json's "
            "dependencies/data_dictionary folder is searched."))
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: the data dictionary's parent).")
    parser.add_argument(
        "--output-json", default="converted_calculated_fields.json",
        help="Conversion artifact filename (default: %(default)s).")
    parser.add_argument(
        "--output-csv", default="converted_calculated_fields.csv",
        help="Audit CSV filename (default: %(default)s).")
    parser.add_argument(
        "--diagnostics", default="excluded_calculated_field_vars.json",
        help="Failed-conversion JSON filename (default: %(default)s).")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit nonzero when any calculated field is not converted.")
    return parser


def _resolve_output_paths(
        output_dir: Path, output_json: str, output_csv: str,
        diagnostics: str, source_path: Path) -> tuple[Path, Path, Path]:
    paths = tuple(
        (output_dir / name).expanduser().resolve()
        for name in (output_json, output_csv, diagnostics))
    if len(set(paths)) != len(paths):
        raise CalculationSchemaError(
            "--output-json, --output-csv, and --diagnostics must resolve "
            "to three distinct files")
    if source_path.resolve() in paths:
        raise CalculationSchemaError(
            "An output path resolves to the input data dictionary; refusing "
            "to overwrite the source")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dictionary_path = (
            Path(args.data_dictionary).expanduser().resolve()
            if args.data_dictionary else _discover_data_dictionary())
        if not dictionary_path.is_file():
            raise FileNotFoundError(
                f"Data dictionary does not exist: {dictionary_path}")
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir else dictionary_path.parent)
        json_path, csv_path, diagnostics_path = _resolve_output_paths(
            output_dir, args.output_json, args.output_csv,
            args.diagnostics, dictionary_path)
        source_bytes = dictionary_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        frame = pd.read_csv(
            io.BytesIO(source_bytes), keep_default_na=False, low_memory=False)
        transformer = TransformCalculatedFields(frame)
        artifact = transformer.build_artifact(
            dictionary_path, source_sha256=source_sha256)
        _write_artifact_set(
            artifact, artifact["calculations"],
            transformer.excluded_conversions,
            json_path, csv_path, diagnostics_path)
    except (CalculationSchemaError, FileNotFoundError, OSError,
            pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    summary = artifact["summary"]
    print(
        f"Converted {summary['converted_count']:,} of "
        f"{summary['calculated_field_count']:,} calculated fields; "
        f"{summary['excluded_count']:,} excluded.")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Diagnostics: {diagnostics_path}")
    if args.strict and summary["excluded_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
