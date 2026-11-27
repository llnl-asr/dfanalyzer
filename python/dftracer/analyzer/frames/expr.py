"""Evaluate the pandas-syntax strings of presets, layers and fact rules on library frames."""

import ast
import io
import math
import operator
import tokenize
from collections.abc import Callable, Mapping
from typing import Any

from dftracer.utils import DataFrame, DType, Series
from dftracer.utils.errors import DFTUtilsTranspileError

_BOOL_TOKENS = {"&": "and", "|": "or", "~": "not"}
# `@name` parses as this prefix plus the name, the prefix pandas eval uses, so a local never shadows a column.
_LOCAL = "__pd_eval_local_"

_COMPARE: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def evaluate(frame: DataFrame, expr: str, names: Mapping[str, Any] | None = None) -> Any:
    """Evaluate a pandas `DataFrame.eval` expression on a library frame.

    `@name` and bare names that are not columns read from `names`. An assignment
    `col = expr` returns a copy of the frame with that column set.
    """
    names = dict(names or {})
    target, body = _parse(expr)
    if "@" not in expr and not names and not _native_differs(body, frame):
        try:
            return frame.eval(expr)
        except (DFTUtilsTranspileError, ValueError):
            pass
    # The native eval has no form for `@name` or a None in `.isin`, reads `"a" in s` as a
    # substring test, and its `==`/`!=` on a missing text value give a missing value where
    # pandas gives False/True (dftracer-utils feat/analyzer-port-gaps); evaluate those in Python.
    value = _Evaluator(frame, names).run(body)
    if target is None:
        return value
    out = frame.copy()
    out[target] = value if isinstance(value, Series) else Series([value] * out.num_rows)
    return out


def query(frame: DataFrame, expr: str, names: Mapping[str, Any] | None = None) -> DataFrame:
    """Rows of `frame` where the pandas `query` expression holds; a missing value is false."""
    mask = evaluate(frame, expr, names)
    return frame.filter(mask.fillna(False))


def expression_names(expr: str) -> set[str]:
    """The column names a pandas `eval` expression reads; `@name` locals and an assignment target are left out."""
    _, body = _parse(expr)
    return {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and not n.id.startswith(_LOCAL)}


def _pandas_source(expr: str) -> str:
    """`expr` as Python source with pandas precedence: `&`, `|` and `~` bind looser than comparisons."""
    tokens = []
    local = False
    for tok in tokenize.generate_tokens(io.StringIO(expr.strip()).readline):
        if tok.type == tokenize.OP and tok.string in _BOOL_TOKENS:
            tok = tok._replace(type=tokenize.NAME, string=_BOOL_TOKENS[tok.string])
        elif tok.type == tokenize.OP and tok.string == "@":
            local = True
            continue
        elif local and tok.type == tokenize.NAME:
            tok = tok._replace(string=_LOCAL + tok.string)
        local = False
        tokens.append((tok.type, tok.string))
    return tokenize.untokenize(tokens).strip()


def _parse(expr: str) -> tuple[str | None, ast.expr]:
    """The assignment target of `expr` (None without one) and the tree of its value."""
    tree = ast.parse(_pandas_source(expr), mode="exec")
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign):
        node = tree.body[0]
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id, node.value
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        return None, tree.body[0].value
    raise ValueError(f"not a pandas eval expression: {expr!r}")


def _native_differs(tree: ast.expr, frame: DataFrame) -> bool:
    """Whether `tree` has a form the native eval answers unlike pandas: `==`/`!=` with a string or on
    text, `in`, or `.isin` (a None in the list matches a missing text value)."""
    columns = set(frame.columns)
    levels = set(frame.index_names())

    def reads_text(node: ast.AST) -> bool:
        for name in ast.walk(node):
            if isinstance(name, ast.Name) and name.id in columns and frame[name.id].dtype.is_text:
                return True
            if isinstance(name, ast.Name) and name.id in levels and frame.get_level_values(name.id).dtype.is_text:
                return True
        return False

    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "isin":
            return True
        if not isinstance(n, ast.Compare):
            continue
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops):
            return True
        if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in n.ops):
            continue
        operands = [n.left, *n.comparators]
        if any(isinstance(c, ast.Constant) and isinstance(c.value, str) for c in operands):
            return True
        if any(reads_text(c) for c in operands):
            return True
    return False


def _object_compare(op: ast.cmpop, left: Any, right: Any) -> Series | None:
    """`left op right` for `==`/`!=` of a Series and a scalar as pandas compares objects; None for other operands.

    A text column against a string: a missing value is unequal. A text column against a non-string
    scalar: unequal, missing stays missing. Another column against a string: unequal on every row.
    Against None: missing on every row of a text column, else unequal with missing kept.
    A text column against a non-text column: unequal, missing where either is missing.
    """
    if not isinstance(op, (ast.Eq, ast.NotEq)):
        return None
    if isinstance(left, Series) and isinstance(right, Series):
        if left.dtype.is_text == right.dtype.is_text:
            return None
        both = left.is_not_null() & right.is_not_null()
        return _constant(left, isinstance(op, ast.NotEq)).where(both)
    if isinstance(right, Series):
        left, right = right, left
    if not isinstance(left, Series) or isinstance(right, Series):
        return None
    ne = isinstance(op, ast.NotEq)
    if left.dtype.is_text and isinstance(right, float) and math.isnan(right):
        right = None
    if right is None:
        return _constant(left, ne).where(_constant(left, False) if left.dtype.is_text else left.is_not_null())
    if left.dtype.is_text and isinstance(right, str):
        return _COMPARE[type(op)](left, right).fillna(ne)
    if left.dtype.is_text:
        return _constant(left, ne).where(left.is_not_null())
    if isinstance(right, str):
        return _constant(left, ne)
    return None


def _bool_as_number(left: Any, right: Any) -> tuple[Any, Any]:
    """A bool column against a number compares as 0/1, as in pandas; the library compare rejects the mix."""

    def number(v: Any) -> bool:
        return isinstance(v, (bool, int, float)) and not isinstance(v, Series)

    def boolean(v: Any) -> bool:
        return isinstance(v, Series) and v.dtype == DType.BOOL

    def numeric(v: Any) -> bool:
        return isinstance(v, Series) and (v.dtype == DType.BOOL or v.dtype.is_numeric)

    if boolean(left) and number(right):
        return left.astype("float64"), float(right)
    if boolean(right) and number(left):
        return float(left), right.astype("float64")
    if (boolean(left) or boolean(right)) and numeric(left) and numeric(right):
        return left.astype("float64"), right.astype("float64")
    return left, right


def _isin(values: Series, items: Any) -> Series:
    """`values.isin(items)` as pandas: a missing text value matches a None or NaN in `items`."""
    if isinstance(items, (str, bytes)):
        raise TypeError(f"isin takes a list-like, not {items!r}")
    items = list(items)
    if not values.dtype.is_text:
        return values.isin(items)
    missing = [v is None or (isinstance(v, float) and math.isnan(v)) for v in items]
    out = values.isin([v for v, m in zip(items, missing) if not m])
    return out | values.is_null() if any(missing) else out


def _constant(like: Series, value: bool) -> Series:
    """`value` on every row of `like`, never missing."""
    return (like.is_null() | like.is_not_null()) if value else (like.is_null() & like.is_not_null())


def _logical(op: type, a: Any, b: Any) -> Any:
    """`a and b` or `a or b` as pandas `&` and `|`: element-wise on a Series, a missing value as unknown."""
    if not isinstance(a, Series) and not isinstance(b, Series):
        return (a and b) if op is ast.And else (a or b)
    if not isinstance(a, Series):
        a, b = b, a
    if isinstance(b, Series):
        return a & b if op is ast.And else a | b
    if op is ast.And:
        return a if b else _constant(a, False)
    return _constant(a, True) if b else a


class _Evaluator:
    def __init__(self, frame: DataFrame, names: Mapping[str, Any]) -> None:
        self._frame = frame
        self._names = names
        self._columns = set(frame.columns) | set(frame.index_names())

    def run(self, tree: ast.expr) -> Any:
        return self._eval(tree)

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id.startswith(_LOCAL):
                return self._names[node.id[len(_LOCAL) :]]
            if node.id in self._columns:
                if node.id in self._frame.columns:
                    return self._frame[node.id]
                return self._frame.get_level_values(node.id)
            if node.id in self._names:
                return self._names[node.id]
            raise KeyError(f"no column named {node.id!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.BoolOp):
            values = [self._eval(v) for v in node.values]
            out = values[0]
            for v in values[1:]:
                out = _logical(type(node.op), out, v)
            return out
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return ~value if isinstance(value, Series) else not value
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            out = None
            for op, right_node in zip(node.ops, node.comparators):
                right = self._eval(right_node)
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, Series) and not isinstance(left, Series):
                    if not isinstance(left, str):
                        # pandas tests the index labels of `right` here, not its values.
                        raise TypeError(f"`{left!r} in <column>` tests index labels in pandas; use .isin")
                    # pandas rewrites a string `in` a column as `==`, and `not in` as `!=`.
                    step = _object_compare(ast.Eq() if isinstance(op, ast.In) else ast.NotEq(), right, left)
                elif isinstance(op, (ast.In, ast.NotIn)) and not isinstance(left, Series):
                    step = (left in right) == isinstance(op, ast.In)
                elif isinstance(op, (ast.In, ast.NotIn)):
                    step = _isin(left, right)
                    if isinstance(op, ast.NotIn):
                        step = ~step
                else:
                    a, b = _bool_as_number(left, right)
                    step = _object_compare(op, a, b)
                    if step is None:
                        step = _COMPARE[type(op)](a, b)
                out = step if out is None else _logical(ast.And, out, step)
                left = right
            return out
        if isinstance(node, ast.Attribute):
            return getattr(self._eval(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        if isinstance(node, ast.Call):
            args = [self._eval(a) for a in node.args]
            kwargs = {k.arg: self._eval(k.value) for k in node.keywords if k.arg is not None}
            if isinstance(node.func, ast.Attribute) and node.func.attr == "isin" and len(args) + len(kwargs) == 1:
                owner = self._eval(node.func.value)
                if isinstance(owner, Series) and (args or "values" in kwargs):
                    return _isin(owner, args[0] if args else kwargs["values"])
            return self._eval(node.func)(*args, **kwargs)
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    def _subscript(self, node: ast.Subscript) -> Any:
        index = self._eval(node.slice)
        # pandas `.str[i]` picks the i-th part of a list cell and the i-th character of a string.
        if isinstance(node.value, ast.Attribute) and node.value.attr == "str":
            values = self._eval(node.value.value)
            if values.dtype == DType.LIST:
                return values.list_get(index)
            # str_slice counts bytes; the regex counts characters and gives a missing value out of range.
            pattern = rf"(?s)^.{{{index}}}(.)" if index >= 0 else rf"(?s)(.).{{{-index - 1}}}\z"
            return values.str_extract(pattern)
        return self._eval(node.value)[index]
