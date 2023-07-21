"""This module contains the data classes that represent resolvable names and expressions."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from dataclasses import fields as getfields
from functools import partial
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence

from griffe.enumerations import ParameterKind
from griffe.exceptions import NameResolutionError
from griffe.logger import LogLevel, get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from griffe.dataclasses import Class, Module


logger = get_logger(__name__)


def _yield(element: str | Expr | tuple[str | Expr, ...]) -> Iterator[str | ExprName]:
    if isinstance(element, str):
        yield element
    elif isinstance(element, tuple):
        for elem in element:
            yield from _yield(elem)
    else:
        yield from element


def _join(elements: Iterable[str | Expr | tuple[str | Expr, ...]], joint: str | Expr) -> Iterator[str | ExprName]:
    it = iter(elements)
    try:
        yield from _yield(next(it))
    except StopIteration:
        return
    for element in it:
        yield from _yield(joint)
        yield from _yield(element)


def _field_as_dict(
    element: str | bool | Expr | list[str | Expr] | None,
    **kwargs: Any,
) -> str | bool | None | list | dict:
    if isinstance(element, Expr):
        return _expr_as_dict(element, **kwargs)
    if isinstance(element, list):
        return [_field_as_dict(elem, **kwargs) for elem in element]
    return element


def _expr_as_dict(expression: Expr, **kwargs: Any) -> dict[str, Any]:
    fields = {
        field.name: _field_as_dict(getattr(expression, field.name), **kwargs)
        for field in sorted(getfields(expression), key=lambda f: f.name)
        if field.name != "parent"
    }
    fields["cls"] = expression.__class__.__name__
    return fields


# TODO: merge in decorators once Python 3.9 is dropped
dataclass_opts: dict[str, bool] = {}
if sys.version_info >= (3, 10):
    dataclass_opts["slots"] = True


@dataclass
class Expr:
    def __str__(self) -> str:
        return "".join(elem if isinstance(elem, str) else elem.name for elem in self)

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from ()

    def as_dict(self, **kwargs: Any) -> dict[str, Any]:
        return _expr_as_dict(self, **kwargs)

    @property
    def path(self) -> str:
        return str(self)

    @property
    def canonical_path(self) -> str:
        return str(self)

    @property
    def canonical_name(self) -> str:
        return self.canonical_path.rsplit(".", 1)[-1]

    @property
    def is_classvar(self) -> bool:
        return isinstance(self, ExprSubscript) and self.canonical_name == "ClassVar"

    @property
    def is_tuple(self) -> bool:
        return isinstance(self, ExprSubscript) and self.canonical_name.lower() == "tuple"

    @property
    def is_iterator(self) -> bool:
        return isinstance(self, ExprSubscript) and self.canonical_name == "Iterator"

    @property
    def is_generator(self) -> bool:
        return isinstance(self, ExprSubscript) and self.canonical_name == "Generator"


@dataclass(eq=False, **dataclass_opts)
class ExprAttribute(Expr):
    left: str | Expr
    right: ExprName

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _yield(self.left)
        yield "."
        yield self.right

    @property
    def path(self) -> str:
        return self.right.path

    @property
    def canonical_path(self) -> str:
        return self.right.canonical_path


@dataclass(eq=False, **dataclass_opts)
class ExprBinOp(Expr):
    left: str | Expr
    operator: str
    right: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _yield(self.left)
        yield f" {self.operator} "
        yield from _yield(self.right)


@dataclass(eq=False, **dataclass_opts)
class ExprBoolOp(Expr):
    operator: str
    values: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _join(self.values, f" {self.operator} ")


@dataclass(eq=False, **dataclass_opts)
class ExprCall(Expr):
    function: Expr
    arguments: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.function
        yield "("
        yield from _join(self.arguments, ", ")
        yield ")"


@dataclass(eq=False, **dataclass_opts)
class ExprCompare(Expr):
    left: str | Expr
    operators: Sequence[str]
    comparators: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _yield(self.left)
        yield " "
        yield from _join(zip_longest(self.operators, [], self.comparators, fillvalue=" "), " ")


@dataclass(eq=False, **dataclass_opts)
class ExprComprehension(Expr):
    target: str | Expr
    iterable: str | Expr
    conditions: Sequence[str | Expr]
    is_async: bool = False

    def __iter__(self) -> Iterator[str | ExprName]:
        if self.is_async:
            yield "async "
        yield "for "
        yield from _yield(self.target)
        yield " in "
        yield from _yield(self.iterable)
        if self.conditions:
            yield " if "
            yield from _join(self.conditions, " if ")


@dataclass(eq=False, **dataclass_opts)
class ExprConstant(Expr):
    value: str

    def __iter__(self) -> Iterator[str | ExprName]:
        yield self.value


@dataclass(eq=False, **dataclass_opts)
class ExprDict(Expr):
    keys: Sequence[str | Expr | None]
    values: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _join(
            (("None" if key is None else key, ": ", value) for key, value in zip(self.keys, self.values)),
            ", ",
        )
        yield "}"


@dataclass(eq=False, **dataclass_opts)
class ExprDictComp(Expr):
    key: str | Expr
    value: str | Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _yield(self.key)
        yield ": "
        yield from _yield(self.value)
        yield from _join(self.generators, " ")
        yield "}"


@dataclass(eq=False, **dataclass_opts)
class ExprExtSlice(Expr):
    dims: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _join(self.dims, ", ")


@dataclass(eq=False, **dataclass_opts)
class ExprFormatted(Expr):
    value: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _yield(self.value)
        yield "}"


@dataclass(eq=False, **dataclass_opts)
class ExprGeneratorExp(Expr):
    element: str | Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _yield(self.element)
        yield " "
        yield from _join(self.generators, " ")


@dataclass(eq=False, **dataclass_opts)
class ExprIfExp(Expr):
    body: str | Expr
    test: str | Expr
    orelse: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _yield(self.body)
        yield " if "
        yield from _yield(self.test)
        yield " else "
        yield from _yield(self.orelse)


@dataclass(eq=False, **dataclass_opts)
class ExprJoinedStr(Expr):
    values: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "f'"
        yield from _join(self.values, "")
        yield "'"


@dataclass(eq=False, **dataclass_opts)
class ExprKeyword(Expr):
    name: str
    value: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield self.name
        yield "="
        yield from _yield(self.value)


@dataclass(eq=False, **dataclass_opts)
class ExprVarPositional(Expr):
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "*"
        yield from self.value


@dataclass(eq=False, **dataclass_opts)
class ExprVarKeyword(Expr):
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "**"
        yield from self.value


@dataclass(eq=False, **dataclass_opts)
class ExprLambda(Expr):
    parameters: Sequence[ExprParameter]
    body: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "lambda "
        yield from _join(self.parameters, ", ")
        yield ": "
        yield from _yield(self.body)


@dataclass(eq=False, **dataclass_opts)
class ExprList(Expr):
    elements: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "["
        yield from _join(self.elements, ", ")
        yield "]"


@dataclass(eq=False, **dataclass_opts)
class ExprListComp(Expr):
    element: str | Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "["
        yield from _yield(self.element)
        yield " "
        yield from _join(self.generators, " ")
        yield "]"


@dataclass(eq=True, **dataclass_opts)
class ExprName(Expr):
    """This class represents a Python object identified by a name in a given scope.

    Attributes:
        source: The name as written in the source code.
    """

    name: str
    parent: ExprName | Module | Class | None = None

    def __iter__(self) -> Iterator[ExprName]:
        yield self

    @property
    def path(self) -> str:
        """Return the full, resolved name.

        If it was given when creating the name, return that.
        If a callable was given, call it and return its result.
        It the name cannot be resolved, return the source.

        Returns:
            The resolved name or the source.
        """
        if isinstance(self.parent, ExprName):
            return f"{self.parent.path}.{self.name}"
        return self.name

    @property
    def canonical_path(self) -> str:
        """Return the canonical name (resolved one, not alias name).

        Returns:
            The canonical name.
        """
        if self.parent is None:
            return self.name
        if isinstance(self.parent, ExprName):
            return f"{self.parent.canonical_path}.{self.name}"
        try:
            return self.parent.resolve(self.name)
        except NameResolutionError:
            return self.name

    @property
    def brief(self) -> str:
        return self.canonical_name

    @property
    def source(self) -> str:
        return self.name


@dataclass(eq=False, **dataclass_opts)
class ExprNamedExpr(Expr):
    target: Expr
    value: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "("
        yield from self.target
        yield " := "
        yield from _yield(self.value)
        yield ")"


@dataclass(eq=False, **dataclass_opts)
class ExprParameter(Expr):
    kind: str
    name: str | None = None
    annotation: Expr | None = None
    default: Expr | None = None


@dataclass(eq=False, **dataclass_opts)
class ExprSet(Expr):
    elements: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _join(self.elements, ", ")
        yield "}"


@dataclass(eq=False, **dataclass_opts)
class ExprSetComp(Expr):
    element: str | Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _yield(self.element)
        yield " "
        yield from _join(self.generators, " ")
        yield "}"


@dataclass(eq=False, **dataclass_opts)
class ExprSlice(Expr):
    lower: str | Expr | None = None
    upper: str | Expr | None = None
    step: str | Expr | None = None

    def __iter__(self) -> Iterator[str | ExprName]:
        if self.lower is not None:
            yield from _yield(self.lower)
        yield ":"
        if self.upper is not None:
            yield from _yield(self.upper)
        if self.step is not None:
            yield ":"
            yield from _yield(self.step)


@dataclass(eq=False, **dataclass_opts)
class ExprSubscript(Expr):
    left: Expr
    slice: Expr  # noqa: A003

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.left
        yield "["
        yield from self.slice
        yield "]"

    @property
    def path(self) -> str:
        return self.left.path

    @property
    def canonical_path(self) -> str:
        return self.left.canonical_path


@dataclass(eq=False, **dataclass_opts)
class ExprTuple(Expr):
    elements: Sequence[str | Expr]
    implicit: bool = False

    def __iter__(self) -> Iterator[str | ExprName]:
        if not self.implicit:
            yield "("
        yield from _join(self.elements, ", ")
        if not self.implicit:
            yield ")"


@dataclass(eq=False, **dataclass_opts)
class ExprUnaryOp(Expr):
    operator: str
    value: str | Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield self.operator
        yield from _yield(self.value)


@dataclass(eq=False, **dataclass_opts)
class ExprYield(Expr):
    value: str | Expr | None = None

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "yield"
        if self.value is not None:
            yield " "
            yield from _yield(self.value)


_unary_op_map = {
    ast.Invert: "~",
    ast.Not: "not ",
    ast.UAdd: "+",
    ast.USub: "-",
}

_binary_op_map = {
    ast.Add: "+",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.LShift: "<<",
    ast.MatMult: "@",
    ast.Mod: "%",
    ast.Mult: "*",
    ast.Pow: "**",
    ast.RShift: ">>",
    ast.Sub: "-",
}

_bool_op_map = {
    ast.And: "and",
    ast.Or: "or",
}

_compare_op_map = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}


def _build_attribute(node: ast.Attribute, parent: Module | Class, **kwargs: Any) -> Expr:
    left = _build(node.value, parent, **kwargs)
    if isinstance(left, ExprName):
        name_parent = left
    elif isinstance(left, ExprAttribute):
        name_parent = left.right
    else:
        name_parent = None
    right = ExprName(node.attr, name_parent)
    return ExprAttribute(left, right)


def _build_binop(node: ast.BinOp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprBinOp(
        _build(node.left, parent, **kwargs),
        _binary_op_map[type(node.op)],
        _build(node.right, parent, **kwargs),
    )


def _build_boolop(node: ast.BoolOp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprBoolOp(
        _bool_op_map[type(node.op)],
        [_build(value, parent, **kwargs) for value in node.values],
    )


def _build_call(node: ast.Call, parent: Module | Class, **kwargs: Any) -> Expr:
    positional_args = [_build(arg, parent, **kwargs) for arg in node.args]
    keyword_args = [_build(kwarg, parent, **kwargs) for kwarg in node.keywords]
    return ExprCall(_build(node.func, parent, **kwargs), [*positional_args, *keyword_args])


def _build_compare(node: ast.Compare, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprCompare(
        _build(node.left, parent, **kwargs),
        [_compare_op_map[type(op)] for op in node.ops],
        [_build(comp, parent, **kwargs) for comp in node.comparators],
    )


def _build_comprehension(node: ast.comprehension, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprComprehension(
        _build(node.target, parent, **kwargs),
        _build(node.iter, parent, **kwargs),
        [_build(condition, parent, **kwargs) for condition in node.ifs],
        is_async=bool(node.is_async),
    )


def _build_constant(
    node: ast.Constant,
    parent: Module | Class,
    *,
    in_formatted_str: bool = False,
    in_joined_str: bool = False,
    parse_strings: bool = False,
    literal_strings: bool = False,
    **kwargs: Any,
) -> str | Expr:
    if isinstance(node.value, str):
        if in_joined_str and not in_formatted_str:
            # We're in a f-string, not in a formatted value, don't keep quotes.
            return node.value
        if parse_strings and not literal_strings:
            # We're in a place where a string could be a type annotation
            # (and not in a Literal[...] type annotation).
            # We parse the string and build from the resulting nodes again.
            # If we fail to parse it (syntax errors), we consider it's a literal string and log a message.
            try:
                parsed = compile(
                    node.value,
                    mode="eval",
                    filename="<string-annotation>",
                    flags=ast.PyCF_ONLY_AST,
                    optimize=1,
                )
            except SyntaxError:
                logger.debug(
                    f"Tried and failed to parse {node.value!r} as Python code, "
                    "falling back to using it as a string literal "
                    "(postponed annotations might help: https://peps.python.org/pep-0563/)",
                )
            else:
                return _build(parsed.body, parent, **kwargs)  # type: ignore[attr-defined]
    return {type(...): lambda _: "..."}.get(type(node.value), repr)(node.value)


def _build_dict(node: ast.Dict, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprDict(
        [None if key is None else _build(key, parent, **kwargs) for key in node.keys],
        [_build(value, parent, **kwargs) for value in node.values],
    )


def _build_dictcomp(node: ast.DictComp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprDictComp(
        _build(node.key, parent, **kwargs),
        _build(node.value, parent, **kwargs),
        [_build(gen, parent, **kwargs) for gen in node.generators],
    )


def _build_formatted(node: ast.FormattedValue, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprFormatted(_build(node.value, parent, in_formatted_str=True, **kwargs))


def _build_generatorexp(node: ast.GeneratorExp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprGeneratorExp(
        _build(node.elt, parent, **kwargs),
        [_build(gen, parent, **kwargs) for gen in node.generators],
    )


def _build_ifexp(node: ast.IfExp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprIfExp(
        _build(node.body, parent, **kwargs),
        _build(node.test, parent, **kwargs),
        _build(node.orelse, parent, **kwargs),
    )


def _build_joinedstr(
    node: ast.JoinedStr,
    parent: Module | Class,
    *,
    in_joined_str: bool = False,  # noqa: ARG001
    **kwargs: Any,
) -> Expr:
    return ExprJoinedStr([_build(value, parent, in_joined_str=True, **kwargs) for value in node.values])


def _build_keyword(node: ast.keyword, parent: Module | Class, **kwargs: Any) -> Expr:
    if node.arg is None:
        return ExprVarKeyword(_build(node.value, parent, **kwargs))
    return ExprKeyword(node.arg, _build(node.value, parent, **kwargs))


def _build_lambda(node: ast.Lambda, parent: Module | Class, **kwargs: Any) -> Expr:
    # TODO: better parameter handling
    return ExprLambda(
        [ExprParameter(ParameterKind.positional_or_keyword.value, arg.arg) for arg in node.args.args],
        _build(node.body, parent, **kwargs),
    )


def _build_list(node: ast.List, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprList([_build(el, parent, **kwargs) for el in node.elts])


def _build_listcomp(node: ast.ListComp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprListComp(_build(node.elt, parent, **kwargs), [_build(gen, parent, **kwargs) for gen in node.generators])


def _build_name(node: ast.Name, parent: Module | Class, **kwargs: Any) -> Expr:  # noqa: ARG001
    return ExprName(node.id, parent)


def _build_named_expr(node: ast.NamedExpr, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprNamedExpr(_build(node.target, parent, **kwargs), _build(node.value, parent, **kwargs))


def _build_set(node: ast.Set, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprSet([_build(el, parent, **kwargs) for el in node.elts])


def _build_setcomp(node: ast.SetComp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprSetComp(_build(node.elt, parent, **kwargs), [_build(gen, parent, **kwargs) for gen in node.generators])


def _build_slice(node: ast.Slice, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprSlice(
        None if node.lower is None else _build(node.lower, parent, **kwargs),
        None if node.upper is None else _build(node.upper, parent, **kwargs),
        None if node.step is None else _build(node.step, parent, **kwargs),
    )


def _build_starred(node: ast.Starred, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprVarPositional(_build(node.value, parent, **kwargs))


def _build_subscript(
    node: ast.Subscript,
    parent: Module | Class,
    *,
    parse_strings: bool = False,
    literal_strings: bool = False,
    in_subscript: bool = False,  # noqa: ARG001
    **kwargs: Any,
) -> Expr:
    left = _build(node.value, parent, **kwargs)
    if parse_strings:
        if isinstance(left, (ExprAttribute, ExprName)) and left.canonical_path in {
            "typing.Literal",
            "typing_extensions.Literal",
        }:
            literal_strings = True
        slice = _build(
            node.slice,
            parent,
            parse_strings=True,
            literal_strings=literal_strings,
            in_subscript=True,
            **kwargs,
        )
    else:
        slice = _build(node.slice, parent, in_subscript=True, **kwargs)
    return ExprSubscript(left, slice)


def _build_tuple(
    node: ast.Tuple,
    parent: Module | Class,
    *,
    in_subscript: bool = False,
    **kwargs: Any,
) -> Expr:
    return ExprTuple([_build(el, parent, **kwargs) for el in node.elts], implicit=in_subscript)


def _build_unaryop(node: ast.UnaryOp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprUnaryOp(_unary_op_map[type(node.op)], _build(node.operand, parent, **kwargs))


def _build_yield(node: ast.Yield, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprYield(None if node.value is None else _build(node.value, parent, **kwargs))


_node_map: dict[type, Callable[[Any, Module | Class], Expr]] = {
    ast.Attribute: _build_attribute,
    ast.BinOp: _build_binop,
    ast.BoolOp: _build_boolop,
    ast.Call: _build_call,
    ast.Compare: _build_compare,
    ast.comprehension: _build_comprehension,
    ast.Constant: _build_constant,  # type: ignore[dict-item]
    ast.Dict: _build_dict,
    ast.DictComp: _build_dictcomp,
    ast.FormattedValue: _build_formatted,
    ast.GeneratorExp: _build_generatorexp,
    ast.IfExp: _build_ifexp,
    ast.JoinedStr: _build_joinedstr,
    ast.keyword: _build_keyword,
    ast.Lambda: _build_lambda,
    ast.List: _build_list,
    ast.ListComp: _build_listcomp,
    ast.Name: _build_name,
    ast.NamedExpr: _build_named_expr,
    ast.Set: _build_set,
    ast.SetComp: _build_setcomp,
    ast.Slice: _build_slice,
    ast.Starred: _build_starred,
    ast.Subscript: _build_subscript,
    ast.Tuple: _build_tuple,
    ast.UnaryOp: _build_unaryop,
    ast.Yield: _build_yield,
}

# TODO: remove once Python 3.8 support is dropped
if sys.version_info < (3, 9):

    def _build_extslice(node: ast.ExtSlice, parent: Module | Class, **kwargs: Any) -> Expr:
        return ExprExtSlice([_build(dim, parent, **kwargs) for dim in node.dims])

    def _build_index(node: ast.Index, parent: Module | Class, **kwargs: Any) -> Expr:
        return _build(node.value, parent, **kwargs)

    _node_map[ast.ExtSlice] = _build_extslice
    _node_map[ast.Index] = _build_index


def _build(node: ast.AST, parent: Module | Class, **kwargs: Any) -> Expr:
    return _node_map[type(node)](node, parent, **kwargs)


def get_expression(
    node: ast.AST | None,
    parent: Module | Class,
    *,
    parse_strings: bool | None = None,
) -> Expr | None:
    """Build an expression from an AST.

    Parameters:
        node: The annotation node.
        parent: The parent used to resolve the name.
        parse_strings: Whether to try and parse strings as type annotations.

    Returns:
        A string or resovable name or expression.
    """
    if node is None:
        return None
    if parse_strings is None:
        try:
            module = parent.module
        except ValueError:
            parse_strings = False
        else:
            parse_strings = not module.imports_future_annotations
    return _build(node, parent, parse_strings=parse_strings)


def safe_get_expression(
    node: ast.AST | None,
    parent: Module | Class,
    *,
    parse_strings: bool | None = None,
    log_level: LogLevel | None = LogLevel.error,
    msg_format: str = "{path}:{lineno}: Failed to get expression from {node_class}: {error}",
) -> Expr | None:
    """Safely (no exception) build a resolvable annotation.

    Parameters:
        node: The annotation node.
        parent: The parent used to resolve the name.
        parse_strings: Whether to try and parse strings as type annotations.
        log_level: Log level to use to log a message. None to disable logging.
        msg_format: A format string for the log message. Available placeholders:
            path, lineno, node, error.

    Returns:
        A string or resovable name or expression.
    """
    try:
        return get_expression(node, parent, parse_strings=parse_strings)
    except Exception as error:  # noqa: BLE001
        if log_level is None:
            return None
        node_class = node.__class__.__name__
        try:
            path: Path | str = parent.relative_filepath
        except ValueError:
            path = "<in-memory>"
        lineno = node.lineno  # type: ignore[union-attr]
        message = msg_format.format(path=path, lineno=lineno, node_class=node_class, error=error)
        getattr(logger, log_level.value)(message)
    return None


_msg_format = "{path}:{lineno}: Failed to get %s expression from {node_class}: {error}"
get_annotation = partial(get_expression, parse_strings=None)
safe_get_annotation = partial(
    safe_get_expression,
    parse_strings=None,
    msg_format=_msg_format % "annotation",
)
get_base_class = partial(get_expression, parse_strings=False)
safe_get_base_class = partial(
    safe_get_expression,
    parse_strings=False,
    msg_format=_msg_format % "base class",
)
get_condition = partial(get_expression, parse_strings=False)
safe_get_condition = partial(
    safe_get_expression,
    parse_strings=False,
    msg_format=_msg_format % "condition",
)


__all__ = [
    "Expr",
    "ExprAttribute",
    "ExprBinOp",
    "ExprBoolOp",
    "ExprCall",
    "ExprCompare",
    "ExprComprehension",
    "ExprConstant",
    "ExprDict",
    "ExprDictComp",
    "ExprExtSlice",
    "ExprFormatted",
    "ExprGeneratorExp",
    "ExprIfExp",
    "ExprJoinedStr",
    "ExprKeyword",
    "ExprVarPositional",
    "ExprVarKeyword",
    "ExprLambda",
    "ExprList",
    "ExprListComp",
    "ExprName",
    "ExprNamedExpr",
    "ExprParameter",
    "ExprSet",
    "ExprSetComp",
    "ExprSlice",
    "ExprSubscript",
    "ExprTuple",
    "ExprUnaryOp",
    "ExprYield",
    "get_annotation",
    "get_base_class",
    "get_condition",
    "get_expression",
    "safe_get_annotation",
    "safe_get_base_class",
    "safe_get_condition",
    "safe_get_expression",
]
