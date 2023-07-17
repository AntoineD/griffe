"""This module contains the data classes that represent resolvable names and expressions."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence

from griffe.exceptions import NameResolutionError
from griffe.logger import LogLevel, get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from griffe.dataclasses import Class, Module


logger = get_logger(__name__)


class Name:
    """This class represents a Python object identified by a name in a given scope.

    Attributes:
        source: The name as written in the source code.
    """

    def __init__(self, source: str, full: str | Callable, *, first_attr_name: bool = True) -> None:
        """Initialize the name.

        Parameters:
            source: The name as written in the source code.
            full: The full, resolved name in the given scope, or a callable to resolve it later.
            first_attr_name: Whether this name is the first in a chain of names representing
                an attribute (dot separated strings).
        """
        self.source: str = source
        if isinstance(full, str):
            self._full: str = full
            self._resolver: Callable = lambda: None
        else:
            self._full = ""
            self._resolver = full
        self.first_attr_name: bool = first_attr_name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.full == other or self.brief == other
        if isinstance(other, Name):
            return self.full == other.full
        if isinstance(other, Expression):
            return self.full == other.source
        raise NotImplementedError(f"uncomparable types: {type(self)} and {type(other)}")

    def __repr__(self) -> str:
        return f"Name(source={self.source!r}, full={self.full!r})"

    def __str__(self) -> str:
        return self.source

    def __iter__(self) -> Iterator[Name]:
        yield self

    @property
    def brief(self) -> str:
        """Return the brief source name.

        Returns:
            The last part of the source name.
        """
        return self.source.rsplit(".", 1)[-1]

    @property
    def full(self) -> str:
        """Return the full, resolved name.

        If it was given when creating the name, return that.
        If a callable was given, call it and return its result.
        It the name cannot be resolved, return the source.

        Returns:
            The resolved name or the source.
        """
        if not self._full:
            try:
                self._full = self._resolver() or self.source
            except NameResolutionError:
                # probably a built-in
                self._full = self.source
        return self._full

    @property
    def canonical(self) -> str:
        """Return the canonical name (resolved one, not alias name).

        Returns:
            The canonical name.
        """
        return self.full.rsplit(".", 1)[-1]

    def as_dict(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        """Return this name's data as a dictionary.

        Parameters:
            **kwargs: Additional serialization options.

        Returns:
            A dictionary.
        """
        return {"source": self.source, "full": self.full}


class Expression(list):
    """This class represents a Python expression.

    For example, it can represent complex annotations such as:

    - `Optional[Dict[str, Tuple[int, bool]]]`
    - `str | Callable | list[int]`

    Expressions are simple lists containing strings, names or expressions.
    Each name in the expression can be resolved to its full name within its scope.
    """

    def __init__(self, *values: str | Expression | Name) -> None:
        """Initialize the expression.

        Parameters:
            *values: The initial values of the expression.
        """
        super().__init__()
        self.extend(values)

    def __str__(self):
        return "".join(str(element) for element in self)

    @property
    def source(self) -> str:
        """Return the expression as written in the source.

        This property is only useful to the AST utils.

        Returns:
            The expression as a string.
        """
        return str(self)

    @property
    def full(self) -> str:
        """Return the full expression as a string with canonical names (imported ones, not aliases).

        This property is only useful to the AST utils.

        Returns:
            The expression as a string.
        """
        parts = []
        for element in self:
            if isinstance(element, str):
                parts.append(element)
            elif isinstance(element, Name):
                parts.append(element.full if element.first_attr_name else element.canonical)
            else:
                parts.append(element.full)
        return "".join(parts)

    @property
    def kind(self) -> str:
        """Return the main type object as a string.

        Returns:
            The main type of this expression.
        """
        return str(self.non_optional).split("[", 1)[0].rsplit(".", 1)[-1].lower()

    @property
    def without_subscript(self) -> Expression:
        """The expression without the subscript part (if any).

        For example, `Generic[T]` becomes `Generic`.
        """
        parts = []
        for element in self:
            if isinstance(element, str) and element == "[":
                break
            parts.append(element)
        return Expression(*parts)

    @property
    def is_tuple(self) -> bool:
        """Tell whether this expression represents a tuple.

        Returns:
            True or False.
        """
        return self.kind == "tuple"

    @property
    def is_iterator(self) -> bool:
        """Tell whether this expression represents an iterator.

        Returns:
            True or False.
        """
        return self.kind == "iterator"

    @property
    def is_generator(self) -> bool:
        """Tell whether this expression represents a generator.

        Returns:
            True or False.
        """
        return self.kind == "generator"

    @property
    def is_classvar(self) -> bool:
        """Tell whether this expression represents a ClassVar.

        Returns:
            True or False.
        """
        return isinstance(self[0], Name) and self[0].full == "typing.ClassVar"

    @cached_property
    def non_optional(self) -> Expression:
        """Return the same expression as non-optional.

        This will return a new expression without
        the `Optional[]` or `| None` parts.

        Returns:
            A non-optional expression.
        """
        if self[-3:] == ["|", " ", "None"]:
            if isinstance(self[0], Expression):
                return self[0]
            return Expression(self[0])
        if self[:3] == ["None", " ", "|"]:
            if isinstance(self[3], Expression):
                return self[3]
            return Expression(self[3])
        if isinstance(self[0], Name) and self[0].full == "typing.Optional":
            if isinstance(self[2], Expression):
                return self[2]
            return Expression(self[2])
        return self

    def tuple_item(self, nth: int) -> str | Name:
        """Return the n-th item of this tuple expression.

        Parameters:
            nth: The item number.

        Returns:
            A string or name.
        """
        #  0  1     2     3
        #       N , N , N
        #       0 1 2 3 4
        return self.non_optional[2][2 * nth]

    def tuple_items(self) -> list[Name | Expression]:
        """Return a tuple items as a list.

        Returns:
            The tuple items.
        """
        return self.non_optional[2][::2]

    def iterator_item(self) -> Name | Expression:
        """Return the item of an iterator.

        Returns:
            The iterator item.
        """
        return self.non_optional[2]

    def generator_items(self) -> tuple[Name | Expression, Name | Expression, Name | Expression]:
        """Return the items of a generator.

        Returns:
            The yield type.
            The send/receive type.
            The return type.
        """
        return self.non_optional[2][0], self[2][2], self[2][4]


def _yield(element: str | Expr | tuple[str | Expr, ...]) -> Iterator[str | ExprName]:
    if isinstance(element, str):
        yield element
    else:
        yield from element


def _join(elements: Iterable[str | Expr], joint: str | Expr) -> Iterator[str | ExprName]:
    it = iter(elements)
    try:
        yield from _yield(next(it))
    except StopIteration:
        return
    for element in it:
        yield from _yield(joint)
        yield from _yield(element)


# TODO: merge in decorators once Python 3.9 is dropped
dataclass_opts = {"eq": False, "frozen": True}
if sys.version_info >= (3, 10):
    dataclass_opts["slots"] = True


class Expr:
    def __str__(self) -> str:
        return "".join(elem if isinstance(elem, str) else elem.source for elem in self)

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from ()


@dataclass(**dataclass_opts)
class ExprArgument(Expr):
    kind: str
    name: str | None = None
    annotation: Expr | None = None
    default: Expr | None = None


@dataclass(**dataclass_opts)
class ExprArguments(Expr):
    arguments: Sequence[ExprArgument]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _join(self.arguments, ", ")


@dataclass(**dataclass_opts)
class ExprAttribute(Expr):
    left: Expr
    right: ExprName

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.left
        yield "."
        yield self.right


@dataclass(**dataclass_opts)
class ExprBinOp(Expr):
    left: Expr
    operator: str
    right: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.left
        yield f" {self.operator} "
        yield from self.right


@dataclass(**dataclass_opts)
class ExprBoolOp(Expr):
    operator: str
    values: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _join(self.values, f" {self.operator} ")


@dataclass(**dataclass_opts)
class ExprCall(Expr):
    function: Expr
    arguments: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.function
        yield "("
        yield from _join(self.arguments, ", ")
        yield ")"


@dataclass(**dataclass_opts)
class ExprCompare(Expr):
    left: Expr
    operators: Sequence[Expr]
    comparators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.left
        yield " "
        yield from _join(zip_longest(self.operators, [], self.comparators, fillvalue=" "), " ")  # type: ignore[arg-type]


@dataclass(**dataclass_opts)
class ExprComprehension(Expr):
    target: Expr
    iterable: Expr
    conditions: Sequence[Expr]
    is_async: bool = False

    def __iter__(self) -> Iterator[str | ExprName]:
        if self.is_async:
            yield "async "
        yield "for "
        yield from self.target
        yield " in "
        yield from self.iterable
        if self.conditions:
            yield " if "
            yield from _join(self.conditions, " if ")


@dataclass(**dataclass_opts)
class ExprConstant(Expr):
    value: str

    def __iter__(self) -> Iterator[str | ExprName]:
        yield self.value


@dataclass(**dataclass_opts)
class ExprDict(Expr):
    keys: Sequence[Expr | None]
    values: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _join(
            (("None" if key is None else key, ": ", value) for key, value in zip(self.keys, self.values)),
            ", ",
        )
        yield "}"


@dataclass(**dataclass_opts)
class ExprDictComp(Expr):
    key: Expr
    value: Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from self.key
        yield ": "
        yield from self.value
        yield from _join(self.generators, " ")
        yield "}"


@dataclass(**dataclass_opts)
class ExprExtSlice(Expr):
    dims: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from _join(self.dims, ", ")


@dataclass(**dataclass_opts)
class ExprFormatted(Expr):
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from self.value
        yield "}"


@dataclass(**dataclass_opts)
class ExprGeneratorExp(Expr):
    element: Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.element
        yield " "
        yield from _join(self.generators, " ")


@dataclass(**dataclass_opts)
class ExprIfExp(Expr):
    body: Expr
    test: Expr
    orelse: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.body
        yield " if "
        yield from self.test
        yield " else "
        yield from self.orelse


@dataclass(**dataclass_opts)
class ExprJoinedStr(Expr):
    values: Sequence[str | Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "f'"
        yield from _join(self.values, "")
        yield "'"


@dataclass(**dataclass_opts)
class ExprKeyword(Expr):
    name: str
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield self.name
        yield "="
        yield from self.value


@dataclass(**dataclass_opts)
class ExprVarPositional(Expr):
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "*"
        yield from self.value


@dataclass(**dataclass_opts)
class ExprVarKeyword(Expr):
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "**"
        yield from self.value


@dataclass(**dataclass_opts)
class ExprLambda(Expr):
    arguments: Sequence[Expr]
    body: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "lambda "
        yield from _join(self.arguments, ", ")
        yield ": "
        yield from self.body


@dataclass(**dataclass_opts)
class ExprList(Expr):
    elements: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "["
        yield from _join(self.elements, ", ")
        yield "]"


@dataclass(**dataclass_opts)
class ExprListComp(Expr):
    element: Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "["
        yield from self.element
        yield " "
        yield from _join(self.generators, " ")
        yield "]"


# @dataclass(**dataclass_opts)
# class ExprName(Name, Expr):
#     ...
ExprName = Name


@dataclass(**dataclass_opts)
class ExprNamedExpr(Expr):
    target: Expr
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "("
        yield from self.target
        yield " := "
        yield from self.value
        yield ")"


@dataclass(**dataclass_opts)
class ExprSet(Expr):
    elements: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from _join(self.elements, ", ")
        yield "}"


@dataclass(**dataclass_opts)
class ExprSetComp(Expr):
    element: Expr
    generators: Sequence[Expr]

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "{"
        yield from self.element
        yield " "
        yield from _join(self.generators, " ")
        yield "}"


@dataclass(**dataclass_opts)
class ExprSlice(Expr):
    lower: Expr
    upper: Expr
    step: Expr | None = None

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.lower
        yield ":"
        yield from self.upper
        if self.step is not None:
            yield ":"
            yield from self.step


@dataclass(**dataclass_opts)
class ExprSubscript(Expr):
    left: Expr
    slice: Expr  # noqa: A003

    def __iter__(self) -> Iterator[str | ExprName]:
        yield from self.left
        yield "["
        yield from self.slice
        yield "]"


@dataclass(**dataclass_opts)
class ExprTuple(Expr):
    elements: Sequence[Expr]
    implicit: bool = False

    def __iter__(self) -> Iterator[str | ExprName]:
        if not self.implicit:
            yield "("
        yield from _join(self.elements, ", ")
        if not self.implicit:
            yield ")"


@dataclass(**dataclass_opts)
class ExprUnaryOp(Expr):
    operator: str
    value: Expr

    def __iter__(self) -> Iterator[str | ExprName]:
        yield self.operator
        yield from self.value


@dataclass(**dataclass_opts)
class ExprYield(Expr):
    value: Expr | None = None

    def __iter__(self) -> Iterator[str | ExprName]:
        yield "yield"
        if self.value is not None:
            yield " "
            yield from self.value


def _build_add(node: ast.Add, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "+"


def _build_and(node: ast.And, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "and"


def _build_arguments(node: ast.arguments, parent: Module | Class, **kwargs: Any) -> Expr:  # noqa: ARG001
    # TODO: better arguments handling
    return ExprArguments([ExprArgument("todo", arg.arg) for arg in node.args])


def _build_attribute(node: ast.Attribute, parent: Module | Class, **kwargs: Any) -> Expr:
    left = _build(node.value, parent, **kwargs)

    if isinstance(left, str):
        resolver = f"str.{node.attr}"
    else:

        def resolver() -> str:  # type: ignore[misc]
            return f"{left.full}.{node.attr}"

    right = ExprName(node.attr, resolver, first_attr_name=False)
    return ExprAttribute(left, right)


def _build_binop(node: ast.BinOp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprBinOp(
        _build(node.left, parent, **kwargs),
        _build(node.op, parent, **kwargs),
        _build(node.right, parent, **kwargs),
    )


def _build_bitand(node: ast.BitAnd, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "&"


def _build_bitor(node: ast.BitOr, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "|"


def _build_bitxor(node: ast.BitXor, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "^"


def _build_boolop(node: ast.BoolOp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprBoolOp(
        _build(node.op, parent, **kwargs),
        [_build(value, parent, **kwargs) for value in node.values],
    )


def _build_call(node: ast.Call, parent: Module | Class, **kwargs: Any) -> Expr:
    positional_args = [_build(arg, parent, **kwargs) for arg in node.args]
    keyword_args = [_build(kwarg, parent, **kwargs) for kwarg in node.keywords]
    return ExprCall(_build(node.func, parent, **kwargs), [*positional_args, *keyword_args])


def _build_compare(node: ast.Compare, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprCompare(
        _build(node.left, parent, **kwargs),
        [_build(op, parent, **kwargs) for op in node.ops],
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


def _build_div(node: ast.Div, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "/"


def _build_eq(node: ast.Eq, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "=="


def _build_floordiv(node: ast.FloorDiv, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "//"


def _build_formatted(node: ast.FormattedValue, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprFormatted(_build(node.value, parent, in_formatted_str=True, **kwargs))


def _build_generatorexp(node: ast.GeneratorExp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprGeneratorExp(
        _build(node.elt, parent, **kwargs),
        [_build(gen, parent, **kwargs) for gen in node.generators],
    )


def _build_gte(node: ast.GtE, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return ">="


def _build_gt(node: ast.Gt, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return ">"


def _build_ifexp(node: ast.IfExp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprIfExp(
        _build(node.body, parent, **kwargs),
        _build(node.test, parent, **kwargs),
        _build(node.orelse, parent, **kwargs),
    )


def _build_invert(node: ast.Invert, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "~"


def _build_in(node: ast.In, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "in"


def _build_is(node: ast.Is, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "is"


def _build_isnot(node: ast.IsNot, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "is not"


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
    # TODO: better arguments handling
    return ExprLambda(
        [ExprArgument("todo", arg.arg) for arg in node.args.args],
        _build(node.body, parent, **kwargs),
    )


def _build_list(node: ast.List, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprList([_build(el, parent, **kwargs) for el in node.elts])


def _build_listcomp(node: ast.ListComp, parent: Module | Class, **kwargs: Any) -> Expr:
    return Expression(_build(node.elt, parent, **kwargs), [_build(gen, parent, **kwargs) for gen in node.generators])


def _build_lshift(node: ast.LShift, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "<<"


def _build_lte(node: ast.LtE, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "<="


def _build_lt(node: ast.Lt, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "<"


def _build_matmult(node: ast.MatMult, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "@"


def _build_mod(node: ast.Mod, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "%"


def _build_mult(node: ast.Mult, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "*"


def _build_name(node: ast.Name, parent: Module | Class, **kwargs: Any) -> ExprName:  # noqa: ARG001
    return ExprName(node.id, partial(parent.resolve, node.id))


def _build_named_expr(node: ast.NamedExpr, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprNamedExpr(_build(node.target, parent, **kwargs), _build(node.value, parent, **kwargs))


def _build_not(node: ast.Not, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "not "


def _build_noteq(node: ast.NotEq, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "!="


def _build_notin(node: ast.NotIn, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "not in"


def _build_or(node: ast.Or, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "or"


def _build_pow(node: ast.Pow, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "**"


def _build_rshift(node: ast.RShift, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return ">>"


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


def _build_sub(node: ast.Sub, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "-"


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
        if left.full in {"typing.Literal", "typing_extensions.Literal"}:  # type: ignore[union-attr]
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


def _build_uadd(node: ast.UAdd, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "+"


def _build_unaryop(node: ast.UnaryOp, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprUnaryOp(_build(node.op, parent, **kwargs), _build(node.operand, parent, **kwargs))


def _build_usub(node: ast.USub, parent: Module | Class, **kwargs: Any) -> str:  # noqa: ARG001
    return "-"


def _build_yield(node: ast.Yield, parent: Module | Class, **kwargs: Any) -> Expr:
    return ExprYield(None if node.value is None else _build(node.value, parent, **kwargs))


_node_map: dict[type, Callable[[Any, Module | Class], str | Name | Expression]] = {
    ast.Add: _build_add,
    ast.And: _build_and,
    ast.arguments: _build_arguments,
    ast.Attribute: _build_attribute,
    ast.BinOp: _build_binop,
    ast.BitAnd: _build_bitand,
    ast.BitOr: _build_bitor,
    ast.BitXor: _build_bitxor,
    ast.BoolOp: _build_boolop,
    ast.Call: _build_call,
    ast.Compare: _build_compare,
    ast.comprehension: _build_comprehension,
    ast.Constant: _build_constant,
    ast.Dict: _build_dict,
    ast.DictComp: _build_dictcomp,
    ast.Div: _build_div,
    ast.Eq: _build_eq,
    ast.FloorDiv: _build_floordiv,
    ast.FormattedValue: _build_formatted,
    ast.GeneratorExp: _build_generatorexp,
    ast.Gt: _build_gt,
    ast.GtE: _build_gte,
    ast.IfExp: _build_ifexp,
    ast.In: _build_in,
    ast.Invert: _build_invert,
    ast.Is: _build_is,
    ast.IsNot: _build_isnot,
    ast.JoinedStr: _build_joinedstr,
    ast.keyword: _build_keyword,
    ast.Lambda: _build_lambda,
    ast.List: _build_list,
    ast.ListComp: _build_listcomp,
    ast.LShift: _build_lshift,
    ast.Lt: _build_lt,
    ast.LtE: _build_lte,
    ast.MatMult: _build_matmult,
    ast.Mod: _build_mod,
    ast.Mult: _build_mult,
    ast.Name: _build_name,
    ast.NamedExpr: _build_named_expr,
    ast.Not: _build_not,
    ast.NotEq: _build_noteq,
    ast.NotIn: _build_notin,
    ast.Or: _build_or,
    ast.Pow: _build_pow,
    ast.RShift: _build_rshift,
    ast.Set: _build_set,
    ast.SetComp: _build_setcomp,
    ast.Slice: _build_slice,
    ast.Starred: _build_starred,
    ast.Sub: _build_sub,
    ast.Subscript: _build_subscript,
    ast.Tuple: _build_tuple,
    ast.UAdd: _build_uadd,
    ast.UnaryOp: _build_unaryop,
    ast.USub: _build_usub,
    ast.Yield: _build_yield,
}

# TODO: remove once Python 3.8 support is dropped
if sys.version_info < (3, 9):

    def _build_extslice(node: ast.ExtSlice, parent: Module | Class, **kwargs: Any) -> Expr:
        return ExprExtSlice([_build(dim, parent, **kwargs) for dim in node.dims])

    def _build_index(node: ast.Index, parent: Module | Class, **kwargs: Any) -> str | Name | Expression:
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
) -> str | Name | Expression | None:
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
) -> str | Name | Expression | None:
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


__all__ = ["Expression", "Name"]
