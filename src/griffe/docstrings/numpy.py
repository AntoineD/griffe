"""This module defines functions to parse Numpy-style docstrings into structured data.

Based on https://numpydoc.readthedocs.io/en/latest/format.html,
it seems Numpydoc is a superset of RST.
Since fully parsing RST is a non-goal of this project,
some things are stripped from the Numpydoc specification.

Rejected as non particularly Pythonic or useful as sections:

- See also: this section feels too subjective (specially crafted as a standard for Numpy itself),
    and there are may ways to reference related items in a docstring, depending on the chosen markup.
- Methods: with a good documentation renderer, methods are easily made accessible or hidden.
    Griffe also has a goal of making the merging of inherited methods configurable (on/off).

Rejected as naturally handled by the user-chosen markup:

- Warnings: this is just markup.
- Notes: again, just markup.
- References: again, just markup.

---

The following sections are supported:

- Deprecated (revisited): we expect a title instead of an RST directive.
    Python has support for deprecating things, so it feels natural
    to structure deprecations.
- Parameters: obviously.
- Returns: obviously.
- Yields: obviously.
- Receives: less used than Yields, but very natural/Pythonic as well.
- Other parameters: used here as documentation for keyword parameters.
- Raises: obviously.
- Warns: less used than Raises, but very natural/Pythonic as well.
- Examples: obviously. Special handling for non-code-blocks `>>>`.
- Attributes: obviously.
"""

from __future__ import annotations

import re
from contextlib import suppress
from textwrap import dedent
from typing import TYPE_CHECKING

from griffe.docstrings.dataclasses import (
    DocstringAttribute,
    DocstringParameter,
    DocstringRaise,
    DocstringReceive,
    DocstringReturn,
    DocstringSection,
    DocstringSectionAttributes,
    DocstringSectionDeprecated,
    DocstringSectionExamples,
    DocstringSectionKind,
    DocstringSectionOtherParameters,
    DocstringSectionParameters,
    DocstringSectionRaises,
    DocstringSectionReceives,
    DocstringSectionReturns,
    DocstringSectionText,
    DocstringSectionWarns,
    DocstringSectionYields,
    DocstringWarn,
    DocstringYield,
)
from griffe.docstrings.utils import parse_annotation, warning
from griffe.expressions import ExprName
from griffe.logger import LogLevel

if TYPE_CHECKING:
    from typing import Any, Literal, Pattern

    from griffe.dataclasses import Docstring
    from griffe.expressions import Expr


_warn = warning(__name__)

_section_kind = {
    "deprecated": DocstringSectionKind.deprecated,
    "parameters": DocstringSectionKind.parameters,
    "other parameters": DocstringSectionKind.other_parameters,
    "returns": DocstringSectionKind.returns,
    "yields": DocstringSectionKind.yields,
    "receives": DocstringSectionKind.receives,
    "raises": DocstringSectionKind.raises,
    "warns": DocstringSectionKind.warns,
    "examples": DocstringSectionKind.examples,
    "attributes": DocstringSectionKind.attributes,
}


def _is_empty_line(line: str) -> bool:
    return not line.strip()


def _is_dash_line(line: str) -> bool:
    return not _is_empty_line(line) and _is_empty_line(line.replace("-", ""))


def _read_block_items(
    docstring: Docstring,
    *,
    offset: int,
    allow_section_blank_line: bool,
    **options: Any,  # noqa: ARG001
) -> tuple[list[list[str]], int]:
    lines = docstring.lines
    if offset >= len(lines):
        return [], offset

    new_offset = offset
    items: list[list[str]] = []

    # skip first empty lines
    while _is_empty_line(lines[new_offset]):
        new_offset += 1

    previous_was_empty = False

    # start processing first item
    current_item = [lines[new_offset]]
    new_offset += 1

    # loop on next lines
    while new_offset < len(lines):
        line = lines[new_offset]

        if line.startswith(4 * " "):
            # continuation line
            current_item.append(line[4:])
            previous_was_empty = False

        elif line.startswith(" "):
            # indent between initial and continuation: append but warn
            cont_indent = len(line) - len(line.lstrip())
            current_item.append(line[cont_indent:])
            _warn(
                docstring,
                new_offset,
                f"Confusing indentation for continuation line {new_offset+1} in docstring, "
                f"should be 4 spaces, not {cont_indent}",
            )
            previous_was_empty = False

        elif _is_empty_line(line):
            # two line breaks indicate the start of a new section
            if previous_was_empty:
                break

            # empty line: preserve it in the current item
            current_item.append("")
            previous_was_empty = True

        else:
            # preserve original behavior, that a single line break between block
            # items triggers a new section
            if not allow_section_blank_line and previous_was_empty:
                break

            # detect the start of a new section
            if new_offset + 1 < len(lines) and lines[new_offset + 1].startswith("---"):
                break

            items.append(current_item)
            current_item = [line]
            previous_was_empty = False

        new_offset += 1

    if current_item:
        items.append(current_item)

    return items, new_offset - 1


def _read_block(docstring: Docstring, *, offset: int) -> tuple[str, int]:
    lines = docstring.lines
    if offset >= len(lines):
        return "", offset

    new_offset = offset
    block: list[str] = []

    # skip first empty lines
    while _is_empty_line(lines[new_offset]):
        new_offset += 1
    while new_offset < len(lines):
        is_empty = _is_empty_line(lines[new_offset])
        if is_empty and new_offset < len(lines) - 1 and _is_dash_line(lines[new_offset + 1]):
            break  # Break if a new unnamed section is reached.

        if is_empty and new_offset < len(lines) - 2 and _is_dash_line(lines[new_offset + 2]):
            break  # Break if a new named section is reached.

        block.append(lines[new_offset])
        new_offset += 1

    return "\n".join(block).rstrip("\n"), new_offset - 1


_RE_OB: str = r"\{"  # opening bracket
_RE_CB: str = r"\}"  # closing bracket
_RE_NAME: str = r"\*{0,2}[_a-z][_a-z0-9]*"
_RE_TYPE: str = r".+"
_RE_RETURNS: Pattern = re.compile(
    rf"""
    (?:
        (?P<nt_name>{_RE_NAME})\s*:\s*(?P<nt_type>{_RE_TYPE})  # name and type
        |  # or
        (?P<name>{_RE_NAME})\s*:\s*  # just name
        |  # or
        \s*:\s*$  # no name, no type
        |  # or
        (?::\s*)?(?P<type>{_RE_TYPE})\s*  # just type
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_RE_YIELDS: Pattern = _RE_RETURNS
_RE_RECEIVES: Pattern = _RE_RETURNS
_RE_PARAMETER: Pattern = re.compile(
    rf"""
    (?P<names>{_RE_NAME}(?:,\s{_RE_NAME})*)
    (?:
        \s:\s
        (?:
            (?:{_RE_OB}(?P<choices>.+){_RE_CB})|
            (?P<type>{_RE_TYPE})
        )?
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_RE_DOCTEST_BLANKLINE: Pattern = re.compile(r"^\s*<BLANKLINE>\s*$")
_RE_DOCTEST_FLAGS: Pattern = re.compile(r"(\s*#\s*doctest:.+)$")


def _read_parameters(
    docstring: Docstring,
    *,
    offset: int,
    warn_unknown_params: bool = True,
    **options: Any,
) -> tuple[list[DocstringParameter], int]:
    parameters = []
    annotation: str | Expr | None

    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    for item in items:
        match = _RE_PARAMETER.match(item[0])
        if not match:
            _warn(docstring, new_offset, f"Could not parse line '{item[0]}'")
            continue

        names = match.group("names").split(", ")
        annotation = match.group("type") or None
        choices = match.group("choices")
        default = None
        if choices:
            choices = choices.split(", ", 1)
            default = choices[0]
        elif annotation:
            match = re.match(r"^(?P<annotation>.+),\s+default(?: |: |=)(?P<default>.+)$", annotation)
            if match:
                default = match.group("default")
                annotation = match.group("annotation")
        if annotation and annotation.endswith(", optional"):
            annotation = annotation[:-10]
        description = "\n".join(item[1:]).rstrip() if len(item) > 1 else ""

        if annotation is None:
            # try to use the annotation from the signature
            for name in names:
                with suppress(AttributeError, KeyError):
                    annotation = docstring.parent.parameters[name].annotation  # type: ignore[union-attr]
                    break
            else:
                _warn(docstring, new_offset, f"No types or annotations for parameters {names}")
        else:
            annotation = parse_annotation(annotation, docstring, log_level=LogLevel.debug)

        if default is None:
            for name in names:
                with suppress(AttributeError, KeyError):
                    default = docstring.parent.parameters[name].default  # type: ignore[union-attr]
                    break

        if warn_unknown_params:
            with suppress(AttributeError):  # for parameters sections in objects without parameters
                params = docstring.parent.parameters  # type: ignore[union-attr]
                for name in names:
                    if name not in params:
                        message = f"Parameter '{name}' does not appear in the function signature"
                        for starred_name in (f"*{name}", f"**{name}"):
                            if starred_name in params:
                                message += f". Did you mean '{starred_name}'?"
                                break
                        _warn(docstring, new_offset, message)

        for name in names:
            parameters.append(DocstringParameter(name, value=default, annotation=annotation, description=description))

    return parameters, new_offset


def _read_parameters_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionParameters | None, int]:
    parameters, new_offset = _read_parameters(docstring, offset=offset, **options)

    if parameters:
        return DocstringSectionParameters(parameters), new_offset

    _warn(docstring, new_offset, f"Empty parameters section at line {offset}")
    return None, new_offset


def _read_other_parameters_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionOtherParameters | None, int]:
    parameters, new_offset = _read_parameters(docstring, offset=offset, warn_unknown_params=False, **options)

    if parameters:
        return DocstringSectionOtherParameters(parameters), new_offset

    _warn(docstring, new_offset, f"Empty other parameters section at line {offset}")
    return None, new_offset


def _read_deprecated_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionDeprecated | None, int]:
    # deprecated
    # SINCE_VERSION
    #     TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty deprecated section at line {offset}")
        return None, new_offset

    if len(items) > 1:
        _warn(docstring, new_offset, f"Too many deprecated items at {offset}")

    item = items[0]
    version = item[0]
    text = dedent("\n".join(item[1:]))
    return DocstringSectionDeprecated(version=version, text=text), new_offset


def _read_returns_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionReturns | None, int]:
    # (NAME : )?TYPE
    #     TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty returns section at line {offset}")
        return None, new_offset

    returns = []
    for index, item in enumerate(items):
        match = _RE_RETURNS.match(item[0])
        if not match:
            _warn(docstring, new_offset, f"Could not parse line '{item[0]}'")
            continue

        groups = match.groupdict()
        name = groups["nt_name"] or groups["name"]
        annotation = groups["nt_type"] or groups["type"]
        text = dedent("\n".join(item[1:]))
        if annotation is None:
            # try to retrieve the annotation from the docstring parent
            with suppress(AttributeError, KeyError, ValueError):
                if docstring.parent.is_function:  # type: ignore[union-attr]
                    annotation = docstring.parent.returns  # type: ignore[union-attr]
                elif docstring.parent.is_attribute:  # type: ignore[union-attr]
                    annotation = docstring.parent.annotation  # type: ignore[union-attr]
                else:
                    raise ValueError
                if len(items) > 1:
                    if annotation.is_tuple:
                        annotation = annotation.slice.elements[index]
                    else:
                        if annotation.is_iterator:
                            return_item = annotation.slice
                        elif annotation.is_generator:
                            return_item = annotation.slice.elements[2]
                        else:
                            raise ValueError
                        if isinstance(return_item, ExprName):
                            annotation = return_item
                        elif return_item.is_tuple:
                            annotation = return_item.slice.elements[index]
                        else:
                            annotation = return_item
        else:
            annotation = parse_annotation(annotation, docstring, log_level=LogLevel.debug)
        returns.append(DocstringReturn(name=name or "", annotation=annotation, description=text))
    return DocstringSectionReturns(returns), new_offset


def _read_yields_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionYields | None, int]:
    # yields
    # (NAME : )?TYPE
    #     TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty yields section at line {offset}")
        return None, new_offset

    yields = []
    for index, item in enumerate(items):
        match = _RE_YIELDS.match(item[0])
        if not match:
            _warn(docstring, new_offset, f"Could not parse line '{item[0]}'")
            continue

        groups = match.groupdict()
        name = groups["nt_name"] or groups["name"]
        annotation = groups["nt_type"] or groups["type"]
        text = dedent("\n".join(item[1:]))
        if annotation is None:
            # try to retrieve the annotation from the docstring parent
            with suppress(AttributeError, KeyError, ValueError):
                annotation = docstring.parent.returns  # type: ignore[union-attr]
                if annotation.is_iterator:
                    yield_item = annotation.slice
                elif annotation.is_generator:
                    yield_item = annotation.slice.elements[0]
                else:
                    raise ValueError
                if isinstance(yield_item, ExprName):
                    annotation = yield_item
                elif yield_item.is_tuple:
                    annotation = yield_item.slice.elements[index]
                else:
                    annotation = yield_item
        else:
            annotation = parse_annotation(annotation, docstring, log_level=LogLevel.debug)
        yields.append(DocstringYield(name=name or "", annotation=annotation, description=text))
    return DocstringSectionYields(yields), new_offset


def _read_receives_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionReceives | None, int]:
    # receives
    # (NAME : )?TYPE
    #     TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty receives section at line {offset}")
        return None, new_offset

    receives = []
    for index, item in enumerate(items):
        match = _RE_RECEIVES.match(item[0])
        if not match:
            _warn(docstring, new_offset, f"Could not parse line '{item[0]}'")
            continue

        groups = match.groupdict()
        name = groups["nt_name"] or groups["name"]
        annotation = groups["nt_type"] or groups["type"]
        text = dedent("\n".join(item[1:]))
        if annotation is None:
            # try to retrieve the annotation from the docstring parent
            with suppress(AttributeError, KeyError):
                annotation = docstring.parent.returns  # type: ignore[union-attr]
                if annotation.is_generator:
                    receives_item = annotation.slice.elements[1]
                    if isinstance(receives_item, ExprName):
                        annotation = receives_item
                    elif receives_item.is_tuple:
                        annotation = receives_item.slice.elements[index]
                    else:
                        annotation = receives_item
        else:
            annotation = parse_annotation(annotation, docstring, log_level=LogLevel.debug)
        receives.append(DocstringReceive(name=name or "", annotation=annotation, description=text))
    return DocstringSectionReceives(receives), new_offset


def _read_raises_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionRaises | None, int]:
    # raises
    # EXCEPTION
    #     TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty raises section at line {offset}")
        return None, new_offset

    raises = []
    for item in items:
        annotation = parse_annotation(item[0], docstring)
        text = dedent("\n".join(item[1:]))
        raises.append(DocstringRaise(annotation=annotation, description=text))
    return DocstringSectionRaises(raises), new_offset


def _read_warns_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionWarns | None, int]:
    # warns
    # WARNING
    #     TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty warns section at line {offset}")
        return None, new_offset

    warns = []
    for item in items:
        annotation = parse_annotation(item[0], docstring)
        text = dedent("\n".join(item[1:]))
        warns.append(DocstringWarn(annotation=annotation, description=text))
    return DocstringSectionWarns(warns), new_offset


def _read_attributes_section(
    docstring: Docstring,
    *,
    offset: int,
    **options: Any,
) -> tuple[DocstringSectionAttributes | None, int]:
    # attributes (for classes)
    # NAME( : TYPE)?
    #    TEXT?
    items, new_offset = _read_block_items(docstring, offset=offset, **options)

    if not items:
        _warn(docstring, new_offset, f"Empty attributes section at line {offset}")
        return None, new_offset

    annotation: str | Expr | None
    attributes = []
    for item in items:
        name_type = item[0]
        if ":" in name_type:
            name, annotation = name_type.split(":", 1)
            name = name.strip()
            annotation = annotation.strip() or None
        else:
            name = name_type
            annotation = None
        if annotation is None:
            with suppress(AttributeError, KeyError):
                annotation = docstring.parent.members[name].annotation  # type: ignore[union-attr]
        else:
            annotation = parse_annotation(annotation, docstring, log_level=LogLevel.debug)
        text = dedent("\n".join(item[1:]))
        attributes.append(DocstringAttribute(name=name, annotation=annotation, description=text))
    return DocstringSectionAttributes(attributes), new_offset


def _read_examples_section(
    docstring: Docstring,
    *,
    offset: int,
    trim_doctest_flags: bool = True,
    **options: Any,  # noqa: ARG001
) -> tuple[DocstringSectionExamples | None, int]:
    text, new_offset = _read_block(docstring, offset=offset)

    sub_sections: list[tuple[Literal[DocstringSectionKind.text, DocstringSectionKind.examples], str]] = []
    in_code_example = False
    in_code_block = False
    current_text: list[str] = []
    current_example: list[str] = []

    for line in text.split("\n"):
        if _is_empty_line(line):
            if in_code_example:
                if current_example:
                    sub_sections.append((DocstringSectionKind.examples, "\n".join(current_example)))
                    current_example = []
                in_code_example = False
            else:
                current_text.append(line)

        elif in_code_example:
            if trim_doctest_flags:
                line = _RE_DOCTEST_FLAGS.sub("", line)  # noqa: PLW2901
                line = _RE_DOCTEST_BLANKLINE.sub("", line)  # noqa: PLW2901
            current_example.append(line)

        elif line.startswith("```"):
            in_code_block = not in_code_block
            current_text.append(line)

        elif in_code_block:
            current_text.append(line)

        elif line.startswith(">>>"):
            if current_text:
                sub_sections.append((DocstringSectionKind.text, "\n".join(current_text).rstrip("\n")))
                current_text = []
            in_code_example = True

            if trim_doctest_flags:
                line = _RE_DOCTEST_FLAGS.sub("", line)  # noqa: PLW2901
            current_example.append(line)

        else:
            current_text.append(line)

    if current_text:
        sub_sections.append((DocstringSectionKind.text, "\n".join(current_text).rstrip("\n")))
    elif current_example:
        sub_sections.append((DocstringSectionKind.examples, "\n".join(current_example)))

    if sub_sections:
        return DocstringSectionExamples(sub_sections), new_offset

    _warn(docstring, new_offset, f"Empty examples section at line {offset}")
    return None, new_offset


_section_reader = {
    DocstringSectionKind.parameters: _read_parameters_section,
    DocstringSectionKind.other_parameters: _read_other_parameters_section,
    DocstringSectionKind.deprecated: _read_deprecated_section,
    DocstringSectionKind.raises: _read_raises_section,
    DocstringSectionKind.warns: _read_warns_section,
    DocstringSectionKind.examples: _read_examples_section,
    DocstringSectionKind.attributes: _read_attributes_section,
    DocstringSectionKind.returns: _read_returns_section,
    DocstringSectionKind.yields: _read_yields_section,
    DocstringSectionKind.receives: _read_receives_section,
}


def parse(
    docstring: Docstring,
    *,
    ignore_init_summary: bool = False,
    trim_doctest_flags: bool = True,
    allow_section_blank_line: bool = False,
    **options: Any,
) -> list[DocstringSection]:
    """Parse a docstring.

    This function iterates on lines of a docstring to build sections.
    It then returns this list of sections.

    Parameters:
        docstring: The docstring to parse.
        ignore_init_summary: Whether to ignore the summary in `__init__` methods' docstrings.
        trim_doctest_flags: Whether to remove doctest flags from Python example blocks.
        allow_section_blank_line: Whether to continue a section if there's an empty line
            between items in a formatted block, like Parameters or Returns.
            If True, you can still create a new section using two empty lines.
        **options: Additional parsing options.

    Returns:
        A list of docstring sections.
    """
    sections: list[DocstringSection] = []
    current_section = []

    in_code_block = False
    lines = docstring.lines

    options = {
        "trim_doctest_flags": trim_doctest_flags,
        "ignore_init_summary": ignore_init_summary,
        "allow_section_blank_line": allow_section_blank_line,
        **options,
    }

    ignore_summary = (
        options["ignore_init_summary"]
        and docstring.parent is not None
        and docstring.parent.name == "__init__"
        and docstring.parent.is_function
        and docstring.parent.parent is not None
        and docstring.parent.parent.is_class
    )

    offset = 2 if ignore_summary else 0

    while offset < len(lines):
        line_lower = lines[offset].lower()

        if in_code_block:
            if line_lower.lstrip(" ").startswith("```"):
                in_code_block = False
            current_section.append(lines[offset])

        elif line_lower in _section_kind and _is_dash_line(lines[offset + 1]):
            if current_section:
                if any(current_section):
                    sections.append(DocstringSectionText("\n".join(current_section).rstrip("\n")))
                current_section = []
            reader = _section_reader[_section_kind[line_lower]]
            section, offset = reader(docstring, offset=offset + 2, **options)  # type: ignore[operator]
            if section:
                sections.append(section)

        elif line_lower.lstrip(" ").startswith("```"):
            in_code_block = True
            current_section.append(lines[offset])

        else:
            current_section.append(lines[offset])

        offset += 1

    if current_section:
        sections.append(DocstringSectionText("\n".join(current_section).rstrip("\n")))

    return sections


__all__ = ["parse"]
