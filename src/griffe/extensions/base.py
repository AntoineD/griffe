"""This module contains the base classes for dealing with extensions."""

from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from inspect import isclass
from typing import TYPE_CHECKING, Any, Sequence, Union

from griffe.agents.nodes import ast_children, ast_kind
from griffe.enumerations import When
from griffe.exceptions import ExtensionNotLoadedError
from griffe.importer import dynamic_import

if TYPE_CHECKING:
    import ast
    from types import ModuleType

    from griffe.agents.inspector import Inspector
    from griffe.agents.nodes import ObjectNode
    from griffe.agents.visitor import Visitor
    from griffe.dataclasses import Attribute, Class, Function, Module, Object


class VisitorExtension:
    """Deprecated in favor of `Extension`. The node visitor extension base class, to inherit from."""

    when: When = When.after_all

    def __init__(self) -> None:
        """Initialize the visitor extension."""
        warnings.warn(
            "Visitor extensions are deprecated in favor of the new, more developer-friendly Extension. "
            "See https://mkdocstrings.github.io/griffe/extensions/",
            DeprecationWarning,
            stacklevel=1,
        )
        self.visitor: Visitor = None  # type: ignore[assignment]

    def attach(self, visitor: Visitor) -> None:
        """Attach the parent visitor to this extension.

        Parameters:
            visitor: The parent visitor.
        """
        self.visitor = visitor

    def visit(self, node: ast.AST) -> None:
        """Visit a node.

        Parameters:
            node: The node to visit.
        """
        getattr(self, f"visit_{ast_kind(node)}", lambda _: None)(node)


class InspectorExtension:
    """Deprecated in favor of `Extension`. The object inspector extension base class, to inherit from."""

    when: When = When.after_all

    def __init__(self) -> None:
        """Initialize the inspector extension."""
        warnings.warn(
            "Inspector extensions are deprecated in favor of the new, more developer-friendly Extension. "
            "See https://mkdocstrings.github.io/griffe/extensions/",
            DeprecationWarning,
            stacklevel=1,
        )
        self.inspector: Inspector = None  # type: ignore[assignment]

    def attach(self, inspector: Inspector) -> None:
        """Attach the parent inspector to this extension.

        Parameters:
            inspector: The parent inspector.
        """
        self.inspector = inspector

    def inspect(self, node: ObjectNode) -> None:
        """Inspect a node.

        Parameters:
            node: The node to inspect.
        """
        getattr(self, f"inspect_{node.kind}", lambda _: None)(node)


class Extension:
    """Base class for Griffe extensions."""

    def visit(self, node: ast.AST) -> None:
        """Visit a node.

        Parameters:
            node: The node to visit.
        """
        getattr(self, f"visit_{ast_kind(node)}", lambda _: None)(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Visit children nodes.

        Parameters:
            node: The node to visit the children of.
        """
        for child in ast_children(node):
            self.visit(child)

    def inspect(self, node: ObjectNode) -> None:
        """Inspect a node.

        Parameters:
            node: The node to inspect.
        """
        getattr(self, f"inspect_{node.kind}", lambda _: None)(node)

    def generic_inspect(self, node: ObjectNode) -> None:
        """Extend the base generic inspection with extensions.

        Parameters:
            node: The node to inspect.
        """
        for child in node.children:
            if not child.alias_target_path:
                self.inspect(child)

    def on_node(self, *, node: ast.AST | ObjectNode) -> None:
        """Run when visiting a new node during static/dynamic analysis.

        Parameters:
            node: The currently visited node.
        """

    def on_instance(self, *, node: ast.AST | ObjectNode, obj: Object) -> None:
        """Run when an Object has been created.

        Parameters:
            node: The currently visited node.
            obj: The object instance.
        """

    def on_members(self, *, node: ast.AST | ObjectNode, obj: Object) -> None:
        """Run when members of an Object have been loaded.

        Parameters:
            node: The currently visited node.
            obj: The object instance.
        """

    def on_module_node(self, *, node: ast.AST | ObjectNode) -> None:
        """Run when visiting a new module node during static/dynamic analysis.

        Parameters:
            node: The currently visited node.
        """

    def on_module_instance(self, *, node: ast.AST | ObjectNode, mod: Module) -> None:
        """Run when a Module has been created.

        Parameters:
            node: The currently visited node.
            mod: The module instance.
        """

    def on_module_members(self, *, node: ast.AST | ObjectNode, mod: Module) -> None:
        """Run when members of a Module have been loaded.

        Parameters:
            node: The currently visited node.
            mod: The module instance.
        """

    def on_class_node(self, *, node: ast.AST | ObjectNode) -> None:
        """Run when visiting a new class node during static/dynamic analysis.

        Parameters:
            node: The currently visited node.
        """

    def on_class_instance(self, *, node: ast.AST | ObjectNode, cls: Class) -> None:
        """Run when a Class has been created.

        Parameters:
            node: The currently visited node.
            cls: The class instance.
        """

    def on_class_members(self, *, node: ast.AST | ObjectNode, cls: Class) -> None:
        """Run when members of a Class have been loaded.

        Parameters:
            node: The currently visited node.
            cls: The class instance.
        """

    def on_function_node(self, *, node: ast.AST | ObjectNode) -> None:
        """Run when visiting a new function node during static/dynamic analysis.

        Parameters:
            node: The currently visited node.
        """

    def on_function_instance(self, *, node: ast.AST | ObjectNode, func: Function) -> None:
        """Run when a Function has been created.

        Parameters:
            node: The currently visited node.
            func: The function instance.
        """

    def on_attribute_node(self, *, node: ast.AST | ObjectNode) -> None:
        """Run when visiting a new attribute node during static/dynamic analysis.

        Parameters:
            node: The currently visited node.
        """

    def on_attribute_instance(self, *, node: ast.AST | ObjectNode, attr: Attribute) -> None:
        """Run when an Attribute has been created.

        Parameters:
            node: The currently visited node.
            attr: The attribute instance.
        """


ExtensionType = Union[VisitorExtension, InspectorExtension, Extension]


class Extensions:
    """This class helps iterating on extensions that should run at different times."""

    def __init__(self, *extensions: ExtensionType) -> None:
        """Initialize the extensions container.

        Parameters:
            *extensions: The extensions to add.
        """
        self._visitors: dict[When, list[VisitorExtension]] = defaultdict(list)
        self._inspectors: dict[When, list[InspectorExtension]] = defaultdict(list)
        self._extensions: list[Extension] = []
        self.add(*extensions)

    def add(self, *extensions: ExtensionType) -> None:
        """Add extensions to this container.

        Parameters:
            *extensions: The extensions to add.
        """
        for extension in extensions:
            if isinstance(extension, VisitorExtension):
                self._visitors[extension.when].append(extension)
            elif isinstance(extension, InspectorExtension):
                self._inspectors[extension.when].append(extension)
            else:
                self._extensions.append(extension)

    def attach_visitor(self, parent_visitor: Visitor) -> Extensions:
        """Attach a parent visitor to the visitor extensions.

        Parameters:
            parent_visitor: The parent visitor, leading the visit.

        Returns:
            Self, conveniently.
        """
        for when in self._visitors:
            for visitor in self._visitors[when]:
                visitor.attach(parent_visitor)
        return self

    def attach_inspector(self, parent_inspector: Inspector) -> Extensions:
        """Attach a parent inspector to the inspector extensions.

        Parameters:
            parent_inspector: The parent inspector, leading the inspection.

        Returns:
            Self, conveniently.
        """
        for when in self._inspectors:
            for inspector in self._inspectors[when]:
                inspector.attach(parent_inspector)
        return self

    @property
    def before_visit(self) -> list[VisitorExtension]:
        """Return the visitors that run before the visit.

        Returns:
            Visitors.
        """
        return self._visitors[When.before_all]

    @property
    def before_children_visit(self) -> list[VisitorExtension]:
        """Return the visitors that run before the children visit.

        Returns:
            Visitors.
        """
        return self._visitors[When.before_children]

    @property
    def after_children_visit(self) -> list[VisitorExtension]:
        """Return the visitors that run after the children visit.

        Returns:
            Visitors.
        """
        return self._visitors[When.after_children]

    @property
    def after_visit(self) -> list[VisitorExtension]:
        """Return the visitors that run after the visit.

        Returns:
            Visitors.
        """
        return self._visitors[When.after_all]

    @property
    def before_inspection(self) -> list[InspectorExtension]:
        """Return the inspectors that run before the inspection.

        Returns:
            Inspectors.
        """
        return self._inspectors[When.before_all]

    @property
    def before_children_inspection(self) -> list[InspectorExtension]:
        """Return the inspectors that run before the children inspection.

        Returns:
            Inspectors.
        """
        return self._inspectors[When.before_children]

    @property
    def after_children_inspection(self) -> list[InspectorExtension]:
        """Return the inspectors that run after the children inspection.

        Returns:
            Inspectors.
        """
        return self._inspectors[When.after_children]

    @property
    def after_inspection(self) -> list[InspectorExtension]:
        """Return the inspectors that run after the inspection.

        Returns:
            Inspectors.
        """
        return self._inspectors[When.after_all]

    def call(self, event: str, *, node: ast.AST | ObjectNode, **kwargs: Any) -> None:
        """Call the extension hook for the given event.

        Parameters:
            event: The trigerred event.
            node: The AST or Object node.
            **kwargs: Additional arguments like a Griffe object.
        """
        for extension in self._extensions:
            getattr(extension, event)(node=node, **kwargs)


builtin_extensions: set[str] = {
    "hybrid",
}


def _load_extension_path(path: str) -> ModuleType:
    module_name = os.path.basename(path).rsplit(".", 1)[0]
    spec = spec_from_file_location(module_name, path)
    if not spec:
        raise ExtensionNotLoadedError(f"Could not import module from path '{path}'")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_extension(
    extension: str | dict[str, Any] | ExtensionType | type[ExtensionType],
) -> ExtensionType | list[ExtensionType]:
    """Load a configured extension.

    Parameters:
        extension: An extension, with potential configuration options.

    Raises:
        ExtensionNotLoadedError: When the extension cannot be loaded,
            either because the module is not found, or because it does not expose
            the Extension attribute. ImportError will bubble up so users can see
            the traceback.

    Returns:
        An extension instance.
    """
    ext_object = None
    ext_classes = (VisitorExtension, InspectorExtension, Extension)

    # If it's already an extension instance, return it.
    if isinstance(extension, ext_classes):
        return extension

    # If it's an extension class, instantiate it (without options) and return it.
    if isclass(extension) and issubclass(extension, ext_classes):
        return extension()

    # If it's a dictionary, we expect the only key to be an import path
    # and the value to be a dictionary of options.
    if isinstance(extension, dict):
        import_path, options = next(iter(extension.items()))

    # Otherwise we consider it's an import path, without options.
    else:
        import_path = str(extension)
        options = {}

    # If the import path contains a colon, we split into path and class name.
    if ":" in import_path:
        import_path, extension_name = import_path.split(":", 1)
    else:
        extension_name = None

    # If the import path corresponds to a built-in extension, expand it.
    if import_path in builtin_extensions:
        import_path = f"griffe.extensions.{import_path}"
    # If the import path is a path to an existing file, load it.
    elif os.path.exists(import_path):
        try:
            ext_object = _load_extension_path(import_path)
        except ImportError as error:
            raise ExtensionNotLoadedError(f"Extension module '{import_path}' could not be found") from error

    # If the extension wasn't loaded yet, we consider the import path
    # to be a Python dotted path like `package.module` or `package.module.Extension`.
    if not ext_object:
        try:
            ext_object = dynamic_import(import_path)
        except ModuleNotFoundError as error:
            raise ExtensionNotLoadedError(f"Extension module '{import_path}' could not be found") from error
        except ImportError as error:
            raise ExtensionNotLoadedError(f"Error while importing extension '{import_path}': {error}") from error

    # If the loaded object is an extension class, instantiate it with options and return it.
    if isclass(ext_object) and issubclass(ext_object, ext_classes):
        return ext_object(**options)  # type: ignore[misc]

    # Otherwise the loaded object is a module, so we get the extension class by name,
    # instantiate it with options and return it.
    if extension_name:
        try:
            return getattr(ext_object, extension_name)(**options)
        except AttributeError as error:
            raise ExtensionNotLoadedError(
                f"Extension module '{import_path}' has no '{extension_name}' attribute",
            ) from error

    # No class name was specified so we search all extension classes in the module
    # instantiate each with the same options, and return them.
    extensions = []
    for obj in vars(ext_object).values():
        if isclass(obj) and issubclass(obj, ext_classes) and obj not in ext_classes:
            extensions.append(obj)
    return [ext(**options) for ext in extensions]


def load_extensions(exts: Sequence[str | dict[str, Any] | ExtensionType | type[ExtensionType]]) -> Extensions:
    """Load configured extensions.

    Parameters:
        exts: A sequence of extension, with potential configuration options.

    Returns:
        An extensions container.
    """
    extensions = Extensions()
    for extension in exts:
        ext = _load_extension(extension)
        if isinstance(ext, list):
            extensions.add(*ext)
        else:
            extensions.add(ext)
    return extensions


__all__ = [
    "builtin_extensions",
    "Extension",
    "Extensions",
    "ExtensionType",
    "InspectorExtension",
    "load_extensions",
    "VisitorExtension",
    "When",
]
