# Extensions

To extract information from your Python source code or compiled module,
Griffe tries to build an Abstract Syntax Tree by parsing the source
with [`ast`][] utilities, and if the source code is not available
(built-in or compiled module), Griffe imports the module and
builds an object tree instead.

Griffe then follows the [Visitor pattern](https://www.wikiwand.com/en/Visitor_pattern)
to walk the tree and extract information.
For ASTs, Griffe uses its [Visitor][griffe.agents.visitor] agent
and for object trees, it uses its [Inspector][griffe.agents.inspector] agent.

At each step of the walk through the tree, in depth-first order,
both the visitor and inspector agents are able to run sub-visitors
or sub-inspectors: visitors or inspectors that walk on a smaller,
or more specific part of the tree. These sub-visitors and sub-inspectors
are what we call **extensions**.

The following flow chart shows an example of an AST visit.
The tree is simplified: actual trees have a lot more nodes
like `if/elif/else` nodes, `try/except/else/finally` nodes,
[and many others][ast.AST].

```mermaid
flowchart TB

M(Module definition) --- C(Class definition) & F(Function definition)
C --- m(Function definition) & A(Variable assignment)
```

The following flow chart shows an example of an object tree inspection.
The tree is simplified as well: [many more types of object are handled][griffe.agents.nodes.ObjectNode]. 

```mermaid 
flowchart TB

M(Module) --- C(Class) & F(Function)
C --- m(Method) & A(Attribute)
```

Now you can see in the following flow chart how extensions
run as part of the visit/inspection:


```mermaid
flowchart TB

a(Node a) --- b(Node b) & c(Node c)
c --- d(Node d) & e(Node e)
agent[Agent] -- 1. visits/inspects --> c
agent[Agent] -- 2. runs extension --> ext[Extension]
ext -- 3. visits/inspects --> c
agent -- 4. moves to next node, etc. --> e

style agent fill:#33f,color:#ccc
style ext fill:#f33,color:#333
```

Extensions can be configured to run either:

- before the agent starts handling the current node
- after the agent has started handling the node,
    but before it has handled its children
- after the agent has started handling the node
    and finished handling its children,
    but before it has finished handling the node.
- after the agent has finished handling the node and its children.

For example, when encountering a class definition node, the agent will:

1. run every extension configured to run before it starts handling the class
1. start handling the class, instantiating a [`Class`][griffe.dataclasses.Class] object,
    and updating its state to reference this class as "current" object
1. run every extension configured to run before it starts handling the children
1. handle the children (behavior is repeated for each child)
1. run every extension configured to run after it has handled the children
1. finish handling the node, updating back to its previous state,
    referencing the class' parent as "current" object again
1. run every extension configured to run after it has handled the node

So the difference between *"after children"* and *"after all"* extensions
is that the agent (which is made available to the extensions) is in a different state,
holding a reference to either the object corresponding to the current node,
or the object corresponding to the current node's parent.

```mermaid
flowchart TB

a(Node a) --- b(Node b) & c(Node c)
c --- d(Node d) & e(Node e)
agent[Agent] --> ext_before_all[Exts]:::exts
agent[Agent] -- starts handling --> c
agent[Agent] --> ext_before_children[Exts]:::exts
agent[Agent] --> ext_after_children[Exts]:::exts
agent[Agent] -- finishes handling --> c
agent[Agent] --> ext_after_all[Exts]:::exts
ext_before_all & ext_before_children & ext_after_children & ext_after_all --> c
agent --> e

style agent fill:#33f,color:#ccc
classDef exts fill:#f33,color:#333
```

```python exec="1"
--8<-- "scripts/draw_svg.py"
```

You can pass extensions to the loader to augment its capacities:

```python
from griffe.loader import GriffeLoader
from griffe.extensions import VisitorExtension, Extensions, When

# import extensions
from some.package import TheirExtension


# or define your own
class ClassStartsAtOddLineNumberExtension(VisitorExtension):
    when = When.after_all

    def visit_classdef(self, node) -> None:
        if node.lineno % 2 == 1:
            self.visitor.current.labels.add("starts at odd line number")


extensions = Extensions(TheirExtension, ClassStartsAtOddLineNumberExtension)
griffe = GriffeLoader(extensions=extensions)
fastapi = griffe.load_module("fastapi")
```

Extensions are subclasses of a custom version of [`ast.NodeVisitor`][ast.NodeVisitor].
Griffe uses a node visitor as well, that we will call the *main visitor*.
The extensions are instantiated with a reference to this main visitor,
so they can benefit from its capacities (navigating the nodes, getting the current
class or module, etc.).

Each time a node is visited, the main visitor will make the extensions visit the node as well.
Implement the `visit_<NODE_TYPE_LOWER>` methods to visit nodes of certain types,
and act on their properties. See the [full list of AST nodes](#ast-nodes).

!!! warning "Important note"
    Because the main visitor recursively walks the tree itself,
    calling extensions on each node,
    **you must not visit child nodes in your `.visit_*` methods!**
    Otherwise, nodes down the tree will be visited twice or more:
    once by the main visitor, and as many times more as your extension is called.
    Let the main visitor do the walking, and just take care of the current node,
    without handling its children.

You can access the main visitor state and data through the `.visitor` attribute,
and the nodes instances are extended with additional attributes and properties:

```python
class MyExtension(Extension):
    def visit_functiondef(self, node) -> None:
        node.parent  # the parent node
        node.children  # the list of children nodes
        node.siblings  # all the siblings nodes, from top to bottom
        node.previous_siblings  # only the previous siblings, from closest to top
        node.next_siblings  # only the next siblings, from closest to bottom
        node.previous  # first previous sibling
        node.next  # first next sibling

        self.visitor  # the main visitor
        self.visitor.current  # the current data object
        self.visitor.current.kind  # the kind of object: module, class, function, attribute
```

See the data classes ([`Module`][griffe.dataclasses.Module],
[`Class`][griffe.dataclasses.Class], [`Function`][griffe.dataclasses.Function]
and [`Attribute`][griffe.dataclasses.Attribute])
for a complete description of their methods and attributes.

Extensions are run at certain moments while walking the Abstract Syntax Tree (AST):

- before the visit/inspection: `When.before_all`.
  The current node has been grafted to its parent.
  If this node represents a data object, the object (`self.visitor.current`/`self.inspector.current`) **is not** yet instantiated.
- before the children visit/inspection: `When.before_children`.
  If this node represents a data object, the object (`self.visitor.current`/`self.inspector.current`) **is** now instantiated.
  Children **have not** yet been visited/inspected.
- after the children visit/inspection: `When.after_children`.
  Children **have** now been visited/inspected.
- after the visit/inspection: `When.after_all`

See [the `When` enumeration][griffe.extensions.When].

To tell the main visitor to run your extension at a certain time,
set its `when` attribute:

```python
class MyExtension(Extension):
    when = When.after_children
```

By default, it will run the extension after the visit/inspection of the node:
that's when the full data for this node and its children is loaded.

## AST nodes

> <table style="border: none; background-color: unset;"><tbody><tr><td>
>
> - [`Add`][ast.Add]
> - [`alias`][ast.alias]
> - [`And`][ast.And]
> - [`AnnAssign`][ast.AnnAssign]
> - [`arg`][ast.arg]
> - [`arguments`][ast.arguments]
> - [`Assert`][ast.Assert]
> - [`Assign`][ast.Assign]
> - [`AsyncFor`][ast.AsyncFor]
> - [`AsyncFunctionDef`][ast.AsyncFunctionDef]
> - [`AsyncWith`][ast.AsyncWith]
> - [`Attribute`][ast.Attribute]
> - [`AugAssign`][ast.AugAssign]
> - [`Await`][ast.Await]
> - [`BinOp`][ast.BinOp]
> - [`BitAnd`][ast.BitAnd]
> - [`BitOr`][ast.BitOr]
> - [`BitXor`][ast.BitXor]
> - [`BoolOp`][ast.BoolOp]
> - [`Break`][ast.Break]
> - `Bytes`[^1]
> - [`Call`][ast.Call]
> - [`ClassDef`][ast.ClassDef]
> - [`Compare`][ast.Compare]
> - [`comprehension`][ast.comprehension]
> - [`Constant`][ast.Constant]
> - [`Continue`][ast.Continue]
> - [`Del`][ast.Del]
> - [`Delete`][ast.Delete]
>
> </td><td>
>
> - [`Dict`][ast.Dict]
> - [`DictComp`][ast.DictComp]
> - [`Div`][ast.Div]
> - `Ellipsis`[^1]
> - [`Eq`][ast.Eq]
> - [`ExceptHandler`][ast.ExceptHandler]
> - [`Expr`][ast.Expr]
> - `Expression`[^1]
> - `ExtSlice`[^2]
> - [`FloorDiv`][ast.FloorDiv]
> - [`For`][ast.For]
> - [`FormattedValue`][ast.FormattedValue]
> - [`FunctionDef`][ast.FunctionDef]
> - [`GeneratorExp`][ast.GeneratorExp]
> - [`Global`][ast.Global]
> - [`Gt`][ast.Gt]
> - [`GtE`][ast.GtE]
> - [`If`][ast.If]
> - [`IfExp`][ast.IfExp]
> - [`Import`][ast.Import]
> - [`ImportFrom`][ast.ImportFrom]
> - [`In`][ast.In]
> - `Index`[^2]
> - `Interactive`[^3]
> - [`Invert`][ast.Invert]
> - [`Is`][ast.Is]
> - [`IsNot`][ast.IsNot]
> - [`JoinedStr`][ast.JoinedStr]
> - [`keyword`][ast.keyword]
>
> </td><td>
>
> - [`Lambda`][ast.Lambda]
> - [`List`][ast.List]
> - [`ListComp`][ast.ListComp]
> - [`Load`][ast.Load]
> - [`LShift`][ast.LShift]
> - [`Lt`][ast.Lt]
> - [`LtE`][ast.LtE]
> - [`Match`][ast.Match]
> - [`MatchAs`][ast.MatchAs]
> - [`match_case`][ast.match_case]
> - [`MatchClass`][ast.MatchClass]
> - [`MatchMapping`][ast.MatchMapping]
> - [`MatchOr`][ast.MatchOr]
> - [`MatchSequence`][ast.MatchSequence]
> - [`MatchSingleton`][ast.MatchSingleton]
> - [`MatchStar`][ast.MatchStar]
> - [`MatchValue`][ast.MatchValue]
> - [`MatMult`][ast.MatMult]
> - [`Mod`][ast.Mod]
> - `Module`[^3]
> - [`Mult`][ast.Mult]
> - [`Name`][ast.Name]
> - `NameConstant`[^1]
> - [`NamedExpr`][ast.NamedExpr]
> - [`Nonlocal`][ast.Nonlocal]
> - [`Not`][ast.Not]
> - [`NotEq`][ast.NotEq]
> - [`NotIn`][ast.NotIn]
> - `Num`[^1]
>
> </td><td>
>
> - [`Or`][ast.Or]
> - [`Pass`][ast.Pass]
> - `pattern`[^3]
> - [`Pow`][ast.Pow]
> - `Print`[^4]
> - [`Raise`][ast.Raise]
> - [`Return`][ast.Return]
> - [`RShift`][ast.RShift]
> - [`Set`][ast.Set]
> - [`SetComp`][ast.SetComp]
> - [`Slice`][ast.Slice]
> - [`Starred`][ast.Starred]
> - [`Store`][ast.Store]
> - `Str`[^1]
> - [`Sub`][ast.Sub]
> - [`Subscript`][ast.Subscript]
> - [`Try`][ast.Try]
> - `TryExcept`[^5]
> - `TryFinally`[^6]
> - [`Tuple`][ast.Tuple]
> - [`UAdd`][ast.UAdd]
> - [`UnaryOp`][ast.UnaryOp]
> - [`USub`][ast.USub]
> - [`While`][ast.While]
> - [`With`][ast.With]
> - [`withitem`][ast.withitem]
> - [`Yield`][ast.Yield]
> - [`YieldFrom`][ast.YieldFrom]
> 
> </td></tr></tbody></table>

[^1]: Deprecated since Python 3.8.
[^2]: Deprecated since Python 3.9.
[^3]: Not documented.
[^4]: `print` became a builtin (instead of a keyword) in Python 3.
[^5]: Now `ExceptHandler`, in the `handlers` attribute of `Try` nodes.
[^6]: Now a list of expressions in the `finalbody` attribute of `Try` nodes.
