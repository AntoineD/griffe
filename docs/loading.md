# Loading data with Python

Griffe provides a shortcut function for simple needs:

```python
import griffe

mkdocs = griffe.load("mkdocs")
```

The [`load`][griffe.loader.load] function accepts a number of parameters.

For more complex needs, create and use a loader:

```python
from griffe.loader import GriffeLoader

loader = GriffeLoader()
mkdocs = loader.load_module("mkdocs")
```

Similarly, the [`GriffeLoader`][griffe.loader.GriffeLoader] accepts
a number of parameters to configure how the modules are found and loaded.

If you don't want to recurse in the submodules:

```python
mkdocs = loader.load_module("mkdocs", submodules=False)
```

## Navigating into the loaded objects

Both the `load` function and the `GriffeLoader.load_module` method
return a [`Module`][griffe.dataclasses.Module] instance.
There are several ways to access members of an object:

- through its `members` attribute, which is a dictionary,
  with the usual `keys()`, `values()` and `items()` methods.
- thanks to its `__getitem__` method. For example `griffe["dataclasses"]`
  returns the `Module` instance representing Griffe's `dataclasses` module.
  Since this module also has members, you can chain calls: `griffe["dataclasses"]["Module"]`.
  Conveniently, you can chain the names with dots in a single call: `griffe["dataclasses.Module"]`.
  You can even pass a tuple instead of a string: `griffe[("dataclasses", "Module")]`.
- through the [`modules`][griffe.dataclasses.Object.modules],
  [`classes`][griffe.dataclasses.Object.classes],
  [`functions`][griffe.dataclasses.Object.functions] and
  [`attributes`][griffe.dataclasses.Object.attributes] properties,
  which take care of filtering members based on their kind, and return dictionaries.

Most of the time, you will only use classes from the [`griffe.dataclasses`][griffe.dataclasses]
and [`griffe.docstrings.dataclasses`][griffe.docstrings.dataclasses] modules.


## Class inheritance

TIP: **New in version 0.30**

WARNING: **Inheritance support is experimental**
Inheritance support was recently added, and might need
some corrections before being fully usable.
Don't hesitate to report any issue that arises
from using inheritance support in Griffe.

Griffe supports class inheritance, both when visiting and inspecting modules.

To access members of a class that are inherited from base classes,
use [`Object.inherited_members`][griffe.dataclasses.Object.inherited_members].
If this is the first time you access inherited members, the base classes
of the given class will be resolved and cached, then the MRO (Method Resolution Order)
will be computed for these bases classes, and a dictionary of inherited members
will be built and cached. Next times you access it, you'll get the cached dictionary.
Make sure to only access `inherited_members` once everything is loaded by Griffe,
to avoid computing things too early. Don't access inherited members
in extensions, while visiting or inspecting a module.

**Important:** only classes from already loaded packages
will be used when computing inherited members.
This gives users control over how deep into inheritance to go,
by pre-loading packages from which you want to inherit members.
For example, if `package_c.ClassC` inherits from `package_b.ClassB`,
itself inheriting from `package_a.ClassA`, and you want
to load `ClassB` members only:

```python
from griffe.loader import GriffeLoader

loader = GriffeLoader()
# note that we don't load package_a
loader.load_module("package_b")
loader.load_module("package_c")
```

If a base class cannot be resolved during computation
of inherited members, Griffe logs a DEBUG message.

If you want to access all members at once (both declared and inherited),
use [`Object.all_members`][griffe.dataclasses.Object.all_members].

If you want to access only declared members,
use [`Object.members`][griffe.dataclasses.Object].

Accessing [`Object.attributes`][griffe.dataclasses.Object.attributes],
[`Object.functions`][griffe.dataclasses.Object.functions],
[`Object.classes`][griffe.dataclasses.Object.classes] or
[`Object.modules`][griffe.dataclasses.Object.modules]
will trigger inheritance computation, so make sure to only call it
once everything is loaded by Griffe. Don't access inherited members
in extensions, while visiting or inspecting a module.

### Limitations

Currently, there are two limitations to our class inheritance support:

1. when visiting (static analysis), some objects are not yet properly recognized as classes,
    for example named tuples. If you inherit from a named tuple,
    its members won't be added to the inherited members of the inheriting class.

    ```python
    MyTuple = namedtuple("MyTuple", "attr1 attr2")


    class MyClass(MyTuple):
        ...
    ```

2. when inspecting (dynamic analysis), ephemeral base classes won't be resolved,
    and therefore their members won't appear in child classes. To circumvent that,
    assign these dynamic classes to variables:

    ```python
    # instead of
    class MyClass(namedtuple("MyTuple", "attr1 attr2")):
        ...


    # do
    MyTuple = namedtuple("MyTuple", "attr1 attr2")


    class MyClass(MyTuple):
        ...
    ```

We will try to lift these limitations in the future.
