"""Executing code a language model wrote.

Be honest about what this is. The sandbox raises the cost of an accident --
it stops a confused or unlucky synthesis from opening files, importing the
operating system, or walking ``__subclasses__`` to escape -- and it does that
BEFORE execution, at compile time, by walking the AST, so a forbidden
construct never runs even one instruction. It does not defend against a
determined adversary. Language-model-authored bytecode tricks, C-extension
escapes, and resource exhaustion beyond the timeout are all out of scope.
Anyone running fully untrusted synthesis output should put the whole
synthesis loop in a container; this module is the seatbelt, not the wall.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import threading
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from inspect import signature

from blotto.protocols import CodeWorldModel

__all__ = [
    "SandboxConfig",
    "Sandbox",
    "SandboxViolation",
    "SandboxTimeout",
    "ProtocolViolation",
    "call_with_timeout",
    "instantiate",
    "check_protocol_methods",
]


DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "math",
        "random",
        "itertools",
        "collections",
        "dataclasses",
        "typing",
        "copy",
        "json",
        "datetime",
        "statistics",
        "functools",
        "operator",
        "re",
    }
)
"""Enough to write a rich simulator -- randomness, data structures, numerics
-- and nothing that touches the world outside the process. ``random`` is in
and ``time``/``os``/``socket`` are out: a model that wants randomness uses the
chance player, and a model that wants the clock wants something it should not
have."""


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Knobs for one sandboxed execution."""

    timeout_seconds: float = 10.0
    max_output_bytes: int = 8192
    allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES


class SandboxViolation(Exception):
    """The source asked for something the sandbox refuses, caught at compile
    time before any of it executed."""


class SandboxTimeout(Exception):
    """The call overran its budget; see ``call_with_timeout`` for what that
    does and does not buy."""


# Names whose mere appearance is refused, called or not: aliasing ``open`` to
# a variable and calling it later is the same request with extra steps.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
)


class _ForbiddenVisitor(ast.NodeVisitor):
    """Collects every violation, so one refusal names them all rather than
    making the author fix them one round-trip at a time."""

    def __init__(self, allowed: frozenset[str]) -> None:
        self.allowed = allowed
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, what: str) -> None:
        self.violations.append(
            f"{what} ({type(node).__name__}) at line {node.lineno}"
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in self.allowed:
                self._flag(node, f"Import of disallowed module {alias.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # ``from __future__ import annotations`` is resolved by the compiler,
        # not the import machinery, and every synthesised module we prompt for
        # starts with it; refusing it would refuse the style guide.
        if node.module != "__future__" and (
            node.module is None or node.module.split(".")[0] not in self.allowed
        ):
            self._flag(
                node,
                f"ImportFrom disallowed module {(node.module or '<relative>')!r}",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._flag(node, f"reference to forbidden name {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = node.attr
        if name.startswith("__") and name.endswith("__"):
            self._flag(node, f"dunder attribute access {name!r}")
        self.generic_visit(node)


def _check_source(source: str, allowed: frozenset[str]) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxViolation(f"syntax error: {exc}") from exc
    visitor = _ForbiddenVisitor(allowed)
    visitor.visit(tree)
    if visitor.violations:
        raise SandboxViolation("; ".join(visitor.violations))


class _CappedBuffer(io.StringIO):
    """Output sink that refuses to buffer past the cap, so a model that
    prints its way through the memory budget fails loudly instead of
    quietly."""

    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap

    def write(self, text: str) -> int:
        if self.tell() + len(text) > self._cap:
            raise SandboxViolation(
                f"sandboxed output exceeded {self._cap} bytes -- write less, "
                "or raise SandboxConfig.max_output_bytes"
            )
        return super().write(text)


def _safe_builtins(import_hook: Callable[..., types.ModuleType]) -> dict[str, object]:
    """The builtins a world model legitimately needs and nothing else.

    ``__build_class__`` and ``__name__`` have to be present or ``class``
    statements fail; they are injected by the sandbox, never authored by the
    model, which is the distinction the dunder rule in the AST walker is
    drawing.
    """
    import builtins as _builtins

    names = (
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict", "dir",
        "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash",
        "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
        "Exception", "ArithmeticError", "AttributeError", "IndexError",
        "KeyError", "LookupError", "RuntimeError", "StopIteration", "TypeError",
        "ValueError", "ZeroDivisionError", "NotImplementedError",
    )
    namespace: dict[str, object] = {name: getattr(_builtins, name) for name in names}
    namespace["__import__"] = import_hook
    namespace["__build_class__"] = _builtins.__build_class__
    namespace["__name__"] = "cwm_sandbox_module"
    return namespace


class Sandbox:
    """Compile and execute untrusted source in a restricted namespace."""

    _load_counter = 0

    def load(self, source: str, config: SandboxConfig) -> types.ModuleType:
        """Return the module namespace produced by executing ``source``.

        The AST is walked BEFORE compilation of the bytecode that will run:
        refusing at compile time is the whole point, because a check at
        execution time has already let the offending construct run partway.
        """
        _check_source(source, config.allowed_modules)

        def import_hook(
            name: str, *args: object, **kwargs: object
        ) -> types.ModuleType:
            root = name.split(".")[0]
            # ``__future__`` is allowed at runtime for the same reason it is
            # allowed in the AST walk: it is resolved by the compiler and
            # every well-formed synthesised module starts with it.
            if root not in config.allowed_modules and root != "__future__":
                raise SandboxViolation(
                    f"import of disallowed module {name!r} at runtime"
                )
            return import_module(name)

        # A unique name, and registration in sys.modules, because the
        # dataclasses machinery resolves string annotations by looking the
        # defining module up there -- an unregistered module breaks
        # ``@dataclass`` in the synthesised code, and every candidate model
        # will want ``@dataclass``.
        Sandbox._load_counter += 1
        module_name = f"cwm_sandbox_{Sandbox._load_counter}"
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        module.__dict__["__builtins__"] = _safe_builtins(import_hook)
        code = compile(source, "<cwm_sandbox>", "exec")
        buffer = _CappedBuffer(config.max_output_bytes)
        # stdout and stderr both flow through the cap. stdin needs no
        # redirection: ``input`` is not in the safe builtins, so a world
        # model has no way to ask the operator questions in the first place.
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, module.__dict__)  # noqa: S102 - the sandbox IS the point
        return module


def call_with_timeout(
    fn: Callable[..., object],
    args: Iterable[object] = (),
    timeout: float = 10.0,
) -> object:
    """Run ``fn(*args)`` in a worker thread; raise ``SandboxTimeout`` on
    overrun.

    The honest limitation, stated plainly because a timeout that oversells
    itself is worse than none: a Python thread cannot be force-killed. If the
    function hard-spins, the thread leaks and keeps burning CPU until the
    process exits -- the worker is created as a daemon so the interpreter can
    still shut down, but until then the leak is real. This timeout protects
    the CALLER'S control flow (the synthesis loop gets to give up, report,
    and try a different candidate); it does not protect the process from the
    code it ran. Only a process boundary does that.
    """
    result: dict[str, object] = {}

    def runner() -> None:
        try:
            result["value"] = fn(*tuple(args))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            result["error"] = exc

    worker = threading.Thread(target=runner, daemon=True, name="cwm-sandbox-call")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise SandboxTimeout(
            f"call overran {timeout}s and was abandoned (its thread cannot be "
            "force-killed and may still be running)"
        )
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result.get("value")


# The CodeWorldModel surface, as (method -> number of positional parameters
# besides self). ``instantiate`` checks against this table so a model that
# renamed or re-argumented a method is caught here rather than mid-planning.
CWM_METHOD_PARAMS: dict[str, int] = {
    "initial_state": 0,
    "apply_action": 2,
    "get_current_player": 1,
    "get_legal_actions": 1,
    "get_observations": 1,
    "get_rewards": 1,
    "chance_outcomes": 1,
}


class ProtocolViolation(Exception):
    """The synthesised class does not satisfy the protocol it was told to."""


def check_protocol_methods(
    obj: object, methods: dict[str, int], label: str
) -> list[str]:
    """Return a list of human-readable problems, empty when ``obj`` satisfies
    the given (method -> positional arity) table.

    Arity is checked by whether the method can ACCEPT the required positional
    arguments (defaults and ``*args`` are fine) rather than by exact count --
    a model adding an optional parameter is harmless and refusing it would
    waste a refinement round on a non-problem.
    """
    problems: list[str] = []
    for name, required in methods.items():
        attr = getattr(obj, name, None)
        if attr is None:
            problems.append(f"{label}: missing method {name!r}")
            continue
        if not callable(attr):
            problems.append(f"{label}: attribute {name!r} is not callable")
            continue
        try:
            sig = signature(attr)
        except (TypeError, ValueError):
            continue
        positionals = [
            param
            for param in sig.parameters.values()
            if param.kind
            in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        # Drop the bound ``self``/``cls`` if the signature still carries it.
        if positionals and positionals[0].name in ("self", "cls"):
            positionals = positionals[1:]
        mandatory = [param for param in positionals if param.default is param.empty]
        takes_var = any(
            param.kind is param.VAR_POSITIONAL for param in sig.parameters.values()
        )
        if len(mandatory) > required or (not takes_var and len(positionals) < required):
            problems.append(
                f"{label}: method {name!r} takes {len(positionals)} positional "
                f"argument(s) ({len(mandatory)} mandatory), protocol requires "
                f"exactly {required}"
            )
    return problems


def instantiate(namespace: types.ModuleType, class_name: str) -> CodeWorldModel:
    """Find ``class_name`` in a sandboxed namespace, construct it, and verify
    it structurally satisfies ``CodeWorldModel``.

    Raises ``ProtocolViolation`` listing EVERY missing or mis-signatured
    method -- all at once, because the failing method list is the refinement
    prompt's most useful content and dribbling it out one method per
    synthesis round is how call budgets die.
    """
    cls = getattr(namespace, class_name, None)
    if cls is None:
        raise ProtocolViolation(
            f"class {class_name!r} not found in the synthesised module; "
            f"available names: {sorted(vars(namespace))}"
        )
    try:
        instance = cls()
    except Exception as exc:
        raise ProtocolViolation(
            f"{class_name} could not be constructed with no arguments: {exc}"
        ) from exc
    problems = check_protocol_methods(instance, CWM_METHOD_PARAMS, class_name)
    if problems:
        raise ProtocolViolation("; ".join(problems))
    return instance  # type: ignore[return-value]
