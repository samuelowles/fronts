"""Executing code a language model wrote.

Be honest about what this is and is not.

What the sandbox guarantees: no import statement of any kind reaches the
interpreter -- every ``import`` and ``from ... import ...`` is refused at
compile time, by walking the AST BEFORE any of the source executes -- and the
namespace a module runs in contains no module objects at all, only
pre-bound functions, classes and constants (see ``default_namespace``). A
module object is a pointer into the whole importable graph, because
allowlisted modules re-export dangerous ones as ordinary attributes
(``random._os`` is ``os``); refusing modules as VALUES, not just as imports,
is what closes that class of escape. On top of that, the AST walk refuses
``open``/``exec``/``eval``/``compile``, dunder attribute access,
``__subclasses__``/``__globals__``/``__code__``, and the builtins the module
sees are an explicit allowlist. Module-level code and every call through
``Sandbox.guarded`` run under ``call_with_timeout``.

What it does NOT guarantee: this is still the same interpreter, in the same
process, with the same memory. A determined adversary who finds a way to
mutate an object the sandbox handed it, or to spin a thread the timeout
cannot abandon, is out of scope -- the timeout protects the caller's control
flow, not the process. Anyone running genuinely untrusted synthesis output
should use ``SubprocessSandbox`` (a real process boundary with a hard kill)
or, better, a container, and accept the per-call IPC cost that makes the
subprocess variant unsuitable for an inner MCTS loop.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import copy
import dataclasses
import io
import itertools
import json
import math
import queue
import random
import subprocess
import sys
import threading
import types
import typing
from collections.abc import Callable, Iterable
from inspect import signature

from blotto.game.types import Observation
from blotto.protocols import CodeWorldModel

__all__ = [
    "SandboxConfig",
    "Sandbox",
    "SubprocessSandbox",
    "SandboxViolation",
    "SandboxTimeout",
    "ProtocolViolation",
    "call_with_timeout",
    "default_namespace",
    "guard_methods",
    "instantiate",
    "check_protocol_methods",
]


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Knobs for one sandboxed execution.

    ``namespace_extras`` adds or overrides pre-bound VALUES (never modules)
    in the namespace the source runs in -- a caller with a richer vocabulary
    for its models extends the sandbox here rather than reopening imports.
    """

    timeout_seconds: float = 10.0
    max_output_bytes: int = 8192
    namespace_extras: dict[str, object] | None = None


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


_RNG_SEED = 20260821
"""Seed for the per-load ``rng`` instance. Fixed, so loading the same source
twice runs the same draw sequence; a fresh instance per load, so two modules
never share a stream."""


def default_namespace(extras: dict[str, object] | None = None) -> dict[str, object]:
    """The values a sandboxed module starts with: functions, classes and
    constants, and NOT ONE module object.

    This is the replacement for the module allowlist, and the reason it
    works where the allowlist could not: ``math.sqrt`` is a function with no
    reachable ``_os`` behind it, while ``math`` the module is a doorway into
    the entire import graph. Anything a world model legitimately needs from
    the stdlib is bound here by name, or is not available.
    """
    namespace: dict[str, object] = {
        # numerics
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "fabs": math.fabs,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "isfinite": math.isfinite,
        "hypot": math.hypot,
        # randomness: the class, and one deterministic-per-load instance
        "Random": random.Random,
        "rng": random.Random(_RNG_SEED),
        # structured definitions
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "Any": typing.Any,
        "Optional": typing.Optional,
        # containers and iteration
        "defaultdict": collections.defaultdict,
        "deque": collections.deque,
        "Counter": collections.Counter,
        "chain": itertools.chain,
        "product": itertools.product,
        "islice": itertools.islice,
        "copy": copy.copy,
        "deepcopy": copy.deepcopy,
    }
    if extras:
        namespace.update(extras)
    return namespace


def _bound_names(config: SandboxConfig) -> list[str]:
    return list(default_namespace(config.namespace_extras))


class _ForbiddenVisitor(ast.NodeVisitor):
    """Collects every violation, so one refusal names them all rather than
    making the author fix them one round-trip at a time."""

    def __init__(self, bound: list[str]) -> None:
        self.bound = bound
        self.violations: list[str] = []

    def _flag(self, node: ast.stmt | ast.expr, what: str) -> None:
        self.violations.append(
            f"{what} ({type(node).__name__}) at line {node.lineno}"
        )

    def _import_hint(self) -> str:
        listed = ", ".join(f"`{name}`" for name in self.bound)
        return (
            f"imports are not available in the sandbox; {listed} are already "
            "bound -- use those names directly, and write everything else "
            "yourself"
        )

    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(alias.name for alias in node.names)
        self._flag(node, f"import of {names}")
        # The hint rides along on the first import flagged, so one refusal
        # carries the whole remedy rather than naming the disease only.
        self.violations.append(self._import_hint())
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or "<relative>"
        names = ", ".join(alias.name for alias in node.names)
        self._flag(node, f"import of {names} from {module}")
        self.violations.append(self._import_hint())
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


def _check_source(source: str, bound: list[str]) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxViolation(f"syntax error: {exc}") from exc
    visitor = _ForbiddenVisitor(bound)
    visitor.visit(tree)
    if visitor.violations:
        raise SandboxViolation("; ".join(visitor.violations))


class _CappedBuffer(io.StringIO):
    """Output sink that refuses to buffer past the cap, so a model that
    prints its way through the memory budget fails loudly instead of
    quietly.

    The cap counts UTF-8 BYTES, matching the ``max_output_bytes`` name:
    ``StringIO.tell()`` counts characters, and a model printing emoji or CJK
    text gets several bytes per character -- a character-counted cap would
    quietly admit multiples of the budget the config names."""

    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap
        self._bytes_written = 0

    def write(self, text: str) -> int:
        size = len(text.encode("utf-8"))
        if self._bytes_written + size > self._cap:
            raise SandboxViolation(
                f"sandboxed output exceeded {self._cap} bytes -- write less, "
                "or raise SandboxConfig.max_output_bytes"
            )
        self._bytes_written += size
        return super().write(text)


def _safe_builtins() -> dict[str, object]:
    """The builtins a world model legitimately needs and nothing else.

    There is deliberately NO ``__import__`` here, not even a refusing one:
    imports are refused at compile time and nothing executable ever asks the
    import machinery for anything. ``__build_class__`` and ``__name__`` have
    to be present or ``class`` statements fail; they are injected by the
    sandbox, never authored by the model, which is the distinction the
    dunder rule in the AST walker is drawing.
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
        The ``exec`` itself runs under ``call_with_timeout``, so a module
        whose top level never terminates raises ``SandboxTimeout`` instead of
        wedging the caller (the abandoned thread cannot be force-killed; see
        ``call_with_timeout``).
        """
        _check_source(source, _bound_names(config))

        # A unique name, and temporary registration in sys.modules, because
        # the dataclasses machinery resolves string annotations by looking
        # the defining module up there -- an unregistered module breaks
        # ``@dataclass`` in the synthesised code. The entry is removed in a
        # ``finally`` so 500 refinement calls do not leave 500 dead modules
        # in sys.modules.
        Sandbox._load_counter += 1
        module_name = f"cwm_sandbox_{Sandbox._load_counter}"
        module = types.ModuleType(module_name)
        module.__dict__.update(default_namespace(config.namespace_extras))
        module.__dict__["__builtins__"] = _safe_builtins()
        sys.modules[module_name] = module
        code = compile(source, "<cwm_sandbox>", "exec")

        def run() -> None:
            exec(code, module.__dict__)  # noqa: S102 - the sandbox IS the point

        # The redirect is installed on the CALLER's thread, not inside the
        # worker: a timed-out worker is abandoned mid-``with`` and would
        # otherwise leave stdout pointed at a dead buffer forever. stdout and
        # stderr both flow through the cap. stdin needs no redirection:
        # ``input`` is not in the safe builtins, so a world model has no way
        # to ask the operator questions in the first place.
        buffer = _CappedBuffer(config.max_output_bytes)
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(
                buffer
            ):
                call_with_timeout(run, (), config.timeout_seconds)
        finally:
            sys.modules.pop(module_name, None)
        return module

    @staticmethod
    def guarded(model: CodeWorldModel, timeout: float) -> CodeWorldModel:
        """Wrap ``model`` so every ``CodeWorldModel`` method call runs under
        ``call_with_timeout``.

        The load-time timeout only covers module top-level code. A hang
        inside ``apply_action`` during a 1000-simulation search would wedge
        the planner forever; wrapping each call means the planner gets a
        ``SandboxTimeout`` and can give up on the candidate.
        """
        return guard_methods(model, CWM_METHOD_PARAMS, timeout)  # type: ignore[return-value]


class _GuardedProxy:
    """Forwards the given method names through ``call_with_timeout``.

    Only the named methods exist on the proxy: an ``__getattr__`` that
    answered every name would make ``hasattr``-based protocol checks
    meaningless, which is exactly what ``check_protocol_methods`` and
    ``isinstance`` against the runtime-checkable protocols rely on.
    """

    def __init__(
        self, inner: object, method_names: Iterable[str], timeout: float
    ) -> None:
        self._inner = inner
        self._timeout = timeout
        for name in method_names:
            setattr(self, name, self._wrap(getattr(inner, name)))

    def _wrap(self, fn: Callable[..., object]) -> Callable[..., object]:
        def wrapped(*args: object) -> object:
            return call_with_timeout(fn, args, self._timeout)

        return wrapped


def guard_methods(
    obj: object, method_names: Iterable[str], timeout: float
) -> object:
    """``Sandbox.guarded`` for objects that are not world models -- inference
    samplers, value functions -- wrapping each named method in
    ``call_with_timeout``."""
    return _GuardedProxy(obj, method_names, timeout)


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
    synthesis round is how call budgets die. The final ``isinstance`` check
    is the same claim restated for the type checker: after the arity table
    has passed, the instance must also satisfy the runtime-checkable
    protocol itself.
    """
    cls = getattr(namespace, class_name, None)
    if cls is None:
        raise ProtocolViolation(
            f"class {class_name!r} not found in the synthesised module; "
            f"available names: {sorted(vars(namespace))}"
        )
    try:
        instance: object = cls()
    except Exception as exc:
        raise ProtocolViolation(
            f"{class_name} could not be constructed with no arguments: {exc}"
        ) from exc
    problems = check_protocol_methods(instance, CWM_METHOD_PARAMS, class_name)
    if problems:
        raise ProtocolViolation("; ".join(problems))
    if not isinstance(instance, CodeWorldModel):
        raise ProtocolViolation(
            f"{class_name} satisfies the method table but not the "
            "CodeWorldModel protocol"
        )
    return instance


# ---------------------------------------------------------------------------
# Subprocess sandbox.
#
# The in-process sandbox raises the cost of an accident; it is not a wall,
# and its own docstring says so. This class IS a wall -- or at least a real
# process boundary with a hard kill -- at the price of one inter-process
# round trip per METHOD CALL, which is why nothing in the planning loop uses
# it: an ISMCTS search that makes hundreds of thousands of CWM calls would
# spend its entire budget on pipes. Use it when the source is genuinely
# untrusted and rare calls are acceptable.
# ---------------------------------------------------------------------------


_CHILD_STARTUP_GRACE_SECONDS = 10.0
"""Extra budget the load request gets on top of ``timeout_seconds``: it pays
for spawning a fresh interpreter, which varies by machine and has nothing to
do with the untrusted code's own running time."""


_WORKER = r'''
"""Child half of SubprocessSandbox. Stdlib only; never imports blotto.

Speaks one JSON object per line on stdout, reads one JSON object per line
from stdin. The module under test runs in this process, in a namespace with
no module objects and no ``__import__`` at all, so even a construct the
parent's AST walk somehow missed has nothing to import through.
"""
import builtins
import contextlib
import dataclasses
import io
import json
import math
import random
import sys
import traceback
from collections import Counter, defaultdict, deque
from copy import copy, deepcopy
from dataclasses import dataclass, field
from itertools import chain, islice, product
from typing import Any, Optional

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash",
    "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ArithmeticError", "AttributeError", "IndexError",
    "KeyError", "LookupError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError", "NotImplementedError",
)


class _Capped(io.StringIO):
    # Counts UTF-8 bytes, not characters, for the same reason the parent's
    # _CappedBuffer does: the cap's name promises bytes.
    def __init__(self, cap):
        super().__init__()
        self._cap = cap
        self._bytes_written = 0

    def write(self, text):
        size = len(text.encode("utf-8"))
        if self._bytes_written + size > self._cap:
            raise RuntimeError(
                f"sandboxed output exceeded {self._cap} bytes"
            )
        self._bytes_written += size
        return super().write(text)


def _namespace():
    return {
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
        "pow": math.pow, "pi": math.pi, "e": math.e, "inf": math.inf,
        "isfinite": math.isfinite, "hypot": math.hypot,
        "Random": random.Random, "rng": random.Random(20260821),
        "dataclass": dataclass, "field": field,
        "Any": Any, "Optional": Optional,
        "defaultdict": defaultdict, "deque": deque, "Counter": Counter,
        "chain": chain, "product": product, "islice": islice,
        "copy": copy, "deepcopy": deepcopy,
    }


def _encode(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__name__, **dataclasses.asdict(obj)}
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _send(obj):
    sys.stdout.write(json.dumps(obj, default=_encode) + "\n")
    sys.stdout.flush()


_HANDLES = {}
_NEXT_HANDLE = [0]


def main():
    init = json.loads(sys.stdin.readline())
    ns = _namespace()
    builtins_dict = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    builtins_dict["__build_class__"] = builtins.__build_class__
    builtins_dict["__name__"] = "cwm_sandbox_module"
    ns["__builtins__"] = builtins_dict
    cap = int(init.get("max_output_bytes", 8192))
    try:
        code = compile(init["source"], "<cwm_sandbox>", "exec")
        buffer = _Capped(cap)
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, ns)  # noqa: S102 - the sandbox IS the point
    except BaseException:
        _send({"ok": False, "error": traceback.format_exc(limit=8)})
        return
    _send({
        "ok": True,
        "names": sorted(k for k in ns if not k.startswith("__")),
        "classes": sorted(k for k, v in ns.items() if isinstance(v, type)),
    })
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request["op"] == "construct":
                handle = _NEXT_HANDLE[0]
                _NEXT_HANDLE[0] += 1
                instance = ns[request["name"]]()
                _HANDLES[handle] = instance
                methods = sorted(
                    k
                    for k in dir(instance)
                    if not k.startswith("_") and callable(getattr(instance, k))
                )
                _send({"ok": True, "handle": handle, "methods": methods})
            elif request["op"] == "call":
                instance = _HANDLES[request["handle"]]
                value = getattr(instance, request["method"])(*request["args"])
                _send({"ok": True, "value": value})
            elif request["op"] == "call_module":
                value = ns[request["name"]](*request["args"])
                _send({"ok": True, "value": value})
            else:
                _send({"ok": False, "error": f"unknown op {request['op']!r}"})
        except BaseException:
            _send({"ok": False, "error": traceback.format_exc(limit=4)})


main()
'''


def _revive(value: object) -> object:
    """Rebuild parent-side objects the child could only send as JSON.

    ``Observation`` is the one dataclass on the CWM surface, so it is the one
    type revived; a dict whose keys are all digits and whose values all
    revive to Observations is ``get_observations``'s ``dict[int, Observation]``
    coming back with JSON-stringified keys.
    """
    if isinstance(value, dict):
        marker = value.get("__dataclass__")
        if marker == "Observation":
            fields = {k: v for k, v in value.items() if k != "__dataclass__"}
            try:
                return Observation(**fields)
            except TypeError:
                # A child-side class that shares only the name; hand back the
                # plain dict rather than losing the value.
                return fields
        revived = {k: _revive(v) for k, v in value.items()}
        if (
            revived
            and all(k.isdigit() for k in revived)
            and all(isinstance(v, Observation) for v in revived.values())
        ):
            return {int(k): v for k, v in revived.items()}
        return revived
    if isinstance(value, list):
        return [_revive(item) for item in value]
    return value


class _RemoteCallable:
    def __init__(self, sandbox: SubprocessSandbox, name: str) -> None:
        self._sandbox = sandbox
        self._name = name

    def __call__(self, *args: object) -> object:
        return self._sandbox._value(
            {"op": "call_module", "name": self._name, "args": list(args)}
        )


class _RemoteInstance:
    def __init__(
        self,
        sandbox: SubprocessSandbox,
        handle: int,
        methods: Iterable[str],
        timeout: float,
    ) -> None:
        self._sandbox = sandbox
        self._handle = handle
        self._timeout = timeout
        for name in methods:
            setattr(self, name, self._wrap(name))

    def _wrap(self, name: str) -> Callable[..., object]:
        def wrapped(*args: object) -> object:
            return self._sandbox._value(
                {"op": "call", "handle": self._handle, "method": name, "args": list(args)},
                timeout=self._timeout,
            )

        return wrapped


class _RemoteClass:
    def __init__(self, sandbox: SubprocessSandbox, name: str) -> None:
        self._sandbox = sandbox
        self._name = name

    def __call__(self, *args: object) -> object:
        reply = self._sandbox._request({"op": "construct", "name": self._name})
        if not reply.get("ok"):
            raise SandboxViolation(
                f"constructing {self._name} in the child failed: "
                f"{reply.get('error', 'unknown error')}"
            )
        handle = reply.get("handle")
        methods = reply.get("methods")
        # The reply crossed a process boundary; trust nothing about its
        # shape until it has been checked.
        if (
            not isinstance(handle, int)
            or isinstance(handle, bool)
            or not isinstance(methods, list)
            or not all(isinstance(name, str) for name in methods)
        ):
            raise SandboxViolation(
                f"constructing {self._name} returned a malformed reply"
            )
        return _RemoteInstance(
            self._sandbox, handle, methods, self._sandbox._timeout
        )


class SubprocessSandbox:
    """Load and call synthesised code in a SEPARATE process, killed on
    timeout.

    This is the option for genuinely untrusted source: the child runs
    ``python -c`` with the same no-imports namespace rules, every exchange is
    JSON over stdin/stdout, and an overrun gets a real ``kill()`` -- not the
    abandoned thread the in-process timeout settles for. The price is one
    pipe round trip per METHOD CALL, which is why it is unsuitable for the
    inner MCTS loop: use ``Sandbox`` there, and this when the source has no
    business sharing your interpreter.

    Exception objects cannot cross a process boundary: refusals detected by
    the parent's AST walk raise ``SandboxViolation`` as usual, and anything
    the child raises comes back as text wrapped in a ``SandboxViolation``.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._timeout = 10.0
        #: The most recent child process, exposed so a caller (or a test)
        #: can verify a timeout actually killed it.
        self.process: subprocess.Popen[str] | None = None

    def load(self, source: str, config: SandboxConfig) -> types.ModuleType:
        """Same contract as ``Sandbox.load``; the namespace it returns is a
        proxy whose attribute access constructs and calls in the child."""
        _check_source(source, _bound_names(config))
        self.close()
        self._timeout = config.timeout_seconds
        proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        self.process = proc
        self._replies = queue.Queue()

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._replies.put(line)
            self._replies.put(None)

        threading.Thread(target=pump, daemon=True, name="cwm-subprocess-read").start()

        reply = self._request(
            {
                "op": "load",
                "source": source,
                "max_output_bytes": config.max_output_bytes,
            },
            # Interpreter startup is environment cost, not model code: on a
            # cold Windows box it alone can exceed a sub-second call budget.
            # The kill guarantee stands -- the load budget is larger, not gone.
            timeout=config.timeout_seconds + _CHILD_STARTUP_GRACE_SECONDS,
        )
        if not reply.get("ok"):
            self.close()
            raise SandboxViolation(
                "subprocess sandbox refused or failed the source: "
                f"{reply.get('error', 'unknown error')}"
            )
        raw_classes = reply.get("classes", [])
        raw_names = reply.get("names", [])
        if not isinstance(raw_classes, list) or not isinstance(raw_names, list):
            self.close()
            raise SandboxViolation("child sent a malformed load acknowledgement")
        classes = {name for name in raw_classes if isinstance(name, str)}
        names = {name for name in raw_names if isinstance(name, str)}
        sandbox = self

        def module_getattr(name: str) -> object:
            if name in classes:
                return _RemoteClass(sandbox, name)
            if name in names:
                return _RemoteCallable(sandbox, name)
            raise AttributeError(f"{name!r} is not defined by the sandboxed module")

        module = types.ModuleType("cwm_subprocess_module")
        module.__dict__["__getattr__"] = module_getattr
        module.__dict__["__sandbox__"] = self
        return module

    def _request(
        self, payload: dict[str, object], timeout: float | None = None
    ) -> dict[str, object]:
        """Send one JSON request, return the child's reply dict, hard-killing
        the child if it does not answer inside the budget."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise SandboxViolation("subprocess sandbox has no live child process")
        budget = self._timeout if timeout is None else timeout
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise SandboxViolation(f"child process went away: {exc}") from exc
        try:
            line = self._replies.get(timeout=max(budget, 0.001))
        except queue.Empty:
            # The one thing this class exists to guarantee: a hard kill, not
            # an abandoned thread.
            self.close()
            raise SandboxTimeout(
                f"child overran {budget}s and was killed"
            ) from None
        if line is None:
            detail = self._stderr_tail()
            self.close()
            raise SandboxViolation(f"child process exited before replying: {detail}")
        reply = _revive(json.loads(line))
        # The protocol is one JSON OBJECT per line; anything else means the
        # child is misbehaving or dead, and handing an unvalidated value back
        # would move the failure into whichever caller unpacks it next.
        if not isinstance(reply, dict):
            self.close()
            raise SandboxViolation(
                f"child replied with a {type(reply).__name__}, not a JSON object"
            )
        return reply

    def _value(
        self, payload: dict[str, object], timeout: float | None = None
    ) -> object:
        """One RPC call: reply envelope stripped, errors raised here so a
        call through the proxy behaves like a call in process."""
        reply = self._request(payload, timeout)
        if not reply.get("ok"):
            raise SandboxViolation(
                f"call failed inside the child process: {reply.get('error', '')}"
            )
        return reply.get("value")

    def _stderr_tail(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            return proc.stderr.read()[-2000:] or ""
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        """Kill the child (if any) and drop the pipes. Safe to call twice."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
