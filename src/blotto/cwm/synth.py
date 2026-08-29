"""The synthesis loop: ask a language model to write the simulator.

One candidate per call to ``synthesise``: build the prompt, complete it,
extract the code, sandbox-load it, instantiate the class, run the tests, and
report the pass rate. Refinement -- choosing which candidate to fix next --
lives in ``blotto.cwm.refine``; this module only ever produces ONE candidate,
so a test of the loop is a test of one turn of it.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field

from blotto.cwm.llm import LLMClient, extract_code
from blotto.cwm.sandbox import Sandbox, SandboxConfig, default_namespace, instantiate
from blotto.cwm.tests_from_traj import ModelTest, TestResult
from blotto.game.action_space import ActionCodec
from blotto.game.types import Trajectory
from blotto.protocols import CodeWorldModel

__all__ = [
    "SynthConfig",
    "SynthesisResult",
    "WORLD_MODEL_CLASS",
    "default_api_spec",
    "build_prompt",
    "synthesise",
]


WORLD_MODEL_CLASS = "WorldModel"
"""The single class name the prompt requires. Fixing it in the prompt and
checking for it at instantiate time removes an entire class of "the code is
fine but the entry point is called something else" failure."""


@dataclass(slots=True)
class SynthConfig:
    """Loop budget and sampling knobs.

    ``max_calls`` defaults to 500 because that is the paper's synthesis
    budget -- the budget Gin rummy exhausted at 0.78 train accuracy, which is
    the honest calibration for how far 500 calls gets you.

    ``temperature`` and ``model_name`` are consumed by ``blotto.cli`` when it
    builds a real provider client (``_client_from_env``); the ``LLMClient``
    protocol deliberately has no sampling parameters, so a fixture replay
    cannot drift from its recording.
    """

    model_name: str = "claude-sonnet-4-6"
    max_calls: int = 500
    num_tests_on_init: int = 5
    num_tests_on_error: int = 1
    temperature: float = 0.2


def default_api_spec() -> str:
    """Lift the required API from ``protocols.CodeWorldModel``.

    The required API is the regulariser. Per PAPER.md s3, quoting the paper:
    "Instead of a bottleneck, or a regularization term, the game rules and
    the required OpenSpiel API (used in the unit tests) introduced in the
    context of the LLM act as regularizers to prevent trivial latent spaces
    from being discovered." Stating the API in the prompt -- mechanically,
    from the live protocol, so it can never drift from what the tests call --
    is what stops the synthesiser from inventing a degenerate representation.
    """
    lines: list[str] = []
    for name, method in inspect.getmembers(CodeWorldModel, predicate=inspect.isfunction):
        signature = str(inspect.signature(method))
        doc = inspect.getdoc(method) or ""
        first_line = doc.splitlines()[0] if doc else ""
        lines.append(f"    def {name}{signature}:\n        \"\"\"{first_line}\"\"\" ...")
    return "\n".join(lines)


def _serialise_trajectory(trajectory: Trajectory) -> str:
    codec = ActionCodec()
    steps = []
    for step in trajectory.steps:
        obs = None
        if step.observation is not None:
            obs = {
                "utm_content": step.observation.utm_content,
                "is_partial": step.observation.is_partial,
                "reach": step.observation.reach,
                "hook_rate": step.observation.hook_rate,
                "hold_rate": step.observation.hold_rate,
                "comments": step.observation.comments,
                "shares": step.observation.shares,
                "attributed_conversions": step.observation.attributed_conversions,
                "attribution_coverage": step.observation.attribution_coverage,
                "incrementality": step.observation.incrementality,
                "dark_social_estimate": step.observation.dark_social_estimate,
            }
        steps.append({"action": codec.encode(step.move), "observation": obs})
    return json.dumps(steps, sort_keys=True, separators=(",", ":"))


def build_prompt(
    rules_text: str,
    trajectories: list[Trajectory],
    api_spec: str | None,
    previous_error: str | None,
) -> tuple[str, str]:
    """Return the (system, user) prompt for one synthesis call.

    Contains, always: the natural-language rules, a compact serialisation of
    the trajectories, and the REQUIRED API lifted from
    ``protocols.CodeWorldModel`` -- the API is the regulariser (PAPER.md s3;
    see ``default_api_spec``). Contains, on a refinement pass only, the
    failing test's traceback, because a stack trace is worth a thousand
    words of natural-language diagnosis.
    """
    api = api_spec if api_spec is not None else default_api_spec()
    # Tell the synthesiser what is bound rather than what is permitted. The
    # sandbox refuses every import statement, so a prompt phrased as "these
    # modules are allowed" invites code that cannot load and burns a refinement
    # round rediscovering it. Naming the bindings steers the first attempt.
    bound = ", ".join(sorted(default_namespace()))
    system = (
        "You are a program synthesiser. You write complete, runnable Python "
        "simulators of an organic content-distribution environment, conforming "
        "exactly to a required API. You write code, not explanations. Your "
        "output runs in a sandbox that permits NO import statements at all. "
        "These names are already bound in your module's namespace and are the "
        "only ones available: "
        + bound
        + ". Do not write `import` or `from ... import` anywhere; it will be "
        "rejected before execution. All methods must be deterministic "
        "functions of their arguments; randomness enters only through the "
        "chance player."
    )
    user_lines = [
        "Write a Python module implementing a world model of the environment "
        f"described below. The module MUST define a class named "
        f"{WORLD_MODEL_CLASS} taking no constructor arguments.",
        "",
        "== GAME RULES ==",
        rules_text.strip(),
        "",
        "== REQUIRED API (the class must satisfy this structurally) ==",
        api,
        "",
        "== OBSERVED TRAJECTORIES (action keys and the observations that came "
        "back; is_partial observations are inside the reporting lag and are "
        "provisional) ==",
    ]
    for index, trajectory in enumerate(trajectories):
        user_lines.append(f"trajectory[{index}]: {_serialise_trajectory(trajectory)}")
    if previous_error is not None:
        user_lines += [
            "",
            "== PREVIOUS ATTEMPT FAILED ==",
            "Your previous candidate failed the following test. Return a "
            "corrected, complete module.",
            previous_error.strip(),
        ]
    user_lines += [
        "",
        "Respond with a single fenced Python code block and nothing else.",
    ]
    return system, "\n".join(user_lines)


@dataclass(slots=True)
class SynthesisResult:
    """One candidate and how it fared.

    ``model`` is None when anything in the pipeline failed -- the model
    refused to emit code, the code violated the sandbox, the class was
    missing a protocol method -- and ``error`` says which. A candidate that
    loads but fails every test is a SUCCESS at this layer with pass_rate
    0.0: the loop's job is to fix those, and burying them as errors would
    starve refinement of its most informative starting points.
    """

    source: str
    model: CodeWorldModel | None
    pass_rate: float
    passed: int
    total: int
    llm_calls: int
    error: str | None = None
    failures: list[TestResult] = field(default_factory=list)


def synthesise(
    client: LLMClient,
    config: SynthConfig,
    rules: str,
    trajectories: list[Trajectory],
    tests: list[ModelTest],
    previous_error: str | None = None,
) -> SynthesisResult:
    """Produce one candidate and measure it against ``tests``.

    The caller decides how many tests to pass in: an initial call uses a
    handful (``SynthConfig.num_tests_on_init``) and a refinement pass uses
    the failing one (``num_tests_on_error``), per the paper's REx procedure.
    """
    system, user = build_prompt(rules, trajectories, None, previous_error)
    response = client.complete(system, user)
    source = extract_code(response)

    model: CodeWorldModel
    try:
        namespace = Sandbox().load(source, SandboxConfig())
        model = instantiate(namespace, WORLD_MODEL_CLASS)
    except Exception as exc:
        return SynthesisResult(
            source=source,
            model=None,
            pass_rate=0.0,
            passed=0,
            total=len(tests),
            llm_calls=1,
            error=f"{type(exc).__name__}: {exc}",
        )

    failures: list[TestResult] = []
    passed = 0
    for test in tests:
        result = test.run(model)
        if result.passed:
            passed += 1
        else:
            failures.append(result)
    pass_rate = passed / len(tests) if tests else 0.0
    return SynthesisResult(
        source=source,
        model=model,
        pass_rate=pass_rate,
        passed=passed,
        total=len(tests),
        llm_calls=1,
        failures=failures,
    )
