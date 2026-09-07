"""Names exported in ``__all__`` that no other module or test exercises.

Every test here exists because a name is public API (an operator or a
downstream caller is invited to use it), yet nothing in the repository
touched it. Shipping that is how an export rots: a refactor breaks it and
no suite notices. The heaviest case is deliberate: ``SubprocessSandbox`` is
the option SECURITY.md tells readers to reach for with untrusted source,
and until this file it had no test at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fronts.adapters.composio_io import (
    DryRunAdapter,
    PublishReceipt,
    ToolkitEntry,
    ToolkitRegistry,
)
from fronts.adapters.trajectory import TrajectoryStats, TrajectoryStore
from fronts.config import Paths, load
from fronts.cwm.arena import ArenaConfig, ArenaResult, play_episode
from fronts.cwm.arena import run as arena_run
from fronts.cwm.inference import (
    HISTORY_INFERENCE_CLASS,
    STATE_INFERENCE_CLASS,
    synthesise_history_inference,
    synthesise_state_inference,
)
from fronts.cwm.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxTimeout,
    SandboxViolation,
    SubprocessSandbox,
    _CappedBuffer,
    guard_methods,
)
from fronts.cwm.synth import SynthConfig, default_api_spec
from fronts.cwm.value import (
    VALUE_FUNCTION_CLASS,
    select_best,
    synthesise_value_functions,
)
from fronts.game.legality import (
    ACQUISITION_CTAS,
    LOW_RISK_CLAIM_CLASSES,
    SERVICE_CTAS,
    OperatorPolicy,
)
from fronts.game.priors import (
    ARCHETYPE_PRIORS,
    PLATFORM_PRIORS,
    SEMANTIC_TIER_PRIORS,
    VECTOR_PRIORS,
    ArchetypePrior,
    Band,
    PlatformPrior,
    SemanticPrior,
    VectorPrior,
)
from fronts.game.types import (
    Archetype,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    HookFamily,
    Observation,
    Platform,
    Publish,
    SemanticTier,
    Step,
    Trajectory,
)
from fronts.solvers.signalling import SignalCost, SignallingConfig, separating_power

# ---------------------------------------------------------------------------
# SubprocessSandbox, the SECURITY.md option, previously shipped untested.
# ---------------------------------------------------------------------------

SUBPROCESS_MODEL_SOURCE = '''\
HORIZON = 2


class WorldModel:
    def __init__(self):
        pass

    def initial_state(self):
        return {"day": 0}

    def apply_action(self, state, action):
        return {"day": state["day"] + 1}

    def get_current_player(self, state):
        return -4 if state["day"] >= HORIZON else 0

    def get_legal_actions(self, state):
        return [] if state["day"] >= HORIZON else ["hold"]

    def get_observations(self, state):
        return {}

    def get_rewards(self, state):
        return {0: 1.0}

    def chance_outcomes(self, state):
        return []


def double(x):
    return 2 * x
'''


def test_subprocess_sandbox_loads_and_calls_a_valid_module() -> None:
    """A well-formed module loads, its class constructs, and both method and
    module-function calls cross the process boundary with values intact."""
    sandbox = SubprocessSandbox()
    try:
        namespace = sandbox.load(SUBPROCESS_MODEL_SOURCE, SandboxConfig())
        model = namespace.WorldModel()
        # The reply crosses JSON, so dict keys arrive as strings, the
        # documented shape of everything that comes back from the child.
        assert model.get_rewards({"day": 0}) == {"0": 1.0}
        assert model.get_legal_actions({"day": 0}) == ["hold"]
        assert namespace.double(21) == 42
        missing = "no_such_name"
        with pytest.raises(AttributeError):
            getattr(namespace, missing)
    finally:
        sandbox.close()


def test_subprocess_sandbox_kills_a_runaway_child_on_timeout() -> None:
    """The whole point of the class: an overrun gets a real kill(), not the
    abandoned thread the in-process timeout settles for. ``process`` is
    exposed so this can be asserted rather than trusted."""
    sandbox = SubprocessSandbox()
    source = SUBPROCESS_MODEL_SOURCE.replace(
        "    def get_rewards(self, state):\n        return {0: 1.0}",
        "    def get_rewards(self, state):\n"
        "        while True:\n"
        "            pass\n",
    )
    try:
        model = sandbox.load(source, SandboxConfig(timeout_seconds=0.5)).WorldModel()
        assert sandbox.process is not None
        with pytest.raises(SandboxTimeout, match="killed"):
            model.get_rewards({"day": 0})
        # Killed, not abandoned: the child is gone.
        assert sandbox.process.poll() is not None
    finally:
        sandbox.close()


def test_subprocess_sandbox_refuses_the_known_escape_classes() -> None:
    """The parent's AST walk runs before any process is spawned: imports,
    forbidden names and dunder attribute access never reach a child."""
    for source in (
        "import os",
        "from random import _os",
        "x = object.__subclasses__()",
        "y = open('/etc/hostname')",
    ):
        sandbox = SubprocessSandbox()
        try:
            with pytest.raises(SandboxViolation):
                sandbox.load(source, SandboxConfig())
            assert sandbox.process is None  # nothing was ever spawned
        finally:
            sandbox.close()


def test_in_process_sandbox_does_not_leak_sys_modules_entries() -> None:
    """500 refinement calls must not leave 500 dead modules registered:
    the entry exists only for the duration of the load (dataclass string
    annotations resolve through sys.modules), then is removed in a finally."""
    before = {name for name in sys.modules if name.startswith("cwm_sandbox_")}
    source = "X = 1\n"
    for _ in range(25):
        Sandbox().load(source, SandboxConfig())
    after = {name for name in sys.modules if name.startswith("cwm_sandbox_")}
    assert after == before, f"sandbox loads leak sys.modules entries: {after - before}"


def test_output_cap_counts_bytes_not_characters() -> None:
    """``max_output_bytes`` means bytes. Fifteen accented characters are 30
    UTF-8 bytes; a cap that counted ``StringIO`` characters would admit them
    under a 20-unit budget, quietly allowing one and a half times what the
    config names. (Exercised on the buffer directly: the sandbox namespace
    deliberately has no ``print``, so a loaded module cannot produce output
    at all.)"""
    assert _CappedBuffer(20).write("x" * 20) == 20, "20 bytes at the cap is fine"
    with pytest.raises(SandboxViolation, match="bytes"):
        _CappedBuffer(20).write("é" * 15)


# ---------------------------------------------------------------------------
# guard_methods, Sandbox.guarded for non-world-model objects.
# ---------------------------------------------------------------------------


class _SlowSampler:
    def resample_state(self, history: object, player_id: int) -> dict:
        while True:
            pass

    def resample_history(self, history: object, player_id: int) -> list:
        return []


def test_guard_methods_wraps_only_the_named_methods() -> None:
    guarded = guard_methods(
        _SlowSampler(), ["resample_state", "resample_history"], timeout=0.25
    )
    assert guarded.resample_history([], 0) == []
    with pytest.raises(SandboxTimeout):
        guarded.resample_state([], 0)
    # Only the named methods exist: an __getattr__ answering every name would
    # make the proxy lie to hasattr-based protocol checks.
    assert not hasattr(guarded, "chance_outcomes")


# ---------------------------------------------------------------------------
# Inference and value-function synthesis entry points and their constants.
# ---------------------------------------------------------------------------


class _CapturingClient:
    """Records prompts and hands back one scripted response per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, stop: list[str] | None = None) -> str:
        self.prompts.append((system, user))
        return self._responses.pop(0)


_HISTORY_SAMPLER_SOURCE = f'''\
```python
class {HISTORY_INFERENCE_CLASS}:
    def resample_history(self, obs_action_history, player_id):
        return []
```
'''

_STATE_SAMPLER_SOURCE = f'''\
```python
class {STATE_INFERENCE_CLASS}:
    def resample_state(self, obs_action_history, player_id):
        return {{}}
```
'''

_VALUE_SOURCE = f'''\
```python
class {VALUE_FUNCTION_CLASS}:
    def __call__(self, state, player):
        return 0.5
```
'''


def test_synthesise_history_inference_returns_a_working_sampler() -> None:
    client = _CapturingClient([_HISTORY_SAMPLER_SOURCE])
    sampler = synthesise_history_inference(client, SynthConfig(), "rules", [])
    assert client.prompts, "the sampler must have been asked for"
    assert HISTORY_INFERENCE_CLASS in client.prompts[0][1]
    assert sampler.resample_history([], 0) == []


def test_synthesise_state_inference_prompt_names_the_required_class() -> None:
    client = _CapturingClient([_STATE_SAMPLER_SOURCE])
    sampler = synthesise_state_inference(client, SynthConfig(), "rules", [])
    assert STATE_INFERENCE_CLASS in client.prompts[0][1]
    assert sampler.resample_state([], 0) == {}


def test_synthesise_value_functions_returns_callable_candidates() -> None:
    client = _CapturingClient([_VALUE_SOURCE, _VALUE_SOURCE])
    candidates = synthesise_value_functions(client, SynthConfig(), "rules", 2)
    assert len(candidates) == 2
    assert VALUE_FUNCTION_CLASS in client.prompts[0][1]
    assert candidates[0]({"anything": 1}, 0) == 0.5


def test_select_best_requires_earning_the_place() -> None:
    vf = lambda state, player: 0.5  # noqa: E731 - a one-line ValueFunction
    assert select_best([]) is None
    assert select_best([(vf, 0.4)], baseline=0.5) is None
    assert select_best([(vf, 0.6)], baseline=0.5) is vf
    assert select_best([(vf, 0.6)], baseline=0.5, min_edge=0.2) is None


def test_default_api_spec_lifts_the_live_protocol() -> None:
    """The spec is generated from CodeWorldModel itself so it cannot drift
    from what the tests call; it must name every protocol method."""
    spec = default_api_spec()
    for method in (
        "initial_state",
        "apply_action",
        "get_current_player",
        "get_legal_actions",
        "get_observations",
        "get_rewards",
        "chance_outcomes",
    ):
        assert f"def {method}(" in spec, f"api spec is missing {method}"


# ---------------------------------------------------------------------------
# Arena result type, receipts, registry entries, stats, paths, configs.
# ---------------------------------------------------------------------------


class _OneShotWorld:
    """The smallest model the arena will run: one operator decision, done."""

    def initial_state(self) -> dict:
        return {"done": False}

    def apply_action(self, state: dict, action: object) -> dict:
        return {"done": True}

    def get_current_player(self, state: dict) -> int:
        return -4 if state["done"] else 0

    def get_legal_actions(self, state: dict) -> list:
        return [] if state["done"] else ["hold", "publish|x|hold-catalogue"]

    def get_observations(self, state: dict) -> dict:
        return {}

    def get_rewards(self, state: dict) -> dict:
        return {0: 1.0 if state["done"] else 0.0}

    def chance_outcomes(self, state: dict) -> list:
        return []


def _hold(model: object, state: dict) -> str:
    return "hold"


def _publish(model: object, state: dict) -> str:
    return "publish|x|hold-catalogue"


def test_arena_run_returns_a_complete_audit_trail() -> None:
    result = arena_run(
        [_OneShotWorld()], [_hold, _publish], ArenaConfig(matches_per_pairing=2)
    )
    assert isinstance(result, ArenaResult)
    assert result.hosts == [0]
    assert len(result.scores) == 2
    assert len(result.score_matrix) == 2
    assert sorted(result.survivors + result.rejected) == [0, 1]
    # Both agents play the same one-decision game, so neither is rejected:
    # a rejection here would mean the paired-play machinery is scoring
    # something other than achieved utility.
    assert result.rejected == []
    assert play_episode(_OneShotWorld(), _hold, seed=1) == 1.0


def _publish_move() -> Publish:
    return Publish(
        platform=Platform.TIKTOK,
        format=Format.CAROUSEL,
        archetype=Archetype.ANTI_HERO_RANT,
        vector=EmotionalVector.ANGER_INJUSTICE,
        semantic_tier=SemanticTier.TRIBAL_IDENTITY,
        hook=HookFamily.PATTERN_INTERRUPT,
        avatar="solo_founder",
        cta_mode=CtaMode.START_TRIAL,
        claim_class=ClaimClass.FIRST_PARTY_PROOF,
        angle="launch_theater",
        utm_content="c0412ab9",
    )


def _settled_observation(utm: str) -> Observation:
    return Observation(
        utm_content=utm,
        posted_at="2026-08-01",
        observed_at="2026-08-05",
        reach=1000,
        hook_rate=0.3,
        attributed_conversions=3,
    )


def test_dry_run_adapter_returns_a_stamp_carrying_receipt() -> None:
    receipt = DryRunAdapter().publish(
        _publish_move(), {"url": "https://example.com/pricing", "text": "t"}
    )
    assert isinstance(receipt, PublishReceipt)
    assert receipt.dry_run
    assert receipt.platform is Platform.TIKTOK
    assert "utm_content=c0412ab9" in receipt.url, "the receipt records the join key"


def test_registry_entry_resolves_overrides_into_a_toolkit_entry() -> None:
    registry = ToolkitRegistry()
    base = registry.entry(Platform.X)
    assert isinstance(base, ToolkitEntry)
    overridden = registry.entry(Platform.X, {"publish:x": "X_CREATE_POST_V2"})
    assert isinstance(overridden, ToolkitEntry)
    assert overridden.publish == "X_CREATE_POST_V2"
    assert overridden.analytics == base.analytics
    assert overridden.toolkit == base.toolkit


def test_store_stats_returns_the_cold_start_verdict(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "history.jsonl")
    trajectory = Trajectory(
        account="test",
        steps=[
            Step(move=_publish_move(), observation=_settled_observation("c0412ab9")),
            Step(move=_publish_move(), observation=None),
        ],
    )
    store.save(trajectory)
    stats = store.stats()
    assert isinstance(stats, TrajectoryStats)
    assert stats.moves == 2
    assert stats.settled_observations == 1
    assert stats.archetypes == 1
    assert not stats.ready_for_synthesis, "one archetype cannot clear the floor"


def test_paths_load_from_config_and_default_sensibly(tmp_path: Path) -> None:
    assert Paths().history == Path("data/trajectories.jsonl")
    config_file = tmp_path / "fronts.toml"
    config_file.write_text(
        '[paths]\nhistory = "elsewhere/history.jsonl"\n', encoding="utf-8"
    )
    paths = load(config_file).paths
    assert paths.history == Path("elsewhere/history.jsonl")
    # Untouched fields keep their defaults: setting one path must not
    # quietly un-set the others.
    assert paths.model == Path("data/model.py")


# ---------------------------------------------------------------------------
# Solver configs and claim-class tables.
# ---------------------------------------------------------------------------


def test_signalling_config_guards_the_zero_denominator() -> None:
    free = SignalCost(
        cost_high_type=0.0,
        cost_low_type=0.0,
        gain_from_deception=0.0,
        benefit_high_type=0.0,
    )
    config = SignallingConfig(epsilon=0.5)
    assert separating_power(free, config) == 0.0, (
        "a free claim against a free deception separates nothing, whatever "
        "the guard's size"
    )
    with pytest.raises(ValueError):
        SignallingConfig(epsilon=0.0)


def test_operator_policy_defaults_are_the_corpus_gates() -> None:
    """The two corpus gates, 50/7d creative judgement and 300/14d spend
    commitment (``fronts.game.types``, "Evidence gates"), are what
    ``OperatorPolicy`` enforces by default; this pins the numbers so a
    tuning edit has to be deliberate, not a drive-by."""
    policy = OperatorPolicy()
    assert policy.creative_judgement_min_conversions == 50
    assert policy.creative_judgement_window_days == 7
    assert policy.spend_commitment_min_conversions == 300
    assert policy.spend_commitment_min_days == 14


def test_low_risk_claim_classes_is_the_verifiability_line() -> None:
    assert frozenset(
        {
            ClaimClass.NO_CLAIM,
            ClaimClass.MECHANISM,
            ClaimClass.VERIFIABLE_METRIC,
            ClaimClass.FIRST_PARTY_PROOF,
        }
    ) == LOW_RISK_CLAIM_CLASSES
    # The risky side of the line, named so an added ClaimClass member has to
    # decide where it belongs rather than landing silently.
    assert not LOW_RISK_CLAIM_CLASSES & {
        ClaimClass.OUTCOME_PROMISE,
        ClaimClass.THIRD_PARTY_TESTIMONIAL,
        ClaimClass.COMPARATIVE,
    }


def test_the_lane_split_covers_every_cta_exactly_once() -> None:
    """ACQUISITION_CTAS drives the lane-split rule; SERVICE_CTAS is its
    complement. If the two ever overlap or miss a CTA mode, the day's
    cadence constraint is deciding some posts' lane by accident."""
    assert not ACQUISITION_CTAS & SERVICE_CTAS
    assert frozenset(CtaMode) == ACQUISITION_CTAS | SERVICE_CTAS


# ---------------------------------------------------------------------------
# Prior row types, the schema of the public tables.
# ---------------------------------------------------------------------------


def test_prior_tables_are_typed_by_their_exported_row_types() -> None:
    band = Band(0.1, 0.3, "corpus/file.md")
    assert isinstance(next(iter(ARCHETYPE_PRIORS.values())), ArchetypePrior)
    assert isinstance(next(iter(SEMANTIC_TIER_PRIORS.values())), SemanticPrior)
    assert isinstance(next(iter(PLATFORM_PRIORS.values())), PlatformPrior)
    for prior in VECTOR_PRIORS.values():
        # Unmeasured vectors are None, never a plausible guess.
        assert prior is None or isinstance(prior, VectorPrior)
    # The row types are constructible with cited bands and frozen against
    # silent mutation, like the numbers they carry.
    archetype = ArchetypePrior(
        hook_rate=band,
        hold_rate=band,
        outbound_ctr=band,
        expected_cvr=band,
    )
    assert archetype.hook_rate.mid == 0.2
    assert archetype.source
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclasses raise
        archetype.hook_rate = band  # type: ignore[misc]
