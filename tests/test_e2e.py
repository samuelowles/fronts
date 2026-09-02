"""End-to-end tests: the whole loop, across module boundaries, on outcomes.

The unit suites prove each layer correct in isolation. This file proves the
LOOP -- history through the store, tests from the history, a report a human
reads, search turned into a decision, a day allocated, briefs produced,
moves shipped through an adapter, observations ingested back, and the whole
thing re-planned. Each scenario below exists because a specific class of bug
lives only at that altitude: a refusal that exits the wrong code, a floor
that reports cleared when it is not, a bad model that crashes the reporter
instead of scoring low, a sandbox escape that takes the planner down with
it, a plan that is not reproducible across processes.

Everything runs offline: no network (scenario 8 blocks sockets outright),
no credentials (every key the product reads is cleared), no LLM (where
synthesis must run, ``RecordedClient`` replays a fixture). A test here that
needed any of those would be a broken test.

The eight scenarios, in the order an operator would meet them:

1. The golden path: history -> stats -> tests -> report -> search ->
   allocation -> briefs -> publish -> ingest -> re-plan.
2. The measurement canary, through the report surface an operator reads.
3. A legality refusal is visible and exits exactly 2, with its citation.
4. The cold-start floor is reported honestly, at the boundary.
5. A bad world model scores low and is rejected by the arena.
6. Hostile synthesis output degrades to weaker play, never a crash.
7. The same seed produces a byte-identical plan in a fresh process.
8. All of it works with the network physically unavailable.

Plus the synth -> plan handoff: the inference sampler persisted beside the
model is found by ``plan`` (which names its determinization either way) and
scored by ``accuracy`` (whose inference column is n/a until then).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

# Reused, not reinvented: the hand-authored "synthesised" candidate, the
# rules text and the fixture-wrapper all live in the CWM unit suite already.
from test_cwm import RULES_TEXT, V1_SOURCE, V2_SOURCE, fenced, make_trajectory

from fronts.adapters.composio_io import DryRunAdapter
from fronts.adapters.trajectory import (
    COLD_START_MIN_SETTLED,
    TrajectoryStore,
)
from fronts.cli import _brief_for, _legality_context, _play_trajectory, _sample_chance, app
from fronts.config import default_config
from fronts.cwm.arena import ArenaConfig
from fronts.cwm.arena import run as arena_run
from fronts.cwm.inference import FallbackInference
from fronts.cwm.llm import RecordedClient
from fronts.cwm.reference import ReferenceConfig, ReferenceWorldModel
from fronts.cwm.report import ModelQualityReport
from fronts.cwm.sandbox import Sandbox, SandboxConfig, instantiate
from fronts.cwm.synth import SynthConfig, build_prompt, synthesise
from fronts.cwm.tests_from_traj import NO_CRASH, ModelTest, Tolerance, generate, split
from fronts.game.action_space import ActionCodec
from fronts.game.legality import LegalityEngine
from fronts.game.types import (
    CHANCE_PLAYER,
    OPERATOR,
    TERMINAL_PLAYER,
    ActionKey,
    Archetype,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    HookFamily,
    Kill,
    Move,
    Observation,
    Platform,
    Publish,
    Scale,
    SemanticTier,
    State,
    Step,
    Trajectory,
)
from fronts.protocols import CodeWorldModel
from fronts.solvers.blotto import BlottoAllocator, BlottoConfig, Front
from fronts.solvers.ismcts import ISMCTS, ISMCTSConfig

pytestmark = pytest.mark.e2e

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CODEC = ActionCodec()
_TODAY = date.today()
_ANGLES = ("launch_theater", "enemy_of_slop", "founder_field_notes")


# ---------------------------------------------------------------------------
# Shared helpers. Small, and shaped by what the scenarios below keep needing.
# ---------------------------------------------------------------------------


def _publish(**overrides: object) -> Publish:
    fields: dict[str, object] = {
        "platform": Platform.X,
        "format": Format.TEXT_THREAD,
        "archetype": Archetype.FOUNDER_TRAUMA,
        "vector": EmotionalVector.EXHAUSTION_RELIEF,
        "semantic_tier": SemanticTier.TRIBAL_IDENTITY,
        "hook": HookFamily.CURIOSITY_GAP,
        "avatar": "solo_founder",
        "cta_mode": CtaMode.START_TRIAL,
        "claim_class": ClaimClass.FIRST_PARTY_PROOF,
        "angle": "launch_theater",
        "utm_content": "c0412ab9",
    }
    fields.update(overrides)
    return Publish(**fields)  # type: ignore[arg-type]


def _settled_observation(utm: str, conversions: int) -> Observation:
    return Observation(
        utm_content=utm,
        posted_at=(_TODAY - timedelta(days=5)).isoformat(),
        observed_at=_TODAY.isoformat(),
        reach=2000,
        attributed_conversions=conversions,
        attribution_coverage=0.72,
        incrementality=0.70,
    )


def _write_config(tmp_path: Path, *, measurements: bool = True) -> Path:
    """A config pointing every path at ``tmp_path``.

    Measurements included by default so the evidence gates see an
    above-floor coverage; passing ``measurements=False`` leaves them unset,
    which is how an angle becomes unjudgeable and the coverage rule fires.
    """
    text = (
        "[paths]\n"
        f'history = "{(tmp_path / "history.jsonl").as_posix()}"\n'
        f'rules    = "{(tmp_path / "rules.md").as_posix()}"\n'
        f'model    = "{(tmp_path / "model.py").as_posix()}"\n'
        f'plan     = "{(tmp_path / "plan.json").as_posix()}"\n'
        f'briefs   = "{(tmp_path / "briefs.json").as_posix()}"\n'
        "\n[publish]\ndestination_url = \"https://owles.works/pricing\"\n"
    )
    if measurements:
        text += "\n[measurement]\nattribution_coverage = 0.72\nincrementality = 0.70\n"
    path = tmp_path / "fronts.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _write_plan(tmp_path: Path, moves: list[Move]) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"moves": [_CODEC.encode(move) for move in moves]}), encoding="utf-8"
    )
    return plan


def _refusal_source(output: str) -> str:
    """Pull the source citation out of a printed refusal."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("source:"):
            return stripped[len("source:") :].strip()
    return ""


def _pass_rate(suite: list[ModelTest], model: CodeWorldModel) -> float:
    results = [test.run(model) for test in suite]
    if not results:
        return 0.0
    return sum(1 for result in results if result.passed) / len(results)


def _plan_days(
    model: CodeWorldModel,
    planner: ISMCTS,
    rng: random.Random,
    days: int,
    sims: int,
) -> list[Move]:
    """The CLI's planning loop, compactly: one search per operator day, the
    chance ply resolved between days from the seeded rng."""
    state = model.initial_state()
    plan: list[Move] = []
    for _day in range(days):
        if model.get_current_player(state) == TERMINAL_PLAYER:
            break
        if model.get_current_player(state) == CHANCE_PLAYER:
            state = model.apply_action(state, _sample_chance(model, state, rng))
        if model.get_current_player(state) != OPERATOR:
            break
        result = planner.plan(model, state, OPERATOR, sims)
        plan.append(_CODEC.decode(result.move))
        state = model.apply_action(state, result.move)
        if model.get_current_player(state) == CHANCE_PLAYER:
            state = model.apply_action(state, _sample_chance(model, state, rng))
    return plan


def _first_publish(model: CodeWorldModel, state: State) -> ActionKey:
    legal = model.get_legal_actions(state)
    for action in legal:
        if str(action).startswith("publish"):
            return action
    return legal[0]


# ---------------------------------------------------------------------------
# Scenario 1: the golden path.
# ---------------------------------------------------------------------------


def test_full_loop_from_history_to_publish_and_back(
    reference_trajectory: Callable[..., Trajectory],
    tmp_store: TrajectoryStore,
    dry_adapter: DryRunAdapter,
    no_network: None,
) -> None:
    """The complete loop in one test, because the loop is the product.

    Each stage hands its output to the next through the real boundary: the
    store persists what the reference model recorded, test generation reads
    it back through a save/load round trip, the report carries the pass
    rates, ISMCTS turns the model into decisions, Blotto spreads the day,
    briefs carry the join key, the dry adapter ships intents, ingest brings
    the (empty, honestly empty) observations home, and the enlarged history
    re-plans. The class of bug caught here is the wiring bug: any seam where
    stage N's output is not stage N+1's input -- a dropped ``Trajectory.chance``,
    a move that fails the legality gate the publisher will apply, a store
    that records fewer moves than were shipped -- fails one of the asserts
    below, none of which a per-layer unit test can express.
    """
    config = default_config()
    config.attribution_coverage = 0.72
    config.incrementality = 0.70
    config.destination_url = "https://owles.works/pricing"
    ref_config = ReferenceConfig(horizon=12, seed=7)
    model = ReferenceWorldModel(ref_config)

    # History: the only training signal there is, generated not fetched.
    trajectory = reference_trajectory(horizon=12)
    tmp_store.save(trajectory)
    stats = tmp_store.stats()
    assert stats.moves == len(trajectory.steps)
    assert stats.settled_observations >= 1, "a twelve-day posting run must settle data"

    reloaded = tmp_store.load()
    assert reloaded.steps == trajectory.steps
    assert reloaded.chance == trajectory.chance, (
        "the recorded chance sequence is what makes a transition test a "
        "measurement rather than a coin flip; losing it on save corrupts "
        "every accuracy number read afterwards"
    )

    # Tests from the history, and the report an operator would read.
    tests = generate(reloaded, Tolerance(), include_hidden=False)
    assert tests, "settled history must yield executable tests"
    subject = ReferenceWorldModel(ref_config)
    train, held_out = split(tests, train_fraction=0.7, seed=7)
    train_rate = _pass_rate(train, subject)
    test_rate = _pass_rate(held_out, subject)
    report = ModelQualityReport(
        transition_accuracy_train=train_rate,
        transition_accuracy_test=test_rate,
        inference_accuracy_train=train_rate,
        inference_accuracy_test=test_rate,
        passed_tests=len(train) + len(held_out),
        total_tests=len(train) + len(held_out),
    )
    assert report.transition_accuracy == 1.0, (
        f"the reference model must score 1.0 against its own history through "
        f"the report surface (got {report.transition_accuracy})"
    )

    # Search, not prompting: ISMCTS for the day's decisions.
    planner = ISMCTS(
        ISMCTSConfig(seed=11, simulations=100, rollout_depth=6, rollouts_per_leaf=4)
    )
    plan = _plan_days(model, planner, random.Random(11), days=2, sims=100)
    assert plan, "the plan must contain at least one decision"
    publishes = [move for move in plan if isinstance(move, Publish)]
    assert publishes, "a plan that ships nothing exercises nothing downstream"

    # EVERY planned move must survive the same legality engine the publisher
    # will apply, judged against the context built from the store.
    engine = LegalityEngine(config.policy)
    ctx = _legality_context(tmp_store, config, _TODAY)
    for move in plan:
        verdict = engine.check(move, ctx)
        assert verdict.legal, (
            f"a planned move was refused by the gate that publish applies: "
            f"{verdict.rule}: {verdict.reason} ({verdict.source})"
        )

    # A day's output allocated like a Colonel Blotto force.
    fronts = [
        Front("x", _ANGLES[0], "solo_founder", 10.0),
        Front("tiktok", _ANGLES[1], "solo_founder", 8.0),
        Front("linkedin", _ANGLES[2], "agency_ops", 6.0),
    ]
    allocator = BlottoAllocator(BlottoConfig(units=6, seed=7))
    mixture = allocator.equilibrium_mixture(fronts, samples=60)
    assert mixture, "the allocator must return a mixture, not nothing"
    day_allocation = allocator.sample(mixture, random.Random(42))
    assert sum(day_allocation.values()) == 6, "the budget is spent exactly"

    # Briefs, then publish through the dry-run adapter.
    briefs = [_brief_for(move, config.destination_url) for move in publishes]
    receipts = [
        dry_adapter.publish(move, brief)
        for move, brief in zip(publishes, briefs, strict=True)
    ]
    assert len(dry_adapter.calls) == len(publishes), "every intent was recorded"
    assert all(receipt.dry_run for receipt in receipts)
    assert all(
        f"utm_content={move.utm_content}" in receipt.url
        for receipt, move in zip(receipts, publishes, strict=True)
    ), "the receipt is the record of which join key the post carries"
    assert not [name for name in sys.modules if name.split(".")[0] == "composio"], (
        "a dry run must never load the platform SDK"
    )

    # Record every move, ingest back, and re-plan on the enlarged history.
    moves_before = tmp_store.stats().moves
    for move in plan:
        tmp_store.append_move(move, _TODAY.isoformat())
    pending_utms = [move.utm_content for move in publishes]
    platforms = {move.utm_content: move.platform for move in publishes}
    observations = dry_adapter.fetch_analytics(
        pending_utms, datetime.now() - timedelta(days=7), platforms=platforms
    )
    assert observations == [], "a dry run observes nothing and invents nothing"
    for observation in observations:
        tmp_store.attach_observation(observation)
    tmp_store.mark_settled(datetime.now())
    assert tmp_store.stats().moves == moves_before + len(plan), (
        "the store must grow by exactly the number of moves published"
    )

    replan = planner.plan(model, model.initial_state(), OPERATOR, 40)
    assert replan.move in model.get_legal_actions(model.initial_state()), (
        "the second plan must be a real, legal decision"
    )


# ---------------------------------------------------------------------------
# Scenario 2: the measurement canary, through the report surface.
# ---------------------------------------------------------------------------


def test_reference_model_scores_one_through_the_report_surface(
    reference_trajectory: Callable[..., Trajectory],
) -> None:
    """The instrument must read 1.00 on a known-good model, ON THE REPORT.

    This deliberately duplicates the unit-level canary in ``test_cwm.py``
    (``test_reference_model_scores_one_against_its_own_trajectories``), and
    the duplication is the point: a canary that only exists at the unit
    level does not protect the reporting path. ``ModelQualityReport`` is
    the surface an operator actually reads before trusting a plan, and
    nothing in the unit suite forces the numbers computed from a split to
    arrive in the table unchanged. A report that averaged the splits,
    swapped train for test, or formatted 0.78 as 78 would pass every
    existing unit test and mislead every human who read it.
    """
    horizon = 20
    seed = 7
    trajectory = reference_trajectory(horizon=horizon, seed=seed)
    tests = generate(trajectory, Tolerance(), include_hidden=False)
    assert len(tests) >= 4, "a twenty-day history must yield a splittable suite"

    train, held_out = split(tests, train_fraction=0.7, seed=99)
    assert train and held_out, "both splits must be non-empty to be a split"
    subject = ReferenceWorldModel(ReferenceConfig(horizon=horizon, seed=seed))

    train_rate = _pass_rate(train, subject)
    test_rate = _pass_rate(held_out, subject)
    # The online split is self-play under the model's own policy, recorded
    # and tested like any other history -- the same shape ``fronts accuracy``
    # produces.
    online_trajectory = _play_self_play_episode(subject, steps=30, seed=3)
    online_rate = _pass_rate(
        generate(online_trajectory, Tolerance(), include_hidden=False), subject
    )

    report = ModelQualityReport(
        transition_accuracy_train=train_rate,
        transition_accuracy_test=test_rate,
        transition_accuracy_online=online_rate,
        inference_accuracy_train=train_rate,
        inference_accuracy_test=test_rate,
        inference_accuracy_online=online_rate,
        passed_tests=len(tests),
        total_tests=len(tests),
    )
    assert train_rate == 1.0, f"train split must be exactly 1.0, got {train_rate}"
    assert test_rate == 1.0, f"held-out split must be exactly 1.0, got {test_rate}"
    assert online_rate == 1.0, f"online split must be exactly 1.0, got {online_rate}"
    assert report.transition_accuracy == 1.0

    table = report.format_table()
    for split_name in ("train", "test", "online"):
        assert split_name in table, f"the report must render the {split_name} split"
    assert "1.00" in table, "a perfect model must render as 1.00, not 1 or 100"
    assert "transition" in table and "inference" in table


def _play_self_play_episode(
    model: CodeWorldModel, steps: int, seed: int
) -> Trajectory:
    """A self-play episode through the CLI's own recorder.

    Reuses ``fronts.cli._play_trajectory`` rather than reimplementing the
    observation-recording convention: the online split is only comparable
    to the operator's ``fronts accuracy`` output if it is produced the same
    way.
    """
    rng = random.Random(seed)
    trajectory = _play_trajectory(model, steps=steps, rng=rng)
    assert trajectory.steps, "self-play must record moves"
    return trajectory


# ---------------------------------------------------------------------------
# Scenario 3: refusal is visible and exits 2.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "move",
    [Scale("launch_theater", 1.10), Kill("launch_theater", "no_signal")],
    ids=["scale", "kill"],
)
def test_legality_refusal_surfaces_rule_reason_and_source(
    move: Move, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Below the attribution floor an angle is unjudgeable and BOTH Scale and
    Kill are refused -- and the refusal must be impossible to miss or
    mis-script.

    The bug class is the quiet refusal: an exit code that collides with
    "generic error" teaches an operator's wrapper to retry a move the system
    just told them not to make, and a refusal without its citation leaves
    "says who?" unanswered for an operator eating a day of silence. So the
    exit code must be exactly 2, and stdout must carry the rule, the reason,
    and a source naming the corpus file the rule came from.
    """
    config = _write_config(tmp_path, measurements=False)
    store = TrajectoryStore(tmp_path / "history.jsonl")
    store.append_move(_publish(), posted_at=(_TODAY - timedelta(days=5)).isoformat())
    store.attach_observation(
        _settled_observation("c0412ab9", conversions=400)
    )
    _write_plan(tmp_path, [move])
    history_before = (tmp_path / "history.jsonl").read_bytes()

    code = app(["publish", "--config", str(config)])
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert code == 2, (
        f"refused-by-legality is its own exit code; got {code} with output: "
        f"{output!r}"
    )
    assert "REFUSED" in output
    assert "ATTRIBUTION_COVERAGE" in output, "the rule must be named"
    assert "unjudgeable" in output, "the reason must say why in words"
    source = _refusal_source(output)
    assert source, "the refusal must cite a source"
    assert source.endswith(".md"), f"the source must name a corpus file: {source!r}"
    assert "09_Attribution_and_Analytics" in source
    assert (tmp_path / "history.jsonl").read_bytes() == history_before, (
        "a refused plan ships NOTHING -- not even the legal-looking parts"
    )


def test_untracked_publish_is_refused_with_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A Publish with empty utm_content is refused, also exit 2.

    Untracked output can never produce an observation, so it poisons every
    downstream estimate by being invisible where its consequences land.
    This is a system invariant rather than a corpus finding -- and it is
    only reachable through the CLI because the action codec round-trips
    the empty utm that encode emits. A decode error here (exit 1, "non-
    canonical action keys") would bury the cited refusal behind a generic
    one, which is exactly the quiet-refusal bug this scenario exists to
    catch.
    """
    config = _write_config(tmp_path)
    _write_plan(tmp_path, [_publish(utm_content="")])

    code = app(["publish", "--config", str(config)])
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert code == 2, f"an untracked publish must exit 2, got {code}: {output!r}"
    assert "TRACKED_OUTPUT" in output
    assert "utm_content" in output, "the reason must name the missing join key"
    assert _refusal_source(output), "even a system invariant cites its contract"


# ---------------------------------------------------------------------------
# Scenario 4: cold start is reported honestly.
# ---------------------------------------------------------------------------


def _settled_history(count: int) -> Trajectory:
    """``count`` settled posts across two archetypes and two platforms --
    the floor's shape, with only the count varying."""
    trajectory = Trajectory(account="cold-start")
    for index in range(count):
        platform = (Platform.X, Platform.LINKEDIN)[index % 2]
        archetype = (Archetype.FOUNDER_TRAUMA, Archetype.UGC_REVIEW)[index % 2]
        utm = f"{index:08x}"
        trajectory.steps.append(
            Step(
                move=_publish(
                    utm_content=utm, platform=platform, archetype=archetype
                ),
                observation=_settled_observation(utm, conversions=12),
            )
        )
    return trajectory


def test_cold_start_floor_is_reported_not_silently_passed(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Below the floor the system must SAY SO, and at the floor it must
    clear -- both directions, because a one-sided check cannot tell a
    boundary from a slope.

    The failure mode this catches is optimism encoded as an off-by-one: a
    floor that reports cleared at 29 settled items invites synthesis to fit
    noise ("a model that fits your history and predicts nothing"), and a
    floor that never clears blocks a ready operator. The stats gate and the
    words the CLI prints must agree, because the operator reads the words.
    """
    config = _write_config(tmp_path)
    store = TrajectoryStore(tmp_path / "history.jsonl")
    below = COLD_START_MIN_SETTLED - 1

    store.save(_settled_history(below))
    stats = store.stats()
    assert stats.settled_observations == below
    assert stats.archetypes >= 2 and stats.platforms >= 2, (
        "the fixture must vary everything EXCEPT the count, or the boundary "
        "below tests the wrong gate"
    )
    assert stats.ready_for_synthesis is False, (
        f"{below} settled items must not clear a floor of {COLD_START_MIN_SETTLED}"
    )
    assert app(["stats", "--config", str(config)]) == 0
    words_below = capsys.readouterr().out
    assert "NOT cleared" in words_below, "the CLI must say it in words"
    assert str(COLD_START_MIN_SETTLED) in words_below

    # Grow the store by exactly one settled item, the live way: the move is
    # recorded first, its observation arrives later and joins on utm.
    utm = f"{below:08x}"
    store.append_move(
        _publish(
            utm_content=utm,
            platform=Platform.LINKEDIN,
            archetype=Archetype.UGC_REVIEW,
        ),
        posted_at=(_TODAY - timedelta(days=5)).isoformat(),
    )
    store.attach_observation(_settled_observation(utm, conversions=12))

    grown = store.stats()
    assert grown.settled_observations == COLD_START_MIN_SETTLED
    assert grown.ready_for_synthesis is True, (
        "exactly at the floor the store must be ready"
    )
    assert app(["stats", "--config", str(config)]) == 0
    words_at = capsys.readouterr().out
    assert "CLEARED" in words_at
    assert "NOT cleared" not in words_at


# ---------------------------------------------------------------------------
# Scenario 5: a bad model is visible as a bad model.
# ---------------------------------------------------------------------------


def test_a_failing_model_scores_low_and_is_rejected_by_the_arena(
    reference_trajectory: Callable[..., Trajectory],
) -> None:
    """A wrong model must score badly and lose, not crash the reporter.

    This is the paper's Gin-rummy case: the system's whole value is being
    able to SHOW a bad result (0.78 train, an agent that then loses badly)
    rather than routing around it. The bug class is the reporter that
    raises on a model whose numbers are far off, or the arena that crashes
    on an agent whose model disagrees with the host's action space -- either
    one converts a measurable failure into an outage, and an outage gets
    "fixed" by deleting the measurement.

    ``V1_SOURCE`` is a hand-authored wrong model: it satisfies the protocol
    and loads in the sandbox, but its observations (reach decaying from a
    flat 900) are far outside tolerance of what the reference model actually
    produced. The arena's bad agent is the one that plans inside V1 -- on
    the reference host its action simply does not exist, which is what an
    agent built on a wrong model looks like from the outside.
    """
    trajectory = reference_trajectory(horizon=12)
    tests = generate(trajectory, Tolerance(), include_hidden=False)
    namespace = Sandbox().load(V1_SOURCE, SandboxConfig())
    bad_model = instantiate(namespace, "WorldModel")

    results = [test.run(bad_model) for test in tests]
    pass_rate = sum(1 for result in results if result.passed) / len(results)
    assert 0.0 <= pass_rate < 0.5, (
        f"a wrong model must score low, not crash and not pass: {pass_rate}"
    )
    assert any(not result.passed for result in results)

    report = ModelQualityReport(
        transition_accuracy_train=pass_rate,
        transition_accuracy_test=pass_rate,
        inference_accuracy_train=pass_rate,
        inference_accuracy_test=pass_rate,
        passed_tests=sum(1 for result in results if result.passed),
        total_tests=len(results),
    )
    assert report.transition_accuracy < 0.5
    assert f"{pass_rate:.2f}" in report.format_table(), (
        "the report must render the low number, not an average or an error"
    )

    def bad_model_agent(host: CodeWorldModel, state: State) -> ActionKey:
        """The agent that trusts the wrong model: it asks V1 what to do and
        does that, whatever world it is actually playing in."""
        return _first_publish(bad_model, state)

    reference_host = ReferenceWorldModel(ReferenceConfig(horizon=10, seed=7))
    arena = arena_run(
        [reference_host, bad_model],
        [_first_publish, bad_model_agent],
        ArenaConfig(matches_per_pairing=3),
    )
    assert 1 in arena.rejected, (
        f"the agent built on the bad model must be rejected: {arena!r}"
    )
    assert 0 in arena.survivors, "the good agent must survive"
    assert set(arena.survivors) | set(arena.rejected) == {0, 1}


# ---------------------------------------------------------------------------
# Scenario 6: hostile synthesis output degrades, never crashes.
# ---------------------------------------------------------------------------

# The nine known module-graph escapes (see SECURITY.md and the matching
# regressions in test_cwm.py). Each executed successfully against the first
# version of the sandbox; the fix was to remove module objects from the
# namespace entirely.
_MODULE_GRAPH_ESCAPES: tuple[tuple[str, str], ...] = (
    ("random._os", "import random\nX = random._os.getcwd()\n"),
    ("from-random-import-_os", "from random import _os\nX = _os.getpid()\n"),
    ("typing.sys.modules", "import typing\nX = typing.sys.modules['os']\n"),
    (
        "dataclasses.builtins.open",
        "import dataclasses\nX = dataclasses.builtins.open('/etc/hostname').read()\n",
    ),
    ("collections._sys", "import collections\nX = collections._sys.modules['os']\n"),
    (
        "json.codecs",
        "import json\nX = json.codecs.open('/etc/hostname').read()\n",
    ),
    (
        "statistics.random",
        "import statistics\nX = statistics.random._os.getpid()\n",
    ),
    ("re.enum", "import re\nX = re.enum\n"),
    ("datetime.sys", "import datetime\nX = datetime.sys.executable\n"),
)


@pytest.mark.parametrize(
    "source", [source for _, source in _MODULE_GRAPH_ESCAPES],
    ids=[name for name, _ in _MODULE_GRAPH_ESCAPES],
)
def test_sandbox_escape_attempts_fail_closed_and_the_planner_still_plans(
    source: str,
    recorded_client: Callable[[dict[tuple[str, str], str]], RecordedClient],
    tmp_path: Path,
) -> None:
    """Hostile synthesis output must take the loop down a notch, not down.

    The escape arrives the way a real one would: inside a model response,
    replayed by a ``RecordedClient``, through the genuine prompt-build ->
    extract -> sandbox path. Fail-closed means three things at once here:
    synthesis reports a candidate that failed to load instead of raising;
    nothing is written outside the temp dir (the escapes never execute at
    all -- the sandbox refuses at compile time); and a planner handed the
    resulting ``FallbackInference`` still returns a LEGAL move. That last
    one is the operational point: a compromised or hallucinating provider
    must degrade the system to weaker play, never to an exception in the
    planning loop.
    """
    trajectory = make_trajectory()
    tests = [ModelTest(name="no_crash", kind=NO_CRASH, payload={"steps": 8, "seed": 3})]
    system, user = build_prompt(RULES_TEXT, [trajectory], None, None)
    client = recorded_client({(system, user): fenced(source)})

    before_cwd = sorted(entry.name for entry in Path.cwd().iterdir())
    before_tmp = sorted(entry.name for entry in tmp_path.iterdir())

    result = synthesise(client, SynthConfig(), RULES_TEXT, [trajectory], tests)

    assert result.model is None, "a hostile candidate must not load"
    assert result.error is not None and "SandboxViolation" in result.error, (
        f"the failure must be the sandbox's own refusal: {result.error!r}"
    )
    assert sorted(entry.name for entry in Path.cwd().iterdir()) == before_cwd, (
        "nothing may be written outside the temp dir"
    )
    assert sorted(entry.name for entry in tmp_path.iterdir()) == before_tmp

    # The degradation path: inference synthesis failed with the model, so
    # the planner determinizes with the fallback sampler.
    planning_model = ReferenceWorldModel(ReferenceConfig(horizon=6, seed=1))
    fallback = FallbackInference(model=planning_model)
    planner = ISMCTS(
        ISMCTSConfig(seed=3, simulations=20, rollout_depth=4, rollouts_per_leaf=2),
        inference=fallback,
    )
    state = planning_model.initial_state()
    planned = planner.plan(planning_model, state, OPERATOR, 20)
    legal = planning_model.get_legal_actions(state)
    assert planned.move in legal, (
        "fail-closed must degrade to weaker play, not to an exception or an "
        "illegal move"
    )


# ---------------------------------------------------------------------------
# Scenario 7: determinism across process boundaries.
# ---------------------------------------------------------------------------


def test_same_seed_produces_an_identical_plan_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    """Two separate processes, one seed, byte-identical JSON.

    This is the one scenario that must use subprocesses, and the reason is
    narrow but decisive: in-process determinism can hide a dependence on
    dict or set iteration order, because within one interpreter run a hash
    seed is fixed and iteration order is stable -- the plan reproduces
    perfectly and keeps reproducing until the day it ships from a cron job
    on another box. ``PYTHONHASHSEED`` is deliberately UNSET in the children
    so hash randomisation is live and the two runs differ in every hash the
    interpreter draws; identical output then means identical behaviour, not
    identical luck. A plan an operator cannot re-derive after it disappoints
    is a plan they cannot audit.
    """
    (tmp_path / "model.py").write_text(V2_SOURCE, encoding="utf-8")
    config = _write_config(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    env.pop("PYTHONHASHSEED", None)
    assert "PYTHONHASHSEED" not in env, "hash randomisation must be left live"

    argv = [
        sys.executable,
        "-m",
        "fronts.cli",
        "plan",
        "--config",
        str(config),
        "--seed",
        "7",
        "--sims",
        "40",
        "--days",
        "2",
        "--json",
    ]
    outputs: list[bytes] = []
    for _run in range(2):
        completed = subprocess.run(
            argv, capture_output=True, env=env, cwd=str(tmp_path), timeout=60
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        outputs.append(completed.stdout)

    assert outputs[0], "the plan must print JSON, not nothing"
    assert outputs[0] == outputs[1], (
        "the same seed in a fresh process must produce a byte-identical plan"
    )
    payload = json.loads(outputs[0].decode("utf-8"))
    assert payload["moves"], "the deterministic plan must contain moves"


# ---------------------------------------------------------------------------
# The synth -> plan handoff: the persisted inference sampler.
# ---------------------------------------------------------------------------


PLAN_SAMPLER_SOURCE = '''\
class StateInferenceSampler:
    def resample_state(self, obs_action_history, player_id):
        return {"day": 0, "phase": "operator", "posts": [], "last": None}
'''


def test_plan_determinizes_with_the_sampler_beside_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """``fronts synth`` writes ``inference.py`` beside the model; ``plan``
    must find it, name the determinization it used, and stay deterministic.
    Without the file the same plan runs open-loop -- and says that instead,
    because a planner that will not name its determinization is a planner
    whose confidence cannot be audited."""
    (tmp_path / "model.py").write_text(V2_SOURCE, encoding="utf-8")
    config = _write_config(tmp_path)
    argv = ["plan", "--config", str(config), "--seed", "7", "--sims", "40",
            "--days", "1"]

    assert app(argv) == 0
    assert "determinization: open-loop" in capsys.readouterr().out

    (tmp_path / "inference.py").write_text(PLAN_SAMPLER_SOURCE, encoding="utf-8")
    assert app(argv) == 0
    first = capsys.readouterr().out
    assert "determinization: synthesised sampler" in first
    assert app(argv) == 0
    assert first == capsys.readouterr().out, (
        "a deterministic sampler must leave the plan deterministic"
    )


def test_accuracy_scores_the_sampler_when_one_is_persisted(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """With no sampler on disk the inference column reads n/a; with one it
    carries measured numbers -- possibly damning ones, which is the point."""
    (tmp_path / "model.py").write_text(V2_SOURCE, encoding="utf-8")
    config = _write_config(tmp_path)
    TrajectoryStore(tmp_path / "history.jsonl").save(_settled_history(5))

    assert app(["accuracy", "--config", str(config), "--seed", "3"]) == 0
    without = capsys.readouterr().out
    assert "until a sampler is synthesised" in without

    (tmp_path / "inference.py").write_text(PLAN_SAMPLER_SOURCE, encoding="utf-8")
    assert app(["accuracy", "--config", str(config), "--seed", "3"]) == 0
    with_sampler = capsys.readouterr().out
    assert "autoencoder pass rate" in with_sampler
    assert with_sampler.count("n/a") == 1, (
        "train and online carry measurements; only the test split stays n/a"
    )

    # A sampler that raises on every call must read as a bad score, never a
    # stack trace -- this crashed with a traceback before the exception was
    # counted as a miss inside inference_accuracy.
    (tmp_path / "inference.py").write_text(
        "class StateInferenceSampler:\n"
        "    def resample_state(self, obs_action_history, player_id):\n"
        "        return 1 // 0\n",
        encoding="utf-8",
    )
    assert app(["accuracy", "--config", str(config), "--seed", "3"]) == 0
    hostile = capsys.readouterr().out
    assert "0.00" in hostile, "the hostile sampler scores zero, visibly"


# ---------------------------------------------------------------------------
# Scenario 8: the whole loop with the network physically unavailable.
# ---------------------------------------------------------------------------


def test_golden_path_runs_with_sockets_blocked_and_no_credentials(
    reference_trajectory: Callable[..., Trajectory],
    tmp_store: TrajectoryStore,
    dry_adapter: DryRunAdapter,
    no_network: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The condensed golden path, with the network physically gone.

    ``socket.socket`` itself raises, so any code path that silently reaches
    for the network -- a stray SDK import that dials home, an adapter that
    resolves a credential, a telemetry call nobody remembers adding -- fails
    this test at the exact call site with ``RuntimeError: network access
    attempted``, which is the whole point of blocking the socket rather
    than stubbing it. Every credential the product knows how to read is
    cleared, so the loop must also get by without falling back on a key.
    Offline is not a degraded mode here; it is the mode the loop must be
    correct in, because the box that matters has no network either.
    """
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COMPOSIO_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert not any(
        os.environ.get(key) for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                                        "COMPOSIO_API_KEY")
    ), "the loop must run with every credential absent"

    horizon = 10
    seed = 7
    ref_config = ReferenceConfig(horizon=horizon, seed=seed)
    model = ReferenceWorldModel(ref_config)
    trajectory = reference_trajectory(horizon=horizon, seed=seed)
    tmp_store.save(trajectory)
    assert tmp_store.stats().settled_observations >= 1
    assert tmp_store.load().chance == trajectory.chance

    tests = generate(tmp_store.load(), Tolerance(), include_hidden=False)
    rate = _pass_rate(tests, model)
    report = ModelQualityReport(
        transition_accuracy_train=rate,
        transition_accuracy_test=rate,
        inference_accuracy_train=rate,
        inference_accuracy_test=rate,
        passed_tests=len(tests),
        total_tests=len(tests),
    )
    assert report.transition_accuracy == 1.0

    planner = ISMCTS(
        ISMCTSConfig(seed=5, simulations=30, rollout_depth=5, rollouts_per_leaf=3)
    )
    plan = _plan_days(model, planner, random.Random(5), days=1, sims=30)
    assert plan
    publishes = [move for move in plan if isinstance(move, Publish)]
    assert publishes, "the condensed path must still ship something"

    engine = LegalityEngine()
    legality_ctx = _legality_context(
        tmp_store, default_config(), _TODAY
    )
    for move in plan:
        assert engine.check(move, legality_ctx).legal

    brief = _brief_for(publishes[0], "https://owles.works/pricing")
    receipt = dry_adapter.publish(publishes[0], brief)
    assert receipt.dry_run and dry_adapter.calls

    for move in plan:
        tmp_store.append_move(move, _TODAY.isoformat())
    observations = dry_adapter.fetch_analytics(
        [move.utm_content for move in publishes],
        datetime.now() - timedelta(days=7),
        platforms={move.utm_content: move.platform for move in publishes},
    )
    assert observations == []
    tmp_store.mark_settled(datetime.now())
    replan = planner.plan(model, model.initial_state(), OPERATOR, 30)
    assert replan.move in model.get_legal_actions(model.initial_state())
