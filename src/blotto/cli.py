"""The ``blotto`` command line: the operating loop from docs/OPERATING.md.

argparse from the standard library, and nothing else. A CLI that needs a
wheel installed before it will say anything is a CLI nobody runs on the box
that matters, and the core's zero-dependency guarantee is worth exactly as
much as its least necessary import.

Three conventions carry the whole file:

* Exit codes are a contract: 0 success, 1 error, 2 REFUSED-BY-LEGALITY.
  Code 2 is distinct because a refusal is not a failure -- it is the system
  working. An operator scripting the loop needs to distinguish "retry" from
  "stop and look at what you were about to post", and a refusal that exits
  1 teaches their wrapper to retry it.
* ``publish`` is a dry run unless ``--live`` is passed. Real posts are
  irreversible and outward-facing; every other flag combination may be
  tried freely, but the one that speaks to the internet has to be asked
  for by name.
* Statistics are part of the output, not a debug flag. ``plan`` prints
  visit counts and value estimates for the top actions because an operator
  asked to spend a day's output on a plan is owed the numbers behind it
  (see ``PlanResult`` in ``blotto.protocols``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from blotto.adapters.composio_io import (
    ComposioAdapter,
    DryRunAdapter,
    MissingMeasurementError,
)
from blotto.adapters.trajectory import TrajectoryStore
from blotto.config import BlottoConfig, default_config
from blotto.config import load as load_config
from blotto.cwm.arena import ArenaConfig
from blotto.cwm.arena import run as arena_run
from blotto.cwm.llm import AnthropicClient, OpenAIClient
from blotto.cwm.reference import ReferenceConfig, ReferenceWorldModel
from blotto.cwm.refine import RefineConfig, RefinementNode, RefinementTree
from blotto.cwm.refine import best as best_node
from blotto.cwm.refine import refine as refine_tree
from blotto.cwm.report import ModelQualityReport
from blotto.cwm.sandbox import Sandbox, SandboxConfig, instantiate
from blotto.cwm.synth import WORLD_MODEL_CLASS, SynthConfig, synthesise
from blotto.cwm.tests_from_traj import (
    OBSERVATION_RECONSTRUCTION,
    ModelTest,
    Tolerance,
    generate,
    split,
)
from blotto.game.action_space import ActionCodec, ActionDecodeError, utm_content_id
from blotto.game.legality import LegalityContext, LegalityEngine
from blotto.game.types import (
    CHANCE_PLAYER,
    OPERATOR,
    TERMINAL_PLAYER,
    ActionKey,
    Move,
    Observation,
    Platform,
    Publish,
    State,
    Step,
    Trajectory,
)
from blotto.solvers.ismcts import ISMCTS, ISMCTSConfig

__all__ = ["app"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2

_CODEC = ActionCodec()


# ---------------------------------------------------------------------------
# Parser. Global flags (--config/--seed/--json) are declared once on a parent
# parser shared by the main parser and every subparser, with SUPPRESS
# defaults: a subparser's own default must not overwrite a value the main
# parser already parsed off the command line, and with SUPPRESS an absent
# flag simply never sets the attribute. Handlers read them through
# getattr(args, name, default) accordingly. Both placements work:
# `blotto --seed 5 plan` and `blotto plan --seed 5`.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    globals_parent = argparse.ArgumentParser(add_help=False)
    globals_parent.add_argument(
        "--config", default=argparse.SUPPRESS, metavar="PATH",
        help="config file (default: built-in defaults)",
    )
    globals_parent.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS, metavar="N",
        help="seed every random draw the command makes",
    )
    globals_parent.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="emit machine-readable JSON where a command prints prose",
    )

    parser = argparse.ArgumentParser(
        prog="blotto",
        description="Plan content distribution as an imperfect-information game.",
        parents=[globals_parent],
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    synth = sub.add_parser(
        "synth", parents=[globals_parent],
        help="re-synthesise the world model from updated history",
    )
    synth.add_argument("--history", metavar="PATH", help="trajectory JSONL to learn from")
    synth.add_argument("--rules", metavar="PATH", help="platform policy in prose")
    synth.add_argument("--out", metavar="PATH", help="where to write the model source")
    synth.add_argument("--max-calls", type=int, default=500, metavar="N",
                       help="LLM call budget (default: 500, the paper's)")

    accuracy = sub.add_parser(
        "accuracy", parents=[globals_parent],
        help="train / test / online quality table for the current model",
    )
    accuracy.add_argument("--model", metavar="PATH", help="model source to score")

    plan = sub.add_parser(
        "plan", parents=[globals_parent],
        help="ISMCTS plan for the next N days, with per-action statistics",
    )
    plan.add_argument("--days", type=int, default=1, metavar="N",
                      help="days to plan (default: 1)")
    plan.add_argument("--sims", type=int, default=200, metavar="N",
                      help="ISMCTS simulations per decision (default: 200)")

    sub.add_parser(
        "arena", parents=[globals_parent],
        help="strategies play inside the models; bad ones are rejected",
    )

    ingest = sub.add_parser(
        "ingest", parents=[globals_parent],
        help="pull settled metrics and mark anything inside the lag partial",
    )
    ingest.add_argument("--since", metavar="DATE",
                        help="only pull observations posted on/after this date")

    sub.add_parser(
        "brief", parents=[globals_parent],
        help="emit the plan as content briefs, in JSON",
    )

    publish = sub.add_parser(
        "publish", parents=[globals_parent],
        help="ship the plan via Composio (dry run unless --live)",
    )
    publish.add_argument("--live", action="store_true",
                         help="actually post; without this, nothing leaves the machine")

    sub.add_parser(
        "stats", parents=[globals_parent],
        help="trajectory stats and cold-start readiness",
    )
    return parser


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> BlottoConfig:
    path = getattr(args, "config", None)
    if path is None:
        return default_config()
    return load_config(path)


def _seed_from_args(args: argparse.Namespace) -> int:
    seed = getattr(args, "seed", None)
    return seed if seed is not None else 0


def _rng(args: argparse.Namespace) -> random.Random:
    """A seeded RNG even when --seed was not passed: an unseeded plan is
    unreproducible, and an unreproducible plan cannot be audited after it
    disappoints."""
    return random.Random(_seed_from_args(args))


def _load_model_source(path: Path) -> Any:
    """Sandbox-load a world model from source text.

    The sandbox refuses imports and dunder access by design (see its module
    docstring for why an allowlist could not hold); a model file that will
    not load is an error with the sandbox's own message, not a stack trace
    from this command.
    """
    source = Path(path).read_text(encoding="utf-8")
    namespace = Sandbox().load(source, SandboxConfig())
    return instantiate(namespace, WORLD_MODEL_CLASS)


def _sample_chance(model: Any, state: State, rng: random.Random) -> ActionKey:
    outcomes = model.chance_outcomes(state)
    return rng.choices(
        [key for key, _ in outcomes],
        weights=[probability for _, probability in outcomes],
        k=1,
    )[0]


def _play_trajectory(
    model: Any,
    steps: int,
    rng: random.Random,
) -> Trajectory:
    """Play a random-legal episode of ``model`` and record it as a
    Trajectory, for the online split of ``accuracy``.

    Observations are collected from the OPERATOR's (degraded) view after
    each chance ply, keyed by utm -- the same convention
    ``ReferenceWorldModel.generate_trajectory`` uses, so test generation
    sees one shape regardless of who recorded the history. Chance outcomes
    are recorded, not discarded: replaying a fresh draw would charge the
    model for the dice rather than for itself (``Trajectory.chance``).
    """
    trajectory = Trajectory(account="online")
    state = model.initial_state()
    moves: list[Move] = []
    seen: dict[str, Observation] = {}
    while model.get_current_player(state) != TERMINAL_PLAYER and len(moves) < steps:
        player = model.get_current_player(state)
        if player == CHANCE_PLAYER:
            action = _sample_chance(model, state, rng)
            trajectory.chance.append(action)
            state = model.apply_action(state, action)
            observation = model.get_observations(state).get(OPERATOR)
            if observation is not None and observation.utm_content:
                seen[observation.utm_content] = observation
            continue
        legal = model.get_legal_actions(state)
        if not legal:
            break
        action = rng.choice(legal)
        move = _CODEC.decode(action)
        if isinstance(move, Publish):
            # Re-stamp with a per-step unique id, exactly as
            # ``ReferenceWorldModel.generate_trajectory`` does: the legal set
            # is a fixed catalogue, so a random re-pick publishes two distinct
            # posts under one utm -- and utm is the join key between a move
            # and its observation. Without this the online split of
            # ``accuracy`` scored the REFERENCE model below 1.0, charging it
            # for a bookkeeping collision (see the re-stamp in
            # ``generate_trajectory`` for the full failure mode).
            move = replace(
                move, utm_content=utm_content_id(move, salt=str(len(moves)))
            )
            action = _CODEC.encode(move)
        moves.append(move)
        state = model.apply_action(state, action)
    for move in moves:
        observation = None
        if isinstance(move, Publish):
            observation = seen.get(move.utm_content)
        trajectory.steps.append(Step(move=move, observation=observation))
    return trajectory


def _legality_context(
    store: TrajectoryStore,
    config: BlottoConfig,
    today: date,
) -> LegalityContext:
    """Build the legality context from what the store and config actually
    know, defaulting every unknown to the CONSERVATIVE reading.

    Approximations, stated so nobody has to infer them from behaviour:
    spend and allocation history are not recorded in the trajectory store,
    so they arrive empty (the drawdown and velocity rules see no evidence
    and do not fire -- the gates that depend on recorded evidence are the
    ones this context can feed honestly). Coverage comes from the operator's
    measurement or, when unmeasured, 0.0: unmeasured means unjudgeable,
    which is the reading the coverage rule itself mandates and the reason
    an unconfigured system refuses to scale anything rather than nothing.
    """
    trajectory = store.load()
    ctx = LegalityContext(current_date=today)
    first_seen: dict[str, date] = {}
    settled: list[tuple[Observation, str]] = []
    for step in trajectory.steps:
        if not isinstance(step.move, Publish):
            continue
        if step.observation is None:
            continue
        posted = date.fromisoformat(step.observation.posted_at)
        first_seen.setdefault(step.move.angle, posted)
        if step.observation.is_partial:
            continue
        settled.append((step.observation, step.move.angle))
        ctx.angle_reach[step.move.angle] = (
            ctx.angle_reach.get(step.move.angle, 0) + step.observation.reach
        )
        ctx.angle_total_conversions[step.move.angle] = (
            ctx.angle_total_conversions.get(step.move.angle, 0)
            + step.observation.attributed_conversions
        )
        if config.attribution_coverage is not None:
            ctx.angle_attribution_coverage[step.move.angle] = (
                config.attribution_coverage
            )
    for window_days, target in (
        (7, ctx.settled_conversions_7d),
        (14, ctx.settled_conversions_14d),
    ):
        cutoff = today - timedelta(days=window_days)
        for observation, angle in settled:
            if date.fromisoformat(observation.posted_at) >= cutoff:
                target[angle] = target.get(angle, 0) + observation.attributed_conversions
    for angle, posted in first_seen.items():
        ctx.angle_age_days[angle] = (today - posted).days
    return ctx


def _print_refusal(move: Move, verdict: Any) -> None:
    """Print the refusal an operator is owed: the rule, the reason, and the
    source it cites. The source is not decoration -- an operator asked to
    eat a day of silence is entitled to answer 'says who?' (``Verdict``)."""
    print("REFUSED -- publishing aborted; one move in the plan is illegal:")
    print(f"  move:   {_CODEC.encode(move)}")
    print(f"  rule:   {verdict.rule}")
    print(f"  reason: {verdict.reason}")
    print(f"  source: {verdict.source}")
    print(
        "  (docs/OPERATING.md: a planner that asks for an illegal move is a "
        "bug in the planner; you want this loud, not shipped.)"
    )


def _read_plan(config: BlottoConfig) -> list[Move]:
    path = config.paths.plan
    if not path.exists():
        raise FileNotFoundError(
            f"no plan at {path}; run `blotto plan` first (or set paths.plan "
            "in your config)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return [_CODEC.decode(key) for key in payload["moves"]]
    except ActionDecodeError as exc:
        # A world model emitting non-canonical action keys is a model bug,
        # and naming the file turns a cryptic decode error into a fixable
        # one: re-run synth, or fix the model's action encoding.
        raise ValueError(
            f"{path}: the plan contains a move the codec cannot decode "
            f"({exc}); the world model that wrote it emits non-canonical "
            "action keys"
        ) from exc


def _brief_for(move: Publish, destination_url: str | None) -> dict[str, Any]:
    """One publish move as the JSON brief the creative pipeline consumes.

    The fields are exactly the creative dimensions of ``Publish`` plus the
    utm the finished asset must carry -- the brief is where the join key is
    communicated to a human, which is the one step no amount of automation
    replaces."""
    return {
        "platform": move.platform.value,
        "format": move.format.value,
        "archetype": move.archetype.value,
        "vector": move.vector.value,
        "semantic_tier": move.semantic_tier.value,
        "hook": move.hook.value,
        "avatar": move.avatar,
        "cta_mode": move.cta_mode.value,
        "claim_class": move.claim_class.value,
        "angle": move.angle,
        "utm_content": move.utm_content,
        "url": destination_url,
        "text": (
            f"{move.hook.value.replace('_', ' ').title()} for "
            f"{move.avatar.replace('_', ' ')}: {move.angle.replace('_', ' ')} "
            f"as a {move.format.value}."
        ),
    }


# ---------------------------------------------------------------------------
# Subcommand handlers. Each returns an exit code.
# ---------------------------------------------------------------------------


def _client_from_env() -> Any:
    """Build the LLM client from whichever provider key the environment
    holds. The order is arbitrary and stated: Anthropic first, because the
    synthesis prompts were written against it."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    raise RuntimeError(
        "synthesis needs a language model: export ANTHROPIC_API_KEY or "
        "OPENAI_API_KEY (pip install \"blotto[llm]\" for the SDKs)"
    )


def _cmd_synth(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    history_path = Path(args.history) if args.history else config.paths.history
    rules_path = Path(args.rules) if args.rules else config.paths.rules
    out_path = Path(args.out) if args.out else config.paths.model

    trajectory = TrajectoryStore(history_path).settled_only()
    if not trajectory.steps:
        print(
            f"no settled history at {history_path}; below the cold-start floor "
            "synthesis fits noise (docs/OPERATING.md). Record trajectories first.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    rules_text = rules_path.read_text(encoding="utf-8")
    # Closed deck: the operator's own logs carry no hidden state, so test
    # generation emits observation-reconstruction and no-crash tests only
    # (PAPER.md s3). Passing include_hidden=True here would emit transition
    # tests the recorded history cannot actually anchor.
    tests = generate(trajectory, Tolerance(), include_hidden=False)
    client = _client_from_env()

    initial = synthesise(
        client,
        SynthConfig(max_calls=args.max_calls),
        rules_text,
        [trajectory],
        tests[: SynthConfig().num_tests_on_init],
    )
    if initial.model is None:
        print(
            f"first candidate failed to load: {initial.error}", file=sys.stderr
        )
        return EXIT_ERROR
    tree = RefinementTree(
        nodes=[RefinementNode(initial.source, initial.pass_rate, model=initial.model)],
        rules_text=rules_text,
        trajectories=[trajectory],
    )
    tree = refine_tree(
        client,
        tree,
        tests,
        RefineConfig(num_retries=args.max_calls),
    )
    winner = best_node(tree)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(winner.source, encoding="utf-8")
    print(f"synthesised model written to {out_path}")
    print(f"  candidates:   {len(tree.nodes)}")
    print(f"  pass rate:    {winner.h:.2f} on {len(tests)} settled-history tests")
    print(f"  llm calls:    {1 + sum(node.refinements for node in tree.nodes)}")
    print("  next: `blotto accuracy` before trusting `blotto plan`")
    return EXIT_OK


def _cmd_accuracy(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    model_path = Path(args.model) if args.model else config.paths.model
    model = _load_model_source(model_path)
    trajectory = TrajectoryStore(config.paths.history).settled_only()
    if not trajectory.steps:
        print(
            f"no settled history at {config.paths.history}; nothing to score "
            "against",
            file=sys.stderr,
        )
        return EXIT_ERROR

    tests = generate(trajectory, Tolerance(), include_hidden=False)
    train, held_out = split(tests, train_fraction=0.7, seed=_seed_from_args(args))

    def pass_rate(suite: list[ModelTest]) -> tuple[float, int, int]:
        results = [test.run(model) for test in suite]
        passed = sum(1 for result in results if result.passed)
        return (passed / len(suite) if suite else 0.0), passed, len(suite)

    train_rate, train_passed, train_total = pass_rate(train)
    test_rate, test_passed, test_total = pass_rate(held_out)

    # The online split measures the model on states its own policy visited:
    # a self-play episode under the model, recorded and tested like any
    # other history. This is self-consistency under its own dynamics -- a
    # necessary condition, not the paper's full online protocol, which
    # needs the live loop. Printed as such, because a number that quietly
    # overstates what was measured is worse than a caveat.
    online_trajectory = _play_trajectory(model, steps=30, rng=_rng(args))
    online_tests = generate(online_trajectory, Tolerance(), include_hidden=False)
    online_rate, online_passed, online_total = pass_rate(online_tests)

    report = ModelQualityReport(
        transition_accuracy_train=train_rate,
        transition_accuracy_test=test_rate,
        transition_accuracy_online=online_rate,
        inference_accuracy_train=train_rate,
        inference_accuracy_test=test_rate,
        inference_accuracy_online=online_rate,
        llm_calls=0,
        passed_tests=train_passed + test_passed,
        total_tests=train_total + test_total,
    )
    print(f"model: {model_path}")
    print(report.format_table())
    print(
        "note: splits are pass rates on closed-deck "
        f"{OBSERVATION_RECONSTRUCTION} tests; the online split is self-play "
        "consistency, not the paper's live protocol."
    )
    return EXIT_OK


def _cmd_plan(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    model_path = config.paths.model
    if not model_path.exists():
        print(
            f"no world model at {model_path}; run `blotto synth` first. "
            "(A plan from no model is a guess with a table around it.)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    model = _load_model_source(model_path)
    rng = _rng(args)
    planner = ISMCTS(ISMCTSConfig(seed=_seed_from_args(args)))

    state = model.initial_state()
    moves: list[str] = []
    decisions: list[dict[str, Any]] = []
    for _day in range(max(1, args.days)):
        if model.get_current_player(state) == TERMINAL_PLAYER:
            break
        if model.get_current_player(state) == CHANCE_PLAYER:
            state = model.apply_action(state, _sample_chance(model, state, rng))
        if model.get_current_player(state) != OPERATOR:
            break
        result = planner.plan(model, state, OPERATOR, args.sims)
        ranked = sorted(
            result.visits.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
        top = [
            {
                "action": str(action),
                "visits": visits,
                "value": round(result.values.get(action, 0.0), 4),
            }
            for action, visits in ranked[:5]
        ]
        decisions.append({"move": str(result.move), "value": round(result.value, 4), "top": top})
        moves.append(str(result.move))
        state = model.apply_action(state, result.move)
        if model.get_current_player(state) == CHANCE_PLAYER:
            state = model.apply_action(state, _sample_chance(model, state, rng))

    plan_path = config.paths.plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"moves": moves, "decisions": decisions}, indent=2),
                         encoding="utf-8")

    if getattr(args, "json", False):
        print(json.dumps({"moves": moves, "decisions": decisions}, indent=2))
        return EXIT_OK
    print(f"plan for {len(moves)} decision(s) written to {plan_path}")
    for index, decision in enumerate(decisions):
        print(f"\ndecision {index + 1}: {decision['move']}")
        print(f"  {'action':<62} {'visits':>6} {'value':>14}")
        for entry in decision["top"]:
            print(
                f"  {entry['action'][:62]:<62} {entry['visits']:>6} "
                f"{entry['value']:>14,.1f}"
            )
    print(
        "\nvisit counts are the search's confidence: an action visited twice "
        "against its rival's two hundred is a hedge, not a recommendation."
    )
    return EXIT_OK


def _cmd_arena(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    rng = _rng(args)

    # The arena's real question is "which SYNTHESISED model is bad", and it
    # needs several candidates to ask it. With fewer on disk the tournament
    # still runs -- strategies compete inside what exists -- but the note
    # below says so, because a rejection between strategies is not a
    # rejection between models and reading it as one would be the exact
    # confident-direction error the arena exists to prevent.
    hosts: list[Any] = [ReferenceWorldModel(ReferenceConfig())]
    if config.paths.model.exists():
        hosts.append(_load_model_source(config.paths.model))

    def random_agent(model: Any, state: State) -> ActionKey:
        return rng.choice(model.get_legal_actions(state))

    def first_publish_agent(model: Any, state: State) -> ActionKey:
        legal = model.get_legal_actions(state)
        return next((key for key in legal if str(key).startswith("publish")), legal[0])

    def hold_agent(model: Any, state: State) -> ActionKey:
        legal = model.get_legal_actions(state)
        return next((key for key in legal if str(key) == "hold"), legal[0])

    agents: list[tuple[str, Callable[[Any, State], ActionKey]]] = [
        ("random", random_agent),
        ("first-publish", first_publish_agent),
        ("hold", hold_agent),
    ]
    result = arena_run([model for model in hosts], [agent for _, agent in agents],
                       ArenaConfig(matches_per_pairing=3))
    names = [name for name, _ in agents]
    print("arena: strategies play inside the host model(s); losers are rejected")
    hosts_note = (
        f", {config.paths.model}" if len(hosts) > 1 else " (no synthesised model found)"
    )
    print(f"  hosts: reference{hosts_note}")
    for index, name in enumerate(names):
        status = "SURVIVOR" if index in result.survivors else "REJECTED"
        print(f"  {name:<14} score {result.scores[index]:>10.2f}  {status}")
    print(f"  utility range observed: {result.utility_range:.2f}")
    print(f"  rejected: {[names[i] for i in result.rejected] or 'none'}")
    if len(hosts) == 1:
        print(
            "  note: one host means this compares strategies, not candidate "
            "models; run `blotto synth` and re-run to reject bad models."
        )
    return EXIT_OK


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    store = TrajectoryStore(config.paths.history)
    since = (
        date.fromisoformat(args.since)
        if args.since
        else date.today() - timedelta(days=7)
    )
    trajectory = store.load()
    pending: dict[str, Platform] = {}
    for step in trajectory.steps:
        if isinstance(step.move, Publish) and (
            step.observation is None or step.observation.is_partial
        ):
            pending[step.move.utm_content] = step.move.platform
    if not pending:
        print(f"nothing pending in {config.paths.history}; every post has settled data")
    else:
        # Ingest is a read, so it goes live whenever a credential resolves --
        # unlike publish, whose --live gate exists because posting is
        # irreversible. With no credential the dry adapter records the
        # intended reads and observes nothing.
        has_key = bool(config.composio.api_key or os.environ.get("COMPOSIO_API_KEY"))
        if has_key:
            adapter = ComposioAdapter(
                config.composio,
                attribution_coverage=config.attribution_coverage,
                incrementality=config.incrementality,
            )
        else:
            adapter = DryRunAdapter(
                attribution_coverage=config.attribution_coverage,
                incrementality=config.incrementality,
            )
            print("no COMPOSIO_API_KEY; recording intended reads (dry run)")
        observations = adapter.fetch_analytics(
            list(pending),
            datetime.fromisoformat(since.isoformat()),
            platforms=pending,
        )
        for observation in observations:
            store.attach_observation(observation)
        print(f"attached {len(observations)} observation(s) for {len(pending)} pending post(s)")
    flipped = store.mark_settled(datetime.now())
    print(f"marked {flipped} observation(s) settled (reporting lag cleared)")
    stats = store.stats()
    print(
        f"history: {stats.moves} moves, {stats.settled_observations} settled, "
        f"{stats.partial_observations} partial, {stats.unmatched_utm} unmatched"
    )
    if getattr(args, "json", False):
        print(json.dumps(asdict(stats)))
    return EXIT_OK


def _cmd_brief(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    moves = _read_plan(config)
    briefs = [
        _brief_for(move, config.destination_url)
        for move in moves
        if isinstance(move, Publish)
    ]
    payload = json.dumps(briefs, indent=2)
    briefs_path = config.paths.briefs
    briefs_path.parent.mkdir(parents=True, exist_ok=True)
    briefs_path.write_text(payload, encoding="utf-8")
    print(payload)
    print(f"\n{len(briefs)} brief(s) written to {briefs_path}", file=sys.stderr)
    return EXIT_OK


def _cmd_publish(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    moves = _read_plan(config)
    engine = LegalityEngine(config.policy)
    store = TrajectoryStore(config.paths.history)
    ctx = _legality_context(store, config, date.today())

    # Legality first, adapter second: a plan containing an illegal move is
    # aborted WHOLE rather than partially shipped. OPERATING.md is explicit
    # that publish refuses rather than warns, and shipping the legal half of
    # a plan whose other half was refused would punish the refusal by
    # teaching the operator that it only sometimes matters.
    for move in moves:
        verdict = engine.check(move, ctx)
        if not verdict.legal:
            _print_refusal(move, verdict)
            return EXIT_REFUSED

    publishes = [move for move in moves if isinstance(move, Publish)]
    if not publishes:
        print("plan contains no posts (scale/kill/hold only); nothing to publish")
        return EXIT_OK
    if config.destination_url is None:
        print(
            "no destination_url configured; set [publish] destination_url in "
            "your config before shipping links.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    if args.live:
        adapter: Any = ComposioAdapter(
            config.composio,
            attribution_coverage=config.attribution_coverage,
            incrementality=config.incrementality,
        )
    else:
        adapter = DryRunAdapter(
            attribution_coverage=config.attribution_coverage,
            incrementality=config.incrementality,
        )
    today = date.today().isoformat()
    receipts = []
    for move in publishes:
        receipt = adapter.publish(
            move, _brief_for(move, config.destination_url)
        )
        receipts.append(receipt)
    # Every move in the plan is recorded, posts and non-posts alike: the
    # trajectory is the operator's decision history, and a Scale whose
    # context later vanishes is a Scale the gates cannot judge.
    for move in moves:
        store.append_move(move, today)

    mode = "LIVE" if args.live else "DRY RUN (pass --live to ship)"
    print(f"publish: {mode}")
    for receipt in receipts:
        print(
            f"  {receipt.platform.value:<10} utm={receipt.utm_content} "
            f"action={receipt.action} -> {receipt.url}"
        )
    print(f"{len(receipts)} post(s), {len(moves)} move(s) recorded to {config.paths.history}")
    return EXIT_OK


def _cmd_stats(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    stats = TrajectoryStore(config.paths.history).stats()
    if getattr(args, "json", False):
        print(json.dumps(asdict(stats)))
        return EXIT_OK
    print(f"trajectory: {config.paths.history}")
    print(f"  moves:                 {stats.moves}")
    print(f"  settled observations:  {stats.settled_observations}")
    print(f"  partial observations:  {stats.partial_observations}")
    print(f"  unmatched utm ids:     {stats.unmatched_utm}")
    print(f"  archetypes covered:    {stats.archetypes}")
    print(f"  platforms covered:     {stats.platforms}")
    if stats.ready_for_synthesis:
        print("  cold start: CLEARED (>=30 settled across >=2 archetypes, >=2 platforms)")
    else:
        print(
            "  cold start: NOT cleared -- docs/OPERATING.md wants 30 settled "
            "items across at least two archetypes and two platforms before "
            "synthesis is worth its calls; use the model-free solvers until then."
        )
    return EXIT_OK


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "synth": _cmd_synth,
    "accuracy": _cmd_accuracy,
    "plan": _cmd_plan,
    "arena": _cmd_arena,
    "ingest": _cmd_ingest,
    "brief": _cmd_brief,
    "publish": _cmd_publish,
    "stats": _cmd_stats,
}


def app(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the exit code; the console script wraps it.

    Errors the operator can act on (missing files, missing keys, unmeasured
    attribution) print a message and exit 1; unexpected exceptions propagate
    as tracebacks, because a bug dressed up as a clean error message is a
    bug that never gets reported.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        return _HANDLERS[command](args)
    except MissingMeasurementError as exc:
        print(f"blotto: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"blotto: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(app())
