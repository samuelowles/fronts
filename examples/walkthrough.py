"""One runnable story: the whole loop, offline, in under thirty seconds.

No API keys, no network, no LLM. Everything below runs against the
hand-written ``ReferenceWorldModel`` standing in for a synthesised one --
which is exactly how the system behaves before `fronts synth` has anything
to learn from, and why the model-free solvers exist.

Read this top to bottom; each numbered section is one move in the argument.

    1. History exists before models do.
    2. The cold-start floor is checked, not vibes.
    3. The measuring instrument reads 1.00 on a known-good model.
    4. Search turns compute into a decision -- with the numbers shown.
    5. A day's output is allocated like a Colonel Blotto force.
    6. Angle selection adapts under drift, because the world is adversarial.
    7. A Scale on 12 conversions is IMPOSSIBLE, not discouraged. (Centre-
       piece: this is the moment that shows what the system is for.)
    8. The highest-benchmark archetype is the wrong one to crowd onto.

Run:  PYTHONPATH=src python examples/walkthrough.py
"""

from __future__ import annotations

import random
import tempfile
from datetime import date
from pathlib import Path

from fronts.adapters.trajectory import TrajectoryStore
from fronts.cwm.reference import ReferenceConfig, ReferenceWorldModel
from fronts.cwm.tests_from_traj import Tolerance, generate
from fronts.game.legality import LegalityContext, LegalityEngine, settled_count
from fronts.game.priors import ARCHETYPE_PRIORS
from fronts.game.types import Observation, Scale, Trajectory
from fronts.solvers.blotto import BlottoAllocator, BlottoConfig, Front
from fronts.solvers.congestion import CongestionConfig, crowding_adjusted_ranking
from fronts.solvers.exp3 import EXP3, EXP3Config
from fronts.solvers.ismcts import ISMCTS, ISMCTSConfig

ANGLE = "launch_theater"

POLICY_RNG = random.Random(13)
"""Seeded so the walkthrough prints the same numbers on every run -- the
README quotes them verbatim. The global ``random`` here would reseed from OS
entropy each run and drift the settled/partial counts."""


def banner(number: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"{number}. {title}")
    print("=" * 78)


def random_policy(model: ReferenceWorldModel, state: object) -> str:
    """A uniform-random legal policy. Legality is a PRECONDITION on
    planning, so even this dumb policy never proposes an illegal move --
    that is the point of section 7."""
    legal = model.get_legal_actions(state)  # type: ignore[attr-defined]
    return POLICY_RNG.choice(legal)


# ---------------------------------------------------------------------------
# 1. Generate a synthetic history from the reference model.
#
# WHY: the closed-deck setting has exactly one training signal -- the
# operator's own moves and the observations that came back. Everything
# downstream (tests, models, plans) is fit against history like this. We
# generate it rather than fetch it so the walkthrough runs with no keys.
# ---------------------------------------------------------------------------

banner(1, "History: the only training signal there is")

CONFIG = ReferenceConfig(horizon=30, seed=7)
RECORDER = ReferenceWorldModel(CONFIG)

trajectories: list[Trajectory] = []
for seed in (7, 21, 42):
    trajectory = RECORDER.generate_trajectory(
        random_policy, steps=30, rng=random.Random(seed)
    )
    trajectories.append(trajectory)
    publishes = sum(1 for step in trajectory.steps if step.observation is not None)
    print(
        f"  seed {seed:>2}: {len(trajectory.steps)} moves, "
        f"{publishes} with observations, "
        f"{len(trajectory.chance)} chance outcomes recorded"
    )

print(
    "\n  The observations are LATE (24-72h) and PARTLY FICTIONAL (coverage\n"
    "  below 1, incrementality below 1) because that is what a dashboard\n"
    "  actually returns. A model trained on clean numbers would be planning\n"
    "  for a world that does not exist."
)

# ---------------------------------------------------------------------------
# 2. Trajectory stats and the cold-start check.
#
# WHY: synthesis below the floor "will produce a model that fits your
# history and predicts nothing" (docs/OPERATING.md). The floor is 30
# settled items across >=2 archetypes and >=2 platforms, and the store
# checks it programmatically instead of leaving it to optimism.
# ---------------------------------------------------------------------------

banner(2, "Cold start: is there enough settled history to learn from?")

merged = Trajectory(account="walkthrough")
for trajectory in trajectories:
    merged.steps.extend(trajectory.steps)
    merged.chance.extend(trajectory.chance)

store_dir = Path(tempfile.mkdtemp(prefix="fronts-walkthrough-"))
store = TrajectoryStore(store_dir / "trajectories.jsonl")
store.save(merged)
stats = store.stats()
print(f"  store:             {store.path}")
print(f"  moves:             {stats.moves}")
print(f"  settled:           {stats.settled_observations}")
print(f"  partial (in lag):  {stats.partial_observations}")
print(f"  unmatched utm:     {stats.unmatched_utm}")
print(f"  archetypes:        {stats.archetypes}")
print(f"  platforms:         {stats.platforms}")
print(f"  ready_for_synthesis: {stats.ready_for_synthesis}")
assert stats.ready_for_synthesis, "demo history should clear the cold-start floor"

# The chance sequence must survive the round trip, or every transition test
# replayed from this store would re-roll the dice and score noise as error.
reloaded = store.load()
assert reloaded.chance == merged.chance, "Trajectory.chance must round-trip"
print(
    "\n  Trajectory.chance survived save/load -- transition tests replayed\n"
    "  from this store measure the model, not the dice."
)

# ---------------------------------------------------------------------------
# 3. Unit tests from history; pass rate must be 1.00.
#
# WHY THIS IS THE MOST IMPORTANT NUMBER IN THE REPO: a measuring instrument
# has to read zero on a known-zero input before any reading it gives means
# anything. The reference model generated this history, so a test suite
# built from that history must score PERFECTLY against the reference model.
# When this harness first ran it read 0.20 -- two bookkeeping bugs, not a
# modelling error -- and every downstream number inherited the corruption
# (see docs/OPERATING.md, "Before you trust any of these numbers").
# ---------------------------------------------------------------------------

banner(3, "The ground-truth test: pass rate must be exactly 1.00")

SUBJECT = ReferenceWorldModel(CONFIG)
total_passed = 0
total_tests = 0
for trajectory in trajectories:
    tests = generate(trajectory, Tolerance(), include_hidden=True)
    results = [test.run(SUBJECT) for test in tests]
    total_passed += sum(1 for result in results if result.passed)
    total_tests += len(results)
pass_rate = total_passed / total_tests
print(f"  tests generated from history: {total_tests}")
print(f"  passed against the model that wrote it: {total_passed}")
print(f"  pass rate: {pass_rate:.2f}")
assert pass_rate == 1.00, (
    f"the harness is charging the model for something other than being "
    f"wrong; every downstream number is noise (got {pass_rate:.2f})"
)
print(
    "\n  1.00, exactly. Tolerances are the tightened ones (counts 0.05,\n"
    "  rates 0.02) -- wide tolerances here would only ever conceal a broken\n"
    "  instrument. A synthesised model is now measured against this scale,\n"
    "  whose maximum is finally KNOWN to be 1.00."
)

# ---------------------------------------------------------------------------
# 4. ISMCTS: one decision, with the statistics behind it.
#
# WHY: the paper's core move is shifting the LLM's job from policy to
# MODEL, and letting search convert compute into decisions. Planning never
# calls a language model. The visit counts and values are printed because
# an operator about to spend a day of output on this plan is owed the
# numbers, not just the pick.
# ---------------------------------------------------------------------------

banner(4, "ISMCTS: search, not prompting (with visit counts and values)")

PLANNER = ISMCTS(
    ISMCTSConfig(simulations=150, rollout_depth=6, rollouts_per_leaf=4, seed=11)
)
state = SUBJECT.initial_state()
result = PLANNER.plan(SUBJECT, state, 0, 150)
ranked = sorted(result.visits.items(), key=lambda item: (-item[1], str(item[0])))
print(f"  chosen move:\n    {result.move}")
# Values are cumulative-margin units and run large in the reference model
# (reward sums every settled observation so far); their RELATIVE size is
# the decision-relevant quantity, so they are printed raw, not rescaled.
print(f"\n  {'action':<62} {'visits':>6} {'value':>13}")
for action, visits in ranked[:5]:
    value = result.values.get(action, 0.0)
    print(f"  {str(action)[:62]:<62} {visits:>6} {value:>13,.0f}")
print(
    f"\n  150 simulations. The top action absorbed {ranked[0][1]} of them --\n"
    "  that concentration IS the planner's confidence. An action visited a\n"
    "  handful of times is a hedge the search could not rule out, not a\n"
    "  recommendation. (Here search runs open-loop against the state; with a\n"
    "  synthesised model, a resample_state sampler would determinize over\n"
    "  what the observations actually support.)"
)

# ---------------------------------------------------------------------------
# 5. Colonel Blotto: a day's output as a force across contested fronts.
#
# WHY: a fixed daily pattern is a PURE strategy, and every pure strategy in
# Blotto is dominated once opponents can read it. The platform's ranking
# and every rival reading your feed are those opponents. The output is a
# DISTRIBUTION over allocations; one day draws from it.
# ---------------------------------------------------------------------------

banner(5, "Blotto: allocate the day's posts across fronts")

fronts = [
    Front("x", ANGLE, "solo_founder", 10.0),
    Front("tiktok", "enemy_of_slop", "solo_founder", 8.0),
    Front("linkedin", "founder_field_notes", "agency_ops", 6.0),
    Front("x", "founder_field_notes", "agency_ops", 4.0),
]
allocator = BlottoAllocator(BlottoConfig(units=10, seed=7))
mixture = allocator.equilibrium_mixture(fronts, samples=200)
day = allocator.sample(mixture, random.Random(42))
print(f"  {len(mixture)} distinct allocations in the equilibrium mixture")
print("  one day drawn from it:")
for front, units in day.items():
    print(f"    {front.platform:<8} {front.angle:<20} {units} post(s)")
assert sum(day.values()) == 10, "the budget is spent exactly"
mixed_exploit = allocator.mixture_exploitability(mixture, fronts, 20)
print(
    f"\n  Sums to exactly 10: the budget is spent whether the plan accounts\n"
    f"  for it or not. Best single response to the whole MIXTURE captures\n"
    f"  {mixed_exploit:.0%} of front value; a committed daily pattern would\n"
    f"  hand a reading opponent near 100%. Randomisation is not sloppiness\n"
    f"  here -- it is the only thing that makes your pattern unreadable."
)

# ---------------------------------------------------------------------------
# 6. EXP3: angle selection that adapts, because rewards are adversarial.
#
# WHY NOT UCB/THOMPSON: they assume each arm's reward distribution is fixed.
# Platform ranking weights drift, and the field crowds whatever paid last
# week -- the reward process is adversarial, and EXP3's guarantee holds
# against exactly that. Watch the distribution follow a moving best arm
# without ever collapsing onto it.
# ---------------------------------------------------------------------------

banner(6, "EXP3: angle selection under drift (200 rounds)")

ARMS = ["launch_theater", "enemy_of_slop", "founder_field_notes"]
exp3 = EXP3(ARMS, EXP3Config(gamma=0.15))
rng = random.Random(9)
# Rewards drift, as real angle payoffs do: launch_theater starts strong and
# decays (crowding + hook burnout); founder_field_notes starts weak and
# improves as the audience accretes. The crossover is around round 70 --
# steep enough that following it is a real test of adaptation, gentle
# enough that lock-in puts up a fight. Neither is announced to the algorithm.
checkpoints: dict[int, dict[str, float]] = {}
for round_index in range(200):
    probabilities = exp3.probabilities()
    if round_index in (0, 50, 100, 150, 199):
        checkpoints[round_index] = dict(probabilities)
    arm = exp3.select(rng)
    base = {
        "launch_theater": max(0.05, 0.9 - 0.012 * round_index),
        "enemy_of_slop": 0.5,
        "founder_field_notes": min(0.9, 0.1 + 0.012 * round_index),
    }
    noise = rng.uniform(-0.05, 0.05)
    exp3.update(arm, max(0.0, min(1.0, base[arm] + noise)))
print(f"  {'round':>6}  " + "  ".join(f"{arm[:16]:<16}" for arm in ARMS))
for round_index, probabilities in checkpoints.items():
    row = "  ".join(f"{probabilities[arm]:<16.3f}" for arm in ARMS)
    print(f"  {round_index:>6}  {row}")
final = checkpoints[199]
assert final["founder_field_notes"] > final["launch_theater"], (
    "EXP3 should have followed the drift by round 200"
)
print(
    "\n  The distribution MOVED with the world (launch_theater decays,\n"
    "  founder_field_notes accretes) and never collapsed: the exploration\n"
    "  floor gamma keeps every arm alive, so no drift event can permanently\n"
    "  hide an angle the way a locked-on UCB would."
)

# ---------------------------------------------------------------------------
# 7. THE CENTREPIECE: a Scale on 12 conversions is refused. At 50, permitted.
#
# WHY THIS IS WHAT THE SYSTEM IS FOR: both corpora name premature scaling
# as the way distribution budgets die -- CPA "frequently explode[s] to $150"
# when an arc is called at 5 conversions. So the gate is not a penalty the
# planner might out-argue; the move does not exist in the legal action set.
# Note the refusal carries its SOURCE: an operator told to wait is owed the
# passage that says so.
# ---------------------------------------------------------------------------

banner(7, "The evidence gate: refused at 12 conversions, permitted at 50")

ENGINE = LegalityEngine()
MOVE = Scale(ANGLE, 1.10)  # below 1.20: a creative nudge, not a budget commit


def gate_context(settled_7d: int) -> LegalityContext:
    """A context with the angle fully judgeable (coverage above the floor)
    and a known count of settled conversions in the trailing 7 days."""
    ctx = LegalityContext(current_date=date(2026, 8, 21))
    ctx.settled_conversions_7d[ANGLE] = settled_7d
    ctx.settled_conversions_14d[ANGLE] = 400
    ctx.angle_age_days[ANGLE] = 20
    ctx.angle_attribution_coverage[ANGLE] = 0.72
    ctx.allocation[ANGLE] = 100.0
    return ctx


print("  ATTEMPT: scale 'launch_theater' by 1.10x on 12 settled conversions")
print("  " + "-" * 76)
verdict = ENGINE.check(MOVE, gate_context(12))
assert not verdict.legal and verdict.rule == "CREATIVE_JUDGEMENT"
print("  REFUSED")
print(f"    rule:   {verdict.rule}")
print(f"    reason: {verdict.reason}")
print(f"    source: {verdict.source}")
print("  " + "-" * 76)
print(
    "\n  The planner CANNOT choose this move at any confidence -- it never\n"
    "  appears in the legal action set. 'Penalised' would mean the search\n"
    "  could still pick it on a lucky rollout; 'illegal' means the question\n"
    "  does not arise."
)

# Partial observations never count, no matter how loudly they brag.
peeking = [
    Observation(
        utm_content="c0412ab9",
        posted_at="2026-08-20",
        observed_at="2026-08-21",
        attributed_conversions=100,
        is_partial=True,
    ),
    Observation(
        utm_content="c07e33aa",
        posted_at="2026-08-15",
        observed_at="2026-08-21",
        attributed_conversions=12,
    ),
]
counted = settled_count(peeking)
print(
    f"\n  And peeking does not help: 100 conversions inside the reporting\n"
    f"  lag plus 12 settled counts as {counted} -- partials are dropped\n"
    f"  entirely, not prorated."
)

print("\n  SAME MOVE, gate cleared: 50 settled conversions in the window.")
print("  " + "-" * 76)
cleared = ENGINE.check(MOVE, gate_context(50))
assert cleared.legal
print(f"  PERMITTED ({cleared.rule})")
print(f"    source: {cleared.source}")
print("  " + "-" * 76)
print(
    "\n  Fifty settled conversions in seven days is one week of receipts.\n"
    "  The gate did not get easier; the EVIDENCE got better. That is the\n"
    "  entire contract: the system holds the door until the data, not the\n"
    "  enthusiasm, walks through it."
)

# ---------------------------------------------------------------------------
# 8. Congestion: the highest-benchmark archetype is the wrong one to crowd.
#
# WHY: everyone read the same benchmark table. Anti-hero rant has the best
# hook rate in the corpus, so the field piles onto it, and an angle's payoff
# falls with occupancy. The right question is never "which archetype is
# best?" but "which archetype is best GIVEN who is already on it?"
# ---------------------------------------------------------------------------

banner(8, "Congestion: best archetype != best bet")

congestion = CongestionConfig(decay="linear", decay_rate=0.5, floor=0.1)
raw = [
    (f"{archetype.value} (hook {prior.hook_rate.mid:.2f})",
     prior.hook_rate.mid * prior.expected_cvr.mid)
    for archetype, prior in ARCHETYPE_PRIORS.items()
]
print("  Raw benchmarks (attention x conversion):")
for name, value in sorted(raw, key=lambda item: -item[1]):
    print(f"    {name:<44}{value:.5f}")

# The field has read the table: the top archetype is crowded, the quiet
# ones are not. Occupancy is the share of players already on each angle.
occupancy = {
    "anti_hero_rant": 0.8,
    "founder_trauma": 0.2,
    "autonomous_voxel": 0.4,
    "ugc_review": 0.1,
}
adjusted = crowding_adjusted_ranking(
    [(archetype.value, prior.hook_rate.mid * prior.expected_cvr.mid)
     for archetype, prior in ARCHETYPE_PRIORS.items()],
    occupancy,
    congestion,
)
print("\n  After congestion (linear decay 0.5, field occupancy applied):")
for name, value in adjusted:
    print(f"    {name:<44}{value:.5f}  (occupancy {occupancy[name]:.1f})")
assert adjusted[0][0] != "anti_hero_rant", "crowding should dethrone the benchmark king"
assert adjusted[0][0] == "founder_trauma"
print(
    f"\n  anti_hero_rant, the highest raw benchmark, drops to "
    f"{dict(adjusted)['anti_hero_rant']:.5f} at 0.8 occupancy and LOSES to\n"
    f"founder_trauma at {dict(adjusted)['founder_trauma']:.5f} -- a lower\n"
    "  benchmark nobody crowded. This is why angle choice is modelled as a\n"
    "  congestion game (Rosenthal 1973) and not a leaderboard."
)

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("END OF WALKTHROUGH")
print("=" * 78)
print(
    "Everything above ran offline in one process: history -> tests -> search\n"
    "-> allocation -> adaptation -> refusal -> re-ranking. The live loop\n"
    "swaps the reference model for a synthesised one (`fronts synth`) and\n"
    "the dry adapters for Composio (`fronts publish --live`); nothing else\n"
    "changes. See docs/OPERATING.md for the Monday routine."
)
