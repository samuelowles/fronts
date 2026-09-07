"""Colonel Blotto allocation across contested fronts.

Origin: the Colonel Blotto game, and specifically Roberson, "The Colonel
Blotto Game", Economic Theory 29, 2006, which characterises the equilibrium
of the symmetric constant-sum case: with equal budgets the unique equilibrium
marginal distribution of forces on a front is uniform on [0, 2b/n] (n fronts,
budget b). This module discretises that construction to indivisible units.

Why a distribution and not a plan: in Blotto with two or more fronts, every
pure strategy is strictly dominated once the opponent can learn it. A fixed
allocation can be read and outbid on every front that matters; only
randomisation makes the opponent's cost of taking a front independent of what
they know about you. The practical translation for distribution is direct:
a fixed daily posting pattern is a pure strategy, and the platform's ranking
plus every rival creator reading your feed are the outbidding opponents. The
output of ``equilibrium_mixture`` is therefore a list of allocations with
probabilities, and ``sample`` draws one day's plan from it.

Fronts are (platform, angle, avatar) triples carried as plain strings. The
mapping from the game's ``Platform`` enum and angle taxonomy to these keys is
the caller's business; a solver that allocates units does not need to know
what the units are posts or dollars.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["Front", "BlottoConfig", "BlottoAllocator"]


@dataclass(frozen=True)
class Front:
    """One contested allocation target with a prize weight.

    ``value`` is the prize for winning the front, not a multiplier on units.
    Higher-value fronts attract more of everyone's budget, which is
    the congestion effect the equilibrium construction prices.
    """

    platform: str
    angle: str
    avatar: str
    value: float


@dataclass
class BlottoConfig:
    """Numeric knobs for the allocator, all with documented defaults."""

    units: int = 10
    """Total indivisible units per period. The budget is spent exactly: every
    allocation this module returns sums to ``units``, never more, never less,
    because an operator's daily output is fixed whether the plan accounts for
    it or not."""

    seed: int | None = None
    """Seed for every random draw. None means system entropy."""

    opponent_model: str = "uniform"
    """How the exploitability search generates candidate opponents:
    "uniform" (random spreads), "proportional" (value-proportional spreads),
    or "observed" (value-proportional with concentration, a stand-in for real
    observed rival play until the caller has data to inject)."""

    observed_concentration: float = 2.0
    """Exponent applied to value shares under the "observed" opponent model.
    Above 1 concentrates rival budget on high-value fronts, which is what
    observed creator behaviour looks like in aggregate."""

    def __post_init__(self) -> None:
        if self.units < 0:
            raise ValueError(f"units must be >= 0, got {self.units}")
        if self.opponent_model not in ("uniform", "proportional", "observed"):
            raise ValueError(
                f"opponent_model must be uniform|proportional|observed, "
                f"got {self.opponent_model!r}"
            )
        if self.observed_concentration <= 0.0:
            raise ValueError(
                f"observed_concentration must be > 0, got {self.observed_concentration}"
            )


class BlottoAllocator:
    """Colonel Blotto allocation with an exact integer budget."""

    def __init__(self, config: BlottoConfig | None = None) -> None:
        self._config = config if config is not None else BlottoConfig()
        self._rng = random.Random(self._config.seed)

    # --- pure strategies ---------------------------------------------------

    def pure_best_response(
        self,
        fronts: Sequence[Front],
        opponent_allocation: dict[Front, int],
    ) -> dict[Front, int]:
        """Greedy best response to a *known* opponent allocation.

        Winning front i costs ``opponent_allocation[i] + 1`` units and pays
        ``value_i``, so the greedy order is by value per unit of cost, the
        fractional-knapsack order, which is the standard best-response
        heuristic for winner-take-more Blotto. Leftover units that can win
        nothing more are parked on the highest-value front so the allocation
        still sums to exactly ``units``; surplus strength is worthless in a
        strict-majority contest but the budget is spent either way.

        Useful against a predictable opponent, and useful as the exploitation
        oracle inside ``exploitability``. Useless as a committed strategy,
        which is the point of the module docstring.
        """
        allocation = {front: 0 for front in fronts}
        if not fronts:
            return allocation

        remaining = self._config.units
        order = sorted(
            fronts,
            key=lambda f: (
                -(f.value / (opponent_allocation.get(f, 0) + 1)),
                -f.value,
                f.platform,
                f.angle,
            ),
        )
        for front in order:
            cost = opponent_allocation.get(front, 0) + 1
            if cost <= remaining:
                allocation[front] = cost
                remaining -= cost
        if remaining > 0:
            strongest = max(fronts, key=lambda f: f.value)
            allocation[strongest] += remaining
        return allocation

    # --- mixed strategies --------------------------------------------------

    def equilibrium_mixture(
        self,
        fronts: Sequence[Front],
        samples: int,
    ) -> list[tuple[dict[Front, int], float]]:
        """Roberson-style mixed strategy, discretised to integer units.

        Construction: front i's value share is ``s_i = value_i / total``; the
        equilibrium marginal puts units on a front uniformly over
        ``[0, floor(2 * units * s_i)]``, truncated at ``units``. Each sample
        draws every marginal independently and is then repaired so the
        allocation sums to exactly ``units``. The repair is exact: strip from
        the least valuable over-committed fronts when over budget, top up the
        fronts with headroom when under.

        Discretising loses the exactness of Roberson's continuous result:
        with integer units the uniform marginal is only approximately
        achievable, and the repair step moves mass around. The direction of
        the construction (spread roughly uniform up to twice the pro-rata
        share, so strong fronts sometimes get nothing) is what survives, and
        it is what makes the mixture's exploitability lower than any
        reasonable pure allocation's.
        """
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")
        if not fronts:
            return []

        caps = self._marginal_caps(fronts)
        counts: dict[tuple[int, ...], int] = {}
        for _ in range(samples):
            draw = [self._rng.randint(0, cap) for cap in caps]
            draw = self._repair(draw, caps)
            key = tuple(draw)
            counts[key] = counts.get(key, 0) + 1

        mixture: list[tuple[dict[Front, int], float]] = []
        for key, count in counts.items():
            allocation = dict(zip(fronts, key, strict=True))
            mixture.append((allocation, count / samples))
        return mixture

    def sample(
        self,
        mixture: list[tuple[dict[Front, int], float]],
        rng: random.Random,
    ) -> dict[Front, int]:
        """Draw one day's allocation from a mixture using the given RNG."""
        if not mixture:
            raise ValueError("cannot sample from an empty mixture")
        total = sum(probability for _, probability in mixture)
        if total <= 0.0:
            raise ValueError("mixture probabilities must sum to a positive value")
        draw = rng.random() * total
        cumulative = 0.0
        for allocation, probability in mixture:
            cumulative += probability
            if draw < cumulative:
                return allocation
        return mixture[-1][0]

    # --- evaluation ----------------------------------------------------------

    def exploitability(
        self,
        allocation: dict[Front, int],
        fronts: Sequence[Front],
        opponent_samples: int,
    ) -> float:
        """Share of total front value a best-responding opponent captures.

        Lower is better. This is the degenerate case of
        ``mixture_exploitability``: the allocation is known to the opponent,
        so a single committed response exploiting it exists. A committed pure
        strategy scores badly here because it can be read; see
        ``mixture_exploitability`` for the mixed-strategy comparison, which is
        the one that shows why the module returns distributions.
        """
        return self.mixture_exploitability(
            [(allocation, 1.0)], fronts, opponent_samples
        )

    def mixture_exploitability(
        self,
        mixture: list[tuple[dict[Front, int], float]],
        fronts: Sequence[Front],
        opponent_samples: int,
    ) -> float:
        """Value share a best-responding opponent captures against a mixture.

        The opponent commits to one response without seeing the draw, so the
        score is ``max over candidate responses o of E_{a ~ mixture}[value o
        captures from a]``, divided by total value. Candidates are the greedy
        best response to each allocation in the mixture plus
        ``opponent_samples`` allocations drawn from the configured opponent
        model; the max over a subset is a lower bound on the true best
        response's capture, which is the conservative direction (it can only
        understate the opponent, never overstate them).

        This is the metric under which mixing wins. Against a known pure
        allocation the expectation collapses and the opponent exploits
        exactly; against a mixture any single response is spread thin across
        draws. Averaging ``exploitability`` over mixture samples instead
        (scoring each draw as if the opponent saw it) would defeat the point
        of measuring a mixed strategy at all.
        """
        if not fronts:
            return 0.0
        if not mixture:
            return 0.0
        if opponent_samples < 0:
            raise ValueError(f"opponent_samples must be >= 0, got {opponent_samples}")

        total_probability = sum(probability for _, probability in mixture)
        if total_probability <= 0.0:
            raise ValueError("mixture probabilities must sum to a positive value")

        candidates: list[dict[Front, int]] = [
            self.pure_best_response(fronts, allocation)
            for allocation, _ in mixture
        ]
        candidates.extend(
            self._random_opponent(fronts) for _ in range(opponent_samples)
        )

        best_capture = max(
            sum(
                (probability / total_probability)
                * self._captured_value(candidate, allocation, fronts)
                for allocation, probability in mixture
            )
            for candidate in candidates
        )
        total_value = sum(front.value for front in fronts)
        if total_value <= 0.0:
            return 0.0
        return best_capture / total_value

    # --- internals -----------------------------------------------------------

    def _marginal_caps(self, fronts: Sequence[Front]) -> list[int]:
        """Per-front draw caps: ``floor(2 * units * value_share)``, in [0, units]."""
        total_value = sum(front.value for front in fronts)
        caps: list[int] = []
        for front in fronts:
            share = front.value / total_value if total_value > 0.0 else 1.0 / len(fronts)
            cap = int(2 * self._config.units * share)
            caps.append(max(0, min(cap, self._config.units)))
        return caps

    def _repair(self, draw: list[int], caps: list[int]) -> list[int]:
        """Force the draw to sum to exactly ``units``, never negative.

        Over budget: remove from the fronts with the most units, breaking ties
        toward the front with the lowest cap (the cap is a proxy for value
        share, so this strips the least valuable over-commitment first).
        Under budget: add to fronts with headroom below their cap, breaking
        ties toward the largest cap. If every front is at cap and budget
        remains (possible only when the caps themselves cannot absorb the
        budget), the excess goes to the largest-cap front, because an
        exact-sum allocation is a harder requirement than a respected marginal.
        """
        units = self._config.units
        result = list(draw)
        surplus = sum(result) - units
        while surplus > 0:
            donors = [i for i, count in enumerate(result) if count > 0]
            if not donors:
                break
            pick = max(donors, key=lambda i: (result[i], -caps[i]))
            take = min(surplus, result[pick])
            result[pick] -= take
            surplus -= take
        deficit = units - sum(result)
        while deficit > 0:
            room = [i for i in range(len(result)) if result[i] < caps[i]]
            if room:
                pick = max(room, key=lambda i: (caps[i] - result[i], caps[i]))
                give = min(deficit, caps[pick] - result[pick])
                result[pick] += give
                deficit -= give
            else:
                pick = max(range(len(result)), key=lambda i: (caps[i], result[i]))
                result[pick] += deficit
                deficit = 0
        return result

    def _random_opponent(self, fronts: Sequence[Front]) -> dict[Front, int]:
        """One candidate opponent allocation under the configured model."""
        units = self._config.units
        allocation = {front: 0 for front in fronts}
        if units == 0 or not fronts:
            return allocation

        if self._config.opponent_model == "uniform":
            weights = [1.0] * len(fronts)
        elif self._config.opponent_model == "proportional":
            weights = [max(front.value, 0.0) for front in fronts]
        else:
            exponent = self._config.observed_concentration
            weights = [max(front.value, 0.0) ** exponent for front in fronts]

        total_weight = sum(weights)
        if total_weight <= 0.0:
            shares = [1.0 / len(fronts)] * len(fronts)
        else:
            shares = [weight / total_weight for weight in weights]

        # Multinomial allocation: each of the units independently lands on a
        # front with the model's probability, then a deterministic repair
        # guarantees the exact sum.
        raw = [0] * len(fronts)
        for _ in range(units):
            draw = self._rng.random()
            cumulative = 0.0
            for index, share in enumerate(shares):
                cumulative += share
                if draw < cumulative:
                    raw[index] += 1
                    break
            else:
                raw[-1] += 1
        raw = self._repair(raw, [units] * len(fronts))
        return dict(zip(fronts, raw, strict=True))

    def _captured_value(
        self,
        attacker: dict[Front, int],
        defender: dict[Front, int],
        fronts: Sequence[Front],
    ) -> float:
        """Value the attacker wins: strict majority takes the whole front."""
        return sum(
            front.value
            for front in fronts
            if attacker.get(front, 0) > defender.get(front, 0)
        )
