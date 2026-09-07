"""Costly signalling: when a claim is credible and when it is noise.

Origin: Spence, "Job Market Signaling", Quarterly Journal of Economics 87(3),
1973. Spence's insight is that a signal separates types not because it is
informative but because it is *differentially expensive*: education signals
productivity only insofar as it costs less for productive people. Transferred
to content, a claim in a post is credible exactly when imitating it costs a
low-quality operator more than the deception is worth. That converts "be
radically transparent" from advice into a computable inequality, and the
inequality usually shows that transparency whose imitation is cheap (naming
your revenue when nobody checks) separates nobody.

The mapping to content is an analogy and rests on two explicit assumptions.
Receivers are Bayesian: they update on the signal rationally rather than
emotionally, which is at best approximately true of a scrolling audience.
And imitation cost is observable *in expectation*: the receiver does not see
what the claim cost to make, only a distributional belief about what it would
cost them. Where either assumption fails, the inequality below still computes
something, but that something is no longer Spence's equilibrium.

Claim classes arrive as plain strings. The mapping from the game's
``ClaimClass`` enum to a ``SignalCost`` belongs to the caller: it encodes
empirical beliefs about what each kind of claim costs to fake, and a solver
should not be in the business of hard-coding those beliefs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["SignalCost", "SignallingConfig", "separates", "separating_power", "ClaimCredibility"]


@dataclass(frozen=True)
class SignalCost:
    """The cost structure of one claim, for the two sender types.

    ``cost_high_type`` / ``cost_low_type``: what it costs a genuine operator /
    an imitator to make the claim truthfully enough to survive scrutiny. The
    gap between them is the signal; a claim equally costly to both types
    carries no information. ``gain_from_deception``: what the low type gets
    from being believed (the value of traffic it cannot retain).
    ``benefit_high_type``: what the high type gets from being believed, net of
    nothing, the gross benefit, against which its own cost is measured.
    """

    cost_high_type: float
    cost_low_type: float
    gain_from_deception: float
    benefit_high_type: float

    def __post_init__(self) -> None:
        for name in (
            "cost_high_type",
            "cost_low_type",
            "gain_from_deception",
            "benefit_high_type",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass
class SignallingConfig:
    """Numeric knobs for the credibility score, all with documented defaults."""

    epsilon: float = 1e-9
    """Denominator guard for the normalised margins in ``separating_power``.
    Any small positive value works; it only matters when a cost and the gain
    it is compared against are both zero, i.e. a free claim against a free
    deception, which by construction has zero separating power anyway."""

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")


def separates(signal_cost: SignalCost) -> bool:
    """True when a separating equilibrium exists for this claim.

    The standard single-crossing conditions, both required (Spence 1973):

    1. The low type will not imitate: its cost of making the claim exceeds
       what the deception earns it, ``cost_low_type > gain_from_deception``.
    2. The high type will still send the signal: the claim costs it no more
       than being believed is worth, ``cost_high_type <= benefit_high_type``.

    Both must hold. A claim cheap enough for imitators is noise no matter how
    expensive it is for the genuine operator; a claim the genuine operator
    cannot afford is a separating equilibrium nobody plays.
    """
    return (
        signal_cost.cost_low_type > signal_cost.gain_from_deception
        and signal_cost.cost_high_type <= signal_cost.benefit_high_type
    )


def separating_power(signal_cost: SignalCost, config: SignallingConfig | None = None) -> float:
    """Normalised strength of the separating condition, in [0, 1].

    Computed as the weaker of the two incentive margins, each normalised by
    the scale of the quantities involved:

        low_margin  = (cost_low_type - gain_from_deception)
                      / max(cost_low_type, gain_from_deception, epsilon)
        high_margin = (benefit_high_type - cost_high_type)
                      / max(benefit_high_type, cost_high_type, epsilon)

    The minimum is taken because a chain separates at its weakest constraint,
    and the result is clamped to [0, 1]: a violated constraint gives a
    negative margin, which reports as zero power rather than as a negative
    number that could masquerade as a ranking signal. ``separates`` is the
    yes/no question; this is the same answer on a scale, for ranking claims
    against each other when several clear the bar.
    """
    resolved = config if config is not None else SignallingConfig()
    low_margin = (
        signal_cost.cost_low_type - signal_cost.gain_from_deception
    ) / max(
        signal_cost.cost_low_type,
        signal_cost.gain_from_deception,
        resolved.epsilon,
    )
    high_margin = (
        signal_cost.benefit_high_type - signal_cost.cost_high_type
    ) / max(
        signal_cost.benefit_high_type,
        signal_cost.cost_high_type,
        resolved.epsilon,
    )
    return max(0.0, min(1.0, min(low_margin, high_margin)))


class ClaimCredibility:
    """Scores and ranks claims by their separating power.

    A thin wrapper: it exists so callers can hold one scorer with one config
    and ask it to rank a slate of (claim, cost) pairs, rather than threading
    the normalisation config through every call site by hand.
    """

    def __init__(self, config: SignallingConfig | None = None) -> None:
        self._config = config if config is not None else SignallingConfig()

    def credibility(self, claim: str, cost: SignalCost) -> float:
        """Separating power of one claim. ``claim`` is an opaque string key."""
        return separating_power(cost, self._config)

    def rank(
        self, claims: Sequence[tuple[str, SignalCost]]
    ) -> list[tuple[str, float]]:
        """Order claims from most to least credible, with scores attached.

        Ties break alphabetically by claim so the order is reproducible. A
        claim that fails ``separates`` scores 0.0 and sinks to the bottom
        rather than being dropped: an incredible claim still appears in the
        output, because "this claim separates nothing" is information the
        caller asked for when they included it.
        """
        scored = [
            (claim, self.credibility(claim, cost)) for claim, cost in claims
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))
