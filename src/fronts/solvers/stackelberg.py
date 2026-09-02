"""Best response to an estimated ranking commitment.

The platform is a Stackelberg leader: it commits to a ranking rule before the
operator moves, and never reveals it. The operator is therefore always a
follower, and correct play is a best response to an *estimate* of the
commitment -- never to the commitment itself, which is unobservable, and
never to a single point estimate of it, which is the mistake this module
exists to talk the caller out of.

Why a point estimate is the wrong default under drift: the estimate carries a
posterior, and a move that maximises the score under the posterior mean can
be brittle across that posterior -- optimal if the weights are near the mean,
catastrophic in the tail, and the tail is where a drifting ranking function
spends its time. Maximin (``worst_case``) picks the move with the best score
under the least favourable estimate in the sampled set. It buys robustness at
a known and computable cost in expected value: the gap between the maximin
move's expected score and the best expected score, which for a serious
posterior is exactly the premium worth considering before paying it.
``value_of_information`` quantifies the other side -- what perfect observation
of the commitment would be worth, i.e. the size of the prize for narrowing the
posterior at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

__all__ = [
    "LeaderEstimate",
    "best_response",
    "robust_best_response",
    "value_of_information",
]

M = TypeVar("M")
"""The move type. Deliberately generic: candidates may be angles, whole posts,
or allocation bundles -- anything the caller's ``score_fn`` can score."""

ScoreFn = Callable[[M, Mapping[str, float]], float]
"""Scores a move under a weight vector: ``score_fn(move, weights) -> float``."""


@dataclass
class LeaderEstimate:
    """One sample from the posterior over the platform's ranking weights.

    ``weights`` is the estimated weight vector the score function consumes;
    ``confidence`` is how much this sample deserves (posterior mass, sample
    size, or a judgement call -- the semantics are the caller's); and
    ``drift_rate`` is the expected per-step magnitude of the random walk in
    the underlying weights, carried per estimate so the caller can weight
    recent samples more heavily than stale ones.
    """

    weights: dict[str, float]
    confidence: float = 1.0
    drift_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.drift_rate < 0.0:
            raise ValueError(f"drift_rate must be >= 0, got {self.drift_rate}")
        if not self.weights:
            raise ValueError("weights must be a non-empty mapping")


def best_response(
    candidate_moves: Sequence[M],
    score_fn: ScoreFn[M],
    estimate: LeaderEstimate,
) -> list[tuple[M, float]]:
    """Rank candidate moves under one point estimate, best first.

    Ties keep the candidate order, so the ranking is reproducible. This is
    the right tool when the posterior has collapsed (drift is negligible and
    the estimate is sharp); with any real spread, prefer
    ``robust_best_response``.
    """
    scored = [(move, score_fn(move, estimate.weights)) for move in candidate_moves]
    return sorted(scored, key=lambda item: -item[1])


def robust_best_response(
    candidate_moves: Sequence[M],
    score_fn: ScoreFn[M],
    estimates: Sequence[LeaderEstimate],
    aggregator: str = "worst_case",
) -> list[tuple[M, float]]:
    """Rank moves by an aggregate of their scores across sampled estimates.

    ``aggregator`` selects the aggregate: "worst_case" scores each move by its
    minimum over the estimate set (maximin -- see the module docstring for
    when that premium is worth paying), "expected" scores it by the
    confidence-weighted mean over the set. Estimates with zero total
    confidence fall back to uniform weights rather than dividing by zero.

    An empty estimate set is rejected: robustness against nothing is just a
    point estimate wearing a heavier name.
    """
    if not estimates:
        raise ValueError("estimates must be non-empty for a robust response")
    if aggregator not in ("worst_case", "expected"):
        raise ValueError(
            f"aggregator must be worst_case|expected, got {aggregator!r}"
        )

    weights_total = sum(estimate.confidence for estimate in estimates)
    aggregated: list[tuple[M, float]] = []
    for move in candidate_moves:
        scores = [score_fn(move, estimate.weights) for estimate in estimates]
        if aggregator == "worst_case":
            aggregated.append((move, min(scores)))
        elif weights_total > 0.0:
            aggregated.append(
                (
                    move,
                    sum(
                        score * estimate.confidence
                        for score, estimate in zip(scores, estimates, strict=True)
                    )
                    / weights_total,
                )
            )
        else:
            aggregated.append((move, sum(scores) / len(scores)))
    return sorted(aggregated, key=lambda item: -item[1])


def value_of_information(
    estimates: Sequence[LeaderEstimate],
    candidate_moves: Sequence[M],
    score_fn: ScoreFn[M],
) -> float:
    """What a perfect observation of the leader's commitment would be worth.

    The clairvoyance gap:

        E_estimate[ best_move score under that estimate ]
        - best_move E_estimate[ score ]

    The first term is the expected score if the operator could see the actual
    weights each period and re-pick optimally; the second is the best
    achievable by committing to one move now under the posterior. The
    difference is the value of partial observability removed from the
    operator's life by never seeing the ranking rule. It is the most useful
    diagnostic this module produces: a large gap says invest in inference
    (more sampling of the posterior, better attribution), a small gap says the
    move choice barely depends on the weights and no amount of spying on the
    platform will pay for itself.

    Confidence weights the expectation, with a uniform fallback when total
    confidence is zero. With a single estimate the gap is always 0 -- there is
    no spread to exploit -- which is the correct answer and a useful sanity
    check on the caller's posterior.
    """
    if not estimates:
        raise ValueError("estimates must be non-empty")
    if not candidate_moves:
        raise ValueError("candidate_moves must be non-empty")

    weights_total = sum(estimate.confidence for estimate in estimates)
    if weights_total > 0.0:
        probabilities = [e.confidence / weights_total for e in estimates]
    else:
        probabilities = [1.0 / len(estimates)] * len(estimates)

    expected_best = sum(
        probability
        * max(score_fn(move, estimate.weights) for move in candidate_moves)
        for estimate, probability in zip(estimates, probabilities, strict=True)
    )
    best_expected = max(
        sum(
            probability * score_fn(move, estimate.weights)
            for estimate, probability in zip(estimates, probabilities, strict=True)
        )
        for move in candidate_moves
    )
    return expected_best - best_expected
