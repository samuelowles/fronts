"""Adversarial bandits over angles: EXP3 and its high-probability variant.

Origin: Auer, Cesa-Bianchi, Freund and Schapire, "The Nonstochastic
Multiarmed Bandit Problem", SIAM Journal on Computing 32(1), 2002.

Why EXP3 and not UCB or Thompson sampling, since this module has to commit to
one: UCB and Thompson are *stochastic* bandits. Their regret guarantees assume
each arm's reward distribution is fixed and unknown, and both will lock onto
an arm whose early rewards were good, exploiting it indefinitely. Angle
selection violates that assumption twice over. The platform's ranking weights
drift continuously (a non-stationary reward process), and the field of rival
creators adapts to whatever we do: if we pile onto an angle because it paid
last week, the crowding that follows is a response to our policy rather than
noise around a fixed mean. That is the adversarial setting EXP3
was built for: it keeps a persistent exploration floor
(gamma) so no drift or crowding event can permanently hide an arm, and its
guarantee holds against an arbitrary reward sequence, not a stochastic one.

Two implementation details that naive EXP3s get wrong and this module does
not. Rewards must already be normalised to [0, 1]; ``update`` validates that
and raises rather than silently corrupting the weights, because an
importance-weighted reward outside [0, 1] is on the wrong scale, and every
subsequent probability computed from it is garbage. And weights are
renormalised whenever the maximum exceeds ``weight_ceiling``: the unbiased
estimator scales reward by 1/p, so under a persistent winner the winning
weight grows without bound and a long enough run overflows to inf, after which
every probability collapses. The standard practical fix is rescaling all
weights by their maximum, which changes no arm's relative weight and therefore
no distribution.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["EXP3Config", "EXP3PConfig", "EXP3", "EXP3P", "regret_bound"]


@dataclass
class EXP3Config:
    """Numeric knobs for EXP3, all with documented defaults."""

    gamma: float = 0.1
    """Exploration rate: the probability mass spread uniformly over all arms
    regardless of weights. Must lie in (0, 1]. Smaller gamma exploits harder
    and adapts slower; the value minimising the regret bound for K arms and T
    rounds is sqrt(K ln K / ((e - 1) T))."""

    weight_ceiling: float = 1e100
    """Renormalise all weights whenever the maximum exceeds this. Any value
    comfortably below the float overflow threshold (~1.8e308) works; the
    choice only affects how often rescaling happens, never the induced
    distribution."""

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.weight_ceiling <= 0.0:
            raise ValueError(f"weight_ceiling must be > 0, got {self.weight_ceiling}")


@dataclass
class EXP3PConfig:
    """Numeric knobs for EXP3.P, all with documented defaults."""

    gamma: float = 0.05
    """Exploration mix, as in EXP3."""

    eta: float = 0.1
    """Learning rate in the weight exponent. The original paper folds this
    into gamma; exposing it separately lets a caller decay the rate over a
    known horizon without also changing the exploration floor."""

    beta: float = 0.01
    """Confidence term added to the capped reward estimate on every update.
    Auer et al. (2002) recommend sqrt(ln K / (K T)) for K arms over a known
    horizon T; 0.01 is that value near K = 5, T = 10^4."""

    weight_ceiling: float = 1e100
    """Renormalisation threshold, as in EXP3Config."""

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.eta <= 0.0:
            raise ValueError(f"eta must be > 0, got {self.eta}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.weight_ceiling <= 0.0:
            raise ValueError(f"weight_ceiling must be > 0, got {self.weight_ceiling}")


def _validate_arms(arms: Sequence[str]) -> list[str]:
    if not arms:
        raise ValueError("arms must be a non-empty sequence")
    unique = list(arms)
    if len(set(unique)) != len(unique):
        raise ValueError("arms must be unique; duplicate arms corrupt the mixture")
    return unique


def _validate_reward(reward: float) -> None:
    """Reject out-of-range rewards instead of corrupting the weights."""
    if not 0.0 <= reward <= 1.0:
        raise ValueError(
            f"reward must be normalised into [0, 1] by the caller, got {reward}"
        )


class EXP3:
    """Exponential-weight algorithm for the nonstochastic (adversarial) bandit.

    Weights start uniform. Each round: draw an arm from the mixed distribution
    (exploit the weights, explore with probability gamma), then update the
    drawn arm's weight by the importance-weighted reward ``reward / p`` where
    p is the probability the arm was actually drawn with, the factor that
    makes an adversarial guarantee possible, since it de-biases the one sample
    we saw into an estimate of what a full-information algorithm would have
    seen.
    """

    arms: list[str]
    gamma: float
    weights: dict[str, float]

    def __init__(self, arms: Sequence[str], config: EXP3Config | None = None) -> None:
        self._config = config if config is not None else EXP3Config()
        self.arms = _validate_arms(arms)
        self.gamma = self._config.gamma
        self.weights = {arm: 1.0 for arm in self.arms}

    def probabilities(self) -> dict[str, float]:
        """Current mixed distribution over arms, summing to 1."""
        total = sum(self.weights.values())
        floor = self.gamma / len(self.arms)
        exploit = (1.0 - self.gamma) / total
        return {
            arm: floor + exploit * weight for arm, weight in self.weights.items()
        }

    def select(self, rng: random.Random) -> str:
        """Draw an arm from the current distribution using the caller's RNG."""
        probabilities = self.probabilities()
        draw = rng.random()
        cumulative = 0.0
        for arm, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                return arm
        return self.arms[-1]

    def update(self, arm: str, reward: float) -> None:
        """Fold one observed reward into the drawn arm's weight.

        The estimate is the importance-weighted reward ``reward / p_arm``,
        exponentiated by ``gamma / K`` as in Algorithm 1 of Auer et al.
        (2002). Rewards outside [0, 1] raise ValueError; see the module
        docstring for why this is validation rather than clipping.
        """
        if arm not in self.weights:
            raise KeyError(f"unknown arm {arm!r}")
        _validate_reward(reward)
        probability = self.probabilities()[arm]
        estimate = reward / probability
        self.weights[arm] *= math.exp(self.gamma * estimate / len(self.arms))
        self._renormalise_if_needed()

    def _renormalise_if_needed(self) -> None:
        """Rescale every weight by the maximum when it passes the ceiling.

        Dividing all weights by a common factor leaves the induced
        distribution unchanged, so this is purely numerical hygiene: it keeps
        a long-running winner from driving the weights to overflow.
        """
        largest = max(self.weights.values())
        if largest > self._config.weight_ceiling:
            for arm in self.weights:
                self.weights[arm] /= largest


class EXP3P:
    """EXP3.P: the high-probability variant, with a confidence term.

    EXP3's guarantee is in expectation. EXP3.P (Auer et al. 2002, section 4)
    adds a confidence term beta to the update so the bound holds with high
    probability (against a single adversarial realisation of the rewards,
    not merely on average over replays). The estimate is additionally capped at
    1 (``s_hat = min(1, reward / p)``), trading a little bias for a bound on
    the variance of the importance-weighted estimator; the cap is what makes
    the high-probability argument go through.

    The cost is a slower, more conservative learner: beta inflates every
    update, so the algorithm is harder to surprise. For angle selection the
    trade is usually right: one adversarial week in which the estimator
    latches onto a decaying angle is the failure this variant
    prevents.
    """

    arms: list[str]
    gamma: float
    weights: dict[str, float]

    def __init__(self, arms: Sequence[str], config: EXP3PConfig | None = None) -> None:
        self._config = config if config is not None else EXP3PConfig()
        self.arms = _validate_arms(arms)
        self.gamma = self._config.gamma
        self.weights = {arm: 1.0 for arm in self.arms}

    def probabilities(self) -> dict[str, float]:
        """Current mixed distribution over arms, summing to 1."""
        total = sum(self.weights.values())
        floor = self.gamma / len(self.arms)
        exploit = (1.0 - self.gamma) / total
        return {arm: floor + exploit * weight for arm, weight in self.weights.items()}

    def select(self, rng: random.Random) -> str:
        """Draw an arm from the current distribution using the caller's RNG."""
        probabilities = self.probabilities()
        draw = rng.random()
        cumulative = 0.0
        for arm, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                return arm
        return self.arms[-1]

    def update(self, arm: str, reward: float) -> None:
        """Update with the capped estimate plus the confidence term.

        ``w_arm *= exp((eta * min(1, reward / p) + beta) / K)``, the paper's
        update with its learning rate exposed as ``eta``. The cap bounds the
        estimate at 1 so a single very unlikely draw cannot dominate the
        weights; beta is the confidence bonus.
        """
        if arm not in self.weights:
            raise KeyError(f"unknown arm {arm!r}")
        _validate_reward(reward)
        probability = self.probabilities()[arm]
        estimate = min(1.0, reward / probability)
        exponent = (self._config.eta * estimate + self._config.beta) / len(self.arms)
        self.weights[arm] *= math.exp(exponent)
        self._renormalise_if_needed()

    def _renormalise_if_needed(self) -> None:
        """Rescale every weight by the maximum when it passes the ceiling."""
        largest = max(self.weights.values())
        if largest > self._config.weight_ceiling:
            for arm in self.weights:
                self.weights[arm] /= largest


def regret_bound(num_arms: int, rounds: int, gamma: float) -> float:
    """Theoretical expected-regret bound for EXP3.

    Formula (Auer et al. 2002, Theorem 3.1, rewards in [0, 1]):

        G_max - E[G_EXP3] <= (K ln K) / gamma + (e - 1) * gamma * T

    where K is the number of arms, T the number of rounds, and G_max the best
    fixed arm's cumulative reward (at most T since rewards are capped at 1).
    Minimised over gamma at gamma* = sqrt(K ln K / ((e - 1) T)) the bound is
    2 sqrt((e - 1) T K ln K), the O(sqrt(T K ln K)) adversarial rate.
    """
    if num_arms < 1:
        raise ValueError(f"num_arms must be >= 1, got {num_arms}")
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    first = num_arms * math.log(num_arms) / gamma
    second = (math.e - 1.0) * gamma * rounds
    return first + second
