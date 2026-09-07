"""Angles as a congestion game.

Origin: Rosenthal, "A Class of Games Possessing Pure-Strategy Nash
Equilibria", International Journal of Game Theory 2, 1973. Rosenthal's
potential function is the reason this module exists: a game that admits one
has the property that every player's incentive to deviate unilaterally is
exactly the change in the potential, so any sequence of strict improvements
walks the potential uphill, cannot cycle, and must terminate at a pure Nash
equilibrium. For angle selection this is the structurally right model: an
angle's payoff to us falls in how many rivals run it, which makes the
highest-benchmark archetype the wrong default whenever everyone else has read
the same benchmark table.

The occupancy convention: ``occupancy = (number of players on the angle - 1) /
(number of players in the game)``, so a solo player faces occupancy 0 and the
last possible joiner faces (N-1)/N < 1. Occupancy is a share, not a headcount,
because the decay curves are calibrated on shares.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "CongestionConfig",
    "congested_payoff",
    "CongestionGame",
    "crowding_adjusted_ranking",
]


@dataclass
class CongestionConfig:
    """Numeric knobs for congestion decay, all with documented defaults."""

    decay: str = "linear"
    """Decay family: "linear" (share = 1 - rate * occupancy), "exponential"
    (share = exp(-rate * occupancy)), or "power" (share = (1 + occupancy)
    ** -rate). Linear is the default because it is the most conservative at
    high occupancy before the floor binds; exponential and power are for when
    crowding damage has been measured to compound."""

    decay_rate: float = 0.5
    """How sharply payoff falls with occupancy. 0 means no congestion at all;
    1 under linear decay zeroes the payoff at full occupancy."""

    floor: float = 0.1
    """Minimum residual payoff share, applied after the decay curve. A floor
    above 0 keeps the game finite-valued and, more importantly, keeps the
    potential bounded, which is what guarantees best-response dynamics
    terminate rather than oscillate forever."""

    def __post_init__(self) -> None:
        if self.decay not in ("linear", "exponential", "power"):
            raise ValueError(
                f"decay must be linear|exponential|power, got {self.decay!r}"
            )
        if self.decay_rate < 0.0:
            raise ValueError(f"decay_rate must be >= 0, got {self.decay_rate}")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {self.floor}")


def _residual_share(occupancy: float, config: CongestionConfig) -> float:
    """Share of base payoff surviving at ``occupancy``, before the floor."""
    if config.decay == "linear":
        return 1.0 - config.decay_rate * occupancy
    if config.decay == "exponential":
        return math.exp(-config.decay_rate * occupancy)
    # math.pow rather than ``**`` so the result is a float by construction,
    # not a float-or-int-by-operand that reads as Any downstream.
    return math.pow(1.0 + occupancy, -config.decay_rate)


def congested_payoff(
    base_payoff: float,
    occupancy: float,
    config: CongestionConfig | None = None,
) -> float:
    """Base payoff discounted by crowding on the angle.

    ``occupancy`` is the share of players already on the angle, in [0, 1];
    values outside raise ValueError because a share above 1 is a modelling
    error, not an input to extrapolate. The decay curve is applied first and
    the floor second, so the floor is a true lower bound on the residual
    share.
    """
    resolved = config if config is not None else CongestionConfig()
    if not 0.0 <= occupancy <= 1.0:
        raise ValueError(f"occupancy must be in [0, 1], got {occupancy}")
    share = max(_residual_share(occupancy, resolved), resolved.floor)
    return base_payoff * share


class CongestionGame:
    """A congestion game over a shared angle space.

    Each player chooses one angle; an angle's payoff to everyone on it decays
    with headcount. ``angle_sets`` optionally restricts which angles a given
    player may play (a creator whose avatar cannot credibly run a genre has a
    restricted set); by default every player may play every angle.
    """

    def __init__(
        self,
        players: Sequence[str],
        angle_payoffs: dict[str, float],
        config: CongestionConfig | None = None,
        angle_sets: dict[str, Sequence[str]] | None = None,
    ) -> None:
        if not players:
            raise ValueError("players must be non-empty")
        if len(set(players)) != len(players):
            raise ValueError("players must be unique")
        if not angle_payoffs:
            raise ValueError("angle_payoffs must be non-empty")
        self.players: list[str] = list(players)
        self.angle_payoffs: dict[str, float] = dict(angle_payoffs)
        self._config = config if config is not None else CongestionConfig()
        default = list(self.angle_payoffs)
        self.angle_sets: dict[str, list[str]] = {}
        for player in self.players:
            allowed = list(angle_sets[player]) if angle_sets and player in angle_sets else default
            if not allowed:
                raise ValueError(f"player {player!r} has an empty angle set")
            unknown = [angle for angle in allowed if angle not in self.angle_payoffs]
            if unknown:
                raise ValueError(f"player {player!r} references unknown angles {unknown}")
            self.angle_sets[player] = allowed

    # --- payoffs -------------------------------------------------------------

    def _occupancy(self, profile: dict[str, str], angle: str) -> float:
        headcount = sum(1 for chosen in profile.values() if chosen == angle)
        return (headcount - 1) / len(self.players) if headcount > 0 else 0.0

    def payoff(self, profile: dict[str, str], player: str) -> float:
        """Player's payoff under ``profile``, given who else shares its angle."""
        return congested_payoff(
            self.angle_payoffs[profile[player]],
            self._occupancy(profile, profile[player]),
            self._config,
        )

    def best_response(self, profile: dict[str, str], player: str) -> str:
        """Player's payoff-maximising angle, holding everyone else fixed.

        Ties break toward the alphabetically first angle so the dynamics are
        reproducible; in a congestion game ties are common (any uncontested
        angle pays its base) and an arbitrary but deterministic rule beats a
        randomised one for auditability.
        """
        best_angle: str | None = None
        best_value = -float("inf")
        for angle in sorted(self.angle_sets[player]):
            trial = dict(profile)
            trial[player] = angle
            value = congested_payoff(
                self.angle_payoffs[angle],
                self._occupancy(trial, angle),
                self._config,
            )
            if value > best_value:
                best_value = value
                best_angle = angle
        assert best_angle is not None
        return best_angle

    def potential(self, profile: dict[str, str]) -> float:
        """Rosenthal's potential for ``profile``.

        ``sum over angles of sum_{k=1..headcount} c_a(k)`` where ``c_a(k)`` is
        the congested payoff of the k-th player to arrive. The defining
        property, and the reason this function and not the welfare sum, is
        that a unilateral deviation changes the potential by exactly the
        deviator's payoff change, so improving deviations strictly increase
        the potential and local maxima are Nash equilibria. Rosenthal (1973).
        """
        total = 0.0
        for angle in self.angle_payoffs:
            headcount = sum(1 for chosen in profile.values() if chosen == angle)
            for kth in range(1, headcount + 1):
                occupancy = (kth - 1) / len(self.players)
                total += congested_payoff(
                    self.angle_payoffs[angle], occupancy, self._config
                )
        return total

    # --- solution concepts -----------------------------------------------------

    def best_response_dynamics(
        self,
        initial_profile: dict[str, str],
        max_iters: int = 1000,
    ) -> dict[str, str]:
        """Iterate strict improvements to a pure Nash equilibrium.

        Convergence is guaranteed rather than hoped for, and the reason is the
        potential: each applied deviation strictly increases it, the potential
        is bounded above (the floor bounds every term and there are finitely
        many profiles), so the walk uphill must end. It ends when no
        player has a strict improvement left, which is the definition of a
        pure Nash equilibrium. ``max_iters`` is a belt-and-braces bound on the
        number of applied deviations, not part of the guarantee.
        """
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        profile = dict(initial_profile)
        unknown_players = [p for p in profile if p not in self.angle_sets]
        if unknown_players:
            raise ValueError(f"profile contains unknown players {unknown_players}")
        missing = [p for p in self.players if p not in profile]
        if missing:
            raise ValueError(f"profile is missing players {missing}")

        for _ in range(max_iters):
            moved = False
            for player in self.players:
                current = self.payoff(profile, player)
                target = self.best_response(profile, player)
                trial = dict(profile)
                trial[player] = target
                if self.payoff(trial, player) > current:
                    profile = trial
                    moved = True
            if not moved:
                break
        return profile

    def is_nash(self, profile: dict[str, str]) -> bool:
        """True when no player gains by a unilateral deviation.

        Weak preferences count as stable: a player indifferent to moving is
        not a deviation the dynamics would take, and in a potential game such
        a profile is still a local maximum of the potential.
        """
        for player in self.players:
            current = self.payoff(profile, player)
            trial = dict(profile)
            trial[player] = self.best_response(profile, player)
            if self.payoff(trial, player) > current:
                return False
        return True


def crowding_adjusted_ranking(
    angles_with_base_payoff: Sequence[tuple[str, float]],
    occupancy: dict[str, float],
    config: CongestionConfig | None = None,
) -> list[tuple[str, float]]:
    """Rank angles by congested rather than raw payoff.

    The practical entry point: given each angle's uncontested payoff and the
    current occupancy of each, return the angles ordered by what they actually
    pay under crowding. ``occupancy`` entries default to 0.0 for angles absent
    from the mapping, so partial field data still ranks cleanly. Ties break
    alphabetically for determinism.
    """
    resolved = config if config is not None else CongestionConfig()
    scored = [
        (angle, congested_payoff(base, occupancy.get(angle, 0.0), resolved))
        for angle, base in angles_with_base_payoff
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))
