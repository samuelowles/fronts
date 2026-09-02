"""Shared fixtures for the end-to-end suite (``tests/test_e2e.py``).

Every fixture here exists to keep the e2e scenarios offline and honest: a
seeded reference trajectory instead of a live account, a store in ``tmp_path``
instead of the operator's real history, a ``RecordedClient`` over a fixture
written into ``tmp_path`` instead of a provider, the dry-run adapter instead
of Composio, and a socket blocker that turns any silent network reach into an
immediate, loudly-named failure.

The unit suites (``test_cwm`` and friends) do not use this file; they keep
their own module-level helpers, and nothing here changes how they collect.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

import pytest

from fronts.adapters.composio_io import DryRunAdapter
from fronts.adapters.trajectory import TrajectoryStore
from fronts.cwm.llm import RecordedClient, prompt_key
from fronts.cwm.reference import ReferenceConfig, ReferenceWorldModel
from fronts.game.types import ActionKey, State, Trajectory
from fronts.protocols import CodeWorldModel


def rotating_publish_policy(model: CodeWorldModel, state: State) -> ActionKey:
    """A DIFFERENT catalogue publish each day, so a generated history spans
    more than one archetype and platform -- the shape the cold-start floor
    asks about. Still legal by construction: legality is a precondition on
    planning, so even a dumb policy never proposes an illegal move."""
    legal = model.get_legal_actions(state)
    publishes = [action for action in legal if str(action).startswith("publish")]
    if not publishes:
        return legal[0]
    step = state.get("hidden", {}).get("step", 0)
    return publishes[int(step) % len(publishes)]


@pytest.fixture
def reference_trajectory() -> Callable[..., Trajectory]:
    """A seeded synthetic history from ``ReferenceWorldModel``.

    A factory rather than a plain trajectory so each scenario picks its own
    horizon (the golden path wants a short cheap episode; a report wants a
    long one) while keeping the seed fixed -- a history that changed between
    runs would make every downstream assertion unrepeatable. The model
    re-stamps each post's utm, so utms are unique and the observation join
    is well posed by construction.
    """

    def _make(horizon: int = 12, seed: int = 7, policy_seed: int = 7) -> Trajectory:
        config = ReferenceConfig(horizon=horizon, seed=seed)
        model = ReferenceWorldModel(config)
        return model.generate_trajectory(
            rotating_publish_policy, horizon, random.Random(policy_seed)
        )

    return _make


@pytest.fixture
def tmp_store(tmp_path: Path) -> TrajectoryStore:
    """A trajectory store inside pytest's ``tmp_path``: the loop can write
    freely because nothing here is anyone's real history."""
    return TrajectoryStore(tmp_path / "history.jsonl")


@pytest.fixture
def recorded_client(
    tmp_path: Path,
) -> Callable[[dict[tuple[str, str], str]], RecordedClient]:
    """Build a ``RecordedClient`` over a fixture written into ``tmp_path``.

    The mapping is (system, user) -> response, keyed exactly as the recording
    client would key it, so a scenario can hand the synthesis loop a response
    it controls (a hostile one, in the sandbox-escape scenario) while the
    loop still exercises the real prompt-build / extract / sandbox path. A
    prompt the mapping did not anticipate raises ``FixtureMiss``, which is
    the correct loud failure, not a plausible fallback.
    """

    def _make(responses: dict[tuple[str, str], str]) -> RecordedClient:
        path = tmp_path / "recorded_responses.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for (system, user), response in responses.items():
                handle.write(
                    json.dumps({"key": prompt_key(system, user), "response": response})
                    + "\n"
                )
        return RecordedClient(path)

    return _make


@pytest.fixture
def dry_adapter() -> DryRunAdapter:
    """The platform adapter the CLI uses without ``--live``: records intents,
    performs no I/O, invents no metrics. Coverage and incrementality are the
    worked-example measurements so ingest paths stay open."""
    return DryRunAdapter(attribution_coverage=0.72, incrementality=0.70)


def _refuse_sockets(*args: object, **kwargs: object) -> None:
    raise RuntimeError("network access attempted")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``socket.socket`` itself raise.

    Not a stub that returns fake data -- an exception whose message names the
    crime. Any code path that silently reaches for the network fails the test
    at the exact call site with ``RuntimeError: network access attempted``,
    which is precisely the failure this fixture exists to make obvious.
    """
    monkeypatch.setattr("socket.socket", _refuse_sockets)
