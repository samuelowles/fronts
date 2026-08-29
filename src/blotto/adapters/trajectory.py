"""The trajectory store: recorded history as append-only JSONL.

The trajectory is the only training signal in the closed-deck setting (see
``Trajectory`` in ``blotto.game.types``), which makes this file the most
consequential persistence in the repository: every model ever synthesised is
fit against what lands here, and a silently corrupted line becomes a
confidently wrong prior rather than a crash.

Two properties are therefore non-negotiable and both are tested:

* Writes are ATOMIC. A rewrite (``save``, ``mark_settled``) lands via a
  temp file in the same directory, flush, ``os.fsync``, then
  ``os.replace``. A process that dies mid-write leaves the previous file
  intact and a stray ``*.tmp`` beside it -- never a half-written trajectory,
  because a truncated history is indistinguishable from a real one once the
  next model is trained on it.
* ``Trajectory.chance`` round-trips. The recorded chance sequence is what
  makes a transition test a measurement rather than a coin flip (the
  docstring on the field says exactly how badly this goes wrong: the
  reference model scored 0.20 against its own trajectories when replay
  re-rolled the dice). Losing it on save does not fail any assertion at
  save time; it degrades every accuracy number read afterwards, which is
  precisely the kind of bug this module exists to make impossible.

Line kinds: ``header`` (account, notes, chance), ``step`` (move with its
observation, from ``save``), ``move`` and ``observation`` (separate lines,
from live recording). The split exists because live history is written
backwards -- a move is appended the moment it ships, its observation days
later -- and the join on ``utm_content`` happens at load, in one place,
rather than being re-implemented by every reader.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime

try:  # Python 3.11+; see composio_io for why the 3.10 fallback exists.
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017
from pathlib import Path

from blotto.game.action_space import ActionCodec
from blotto.game.priors import REPORTING_LAG_HOURS
from blotto.game.types import (
    ActionKey,
    Archetype,
    Move,
    Observation,
    Platform,
    Publish,
    Step,
    Trajectory,
)

__all__ = [
    "TrajectoryStore",
    "TrajectoryStats",
    "COLD_START_MIN_SETTLED",
    "COLD_START_MIN_ARCHETYPES",
    "COLD_START_MIN_PLATFORMS",
]

_REPORTING_LAG_HOURS_HIGH = REPORTING_LAG_HOURS.high
"""Observations flip to settled only after the LONGEST measured lag has
cleared. Using the low end would mark data settled while it can still
move, which is peeking with extra steps; the high end is the reading the
evidence gates already take (``settled_count`` drops partials entirely)."""

# The cold-start floor, from docs/OPERATING.md: "30 published items with
# settled metrics, spanning at least two archetypes and two platforms.
# Below that, synthesis will produce a model that fits your history and
# predicts nothing." Stated as constants here so `stats()` reports the
# floor programmatically and the CLI can print readiness rather than
# leaving the operator to check three numbers against a document.
COLD_START_MIN_SETTLED = 30
COLD_START_MIN_ARCHETYPES = 2
COLD_START_MIN_PLATFORMS = 2


@dataclass(frozen=True, slots=True)
class TrajectoryStats:
    """Counts an operator needs before trusting synthesis, plus the
    cold-start verdict. ``ready_for_synthesis`` encodes the whole floor at
    once because the individual counts only matter together: 30 settled
    items on one archetype is a narrow game, not a cold start cleared."""

    moves: int
    settled_observations: int
    partial_observations: int
    unmatched_utm: int
    archetypes: int
    platforms: int
    ready_for_synthesis: bool


def _normalise(moment: datetime) -> datetime:
    """Compare timestamps in one frame: naive stays naive, aware becomes
    naive UTC. Mixing the two raises instead of comparing, and would
    surface as an ingest crash far from the timestamp that caused it."""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(UTC).replace(tzinfo=None)


class TrajectoryStore:
    """JSONL-backed trajectory storage for one account."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._codec = ActionCodec()

    # -- reading --------------------------------------------------------------

    def _records(self) -> list[dict]:
        """Parse every line. A truncated FINAL line is skipped -- that is
        the one a crash mid-append can leave -- while a malformed line in
        the middle raises, because interior corruption means the file was
        rewritten badly and every reading after it is suspect."""
        if not self.path.exists():
            return []
        records: list[dict] = []
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.readlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue
                raise
        return records

    def load(self) -> Trajectory:
        """Rebuild the trajectory, joining observations to moves on utm.

        The join takes the LATEST observation for a utm: analytics pulls
        refresh numbers, and the later line is the fresher reading.
        Observations whose utm matches no recorded move are dropped from
        the trajectory (there is nothing to attach them to) but remain
        visible to ``stats()`` as ``unmatched_utm`` -- an unmatched id
        usually means the move log predates the utm stamping rule, which is
        worth knowing about rather than silently discarding.
        """
        trajectory = Trajectory()
        position: dict[str, int] = {}
        for record in self._records():
            kind = record.get("kind")
            if kind == "header":
                trajectory.account = record.get("account", "")
                trajectory.notes = record.get("notes", "")
                trajectory.chance = [
                    ActionKey(key) for key in record.get("chance", [])
                ]
            elif kind == "step":
                move = self._codec.decode(record["action"])
                observation = _decode_observation(record.get("observation"))
                trajectory.steps.append(
                    Step(move=move, observation=observation)
                )
                _index_position(position, move, len(trajectory.steps) - 1)
            elif kind == "move":
                move = self._codec.decode(record["action"])
                trajectory.steps.append(Step(move=move, observation=None))
                _index_position(position, move, len(trajectory.steps) - 1)
            elif kind == "observation":
                observation = _decode_observation(record.get("observation"))
                if observation is None:
                    continue
                index = position.get(observation.utm_content)
                if index is not None:
                    trajectory.steps[index] = replace(
                        trajectory.steps[index], observation=observation
                    )
        return trajectory

    def settled_only(self) -> Trajectory:
        """The trajectory reduced to steps whose observation has settled.

        Steps with partial or absent observations are DROPPED rather than
        blanked: a blanked step keeps the move visible to test generation
        while hiding the number, and anything that keeps a provisional
        number one refactor away from a test case is a footgun. The honest
        cost is a shorter action history, which ``generate`` in
        ``blotto.cwm.tests_from_traj`` already tolerates -- it slices
        prefixes by position, not by continuity.
        """
        trajectory = self.load()
        trajectory.steps = [
            step
            for step in trajectory.steps
            if step.observation is not None and not step.observation.is_partial
        ]
        return trajectory

    def stats(self) -> TrajectoryStats:
        """Counts plus the cold-start verdict from docs/OPERATING.md."""
        trajectory = self.load()
        settled = 0
        partial = 0
        archetypes: set[Archetype] = set()
        platforms: set[Platform] = set()
        seen_utms: set[str] = set()
        for step in trajectory.steps:
            if not isinstance(step.move, Publish):
                continue
            seen_utms.add(step.move.utm_content)
            if step.observation is None:
                continue
            if step.observation.is_partial:
                partial += 1
                continue
            settled += 1
            archetypes.add(step.move.archetype)
            platforms.add(step.move.platform)
        unmatched = sum(
            1
            for record in self._records()
            if record.get("kind") == "observation"
            and record["observation"]["utm_content"] not in seen_utms
        )
        ready = (
            settled >= COLD_START_MIN_SETTLED
            and len(archetypes) >= COLD_START_MIN_ARCHETYPES
            and len(platforms) >= COLD_START_MIN_PLATFORMS
        )
        return TrajectoryStats(
            moves=len(trajectory.steps),
            settled_observations=settled,
            partial_observations=partial,
            unmatched_utm=unmatched,
            archetypes=len(archetypes),
            platforms=len(platforms),
            ready_for_synthesis=ready,
        )

    # -- writing --------------------------------------------------------------

    def append_move(self, move: Move, posted_at: str) -> None:
        """Record one move the moment it ships, before any data exists.

        Appends are single lines, flushed and fsynced: a crash between two
        appends loses nothing, and a crash inside one leaves a truncated
        final line that ``_records`` recognises and skips.
        """
        self._append(
            {
                "kind": "move",
                "action": self._codec.encode(move),
                "posted_at": posted_at,
            }
        )

    def attach_observation(self, observation: Observation) -> None:
        """Record one observation; the join to its move happens at load.

        The join key is ``utm_content`` and nothing else -- which is why an
        untracked publish is illegal (``TRACKED_OUTPUT`` in the legality
        engine): a move whose observation cannot be matched is a move the
        system cannot learn from.
        """
        self._append({"kind": "observation", "observation": asdict(observation)})

    def save(self, trajectory: Trajectory) -> None:
        """Replace the file with ``trajectory``, atomically.

        Round-trips ``chance``: the header line carries the full recorded
        sequence, so a trajectory saved and reloaded keeps the exact branch
        its transitions were recorded under.
        """
        lines: list[str] = [
            json.dumps(
                {
                    "kind": "header",
                    "account": trajectory.account,
                    "notes": trajectory.notes,
                    "chance": [str(key) for key in trajectory.chance],
                }
            )
            + "\n"
        ]
        for step in trajectory.steps:
            lines.append(
                json.dumps(
                    {
                        "kind": "step",
                        "action": self._codec.encode(step.move),
                        "observation": (
                            asdict(step.observation)
                            if step.observation is not None
                            else None
                        ),
                    }
                )
                + "\n"
            )
        self._atomic_write(lines)

    def mark_settled(self, now: datetime) -> int:
        """Flip stale partials to settled; return how many flipped.

        Idempotent, and a no-op rewrite when nothing changed -- rewriting
        an unchanged file is a needless window for the very corruption the
        atomic write exists to prevent. Timestamps are compared in one
        frame (see ``_normalise``); posted_at is day-granular by the game
        layer's convention, so a post flips on the first ``now`` at least
        ``REPORTING_LAG_HOURS.high`` past its posting DAY.
        """
        when = _normalise(now)
        changed = 0
        lines = [line for line in self._raw_lines() if line.strip()]
        rewritten: list[str] = []
        for line in lines:
            record = json.loads(line)
            observation = record.get("observation")
            if (
                record.get("kind") in ("observation", "step")
                and observation
                and observation.get("is_partial")
            ):
                posted = _normalise(
                    datetime.fromisoformat(str(observation["posted_at"]))
                )
                if (when - posted).total_seconds() >= _REPORTING_LAG_HOURS_HIGH * 3600:
                    observation["is_partial"] = False
                    changed += 1
                    line = json.dumps(record) + "\n"
            rewritten.append(line)
        if changed:
            self._atomic_write(rewritten)
        return changed

    # -- plumbing --------------------------------------------------------------

    def _raw_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as handle:
            return handle.readlines()

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _atomic_write(self, lines: list[str]) -> None:
        """Temp file in the SAME directory, flush, fsync, ``os.replace``.

        Same-directory is the load-bearing detail: ``os.replace`` is atomic
        within a filesystem, and a temp file on another volume makes the
        final rename a copy -- non-atomic, and exactly the window a crash
        uses to truncate the history. On any failure the temp file is
        removed and the previous file stands untouched.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle_fd, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f"{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt between write
            # and replace must clean up too, or every ^C litters a temp file.
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
            raise


def _index_position(position: dict[str, int], move: Move, index: int) -> None:
    """Record where each publish sits so later observation lines can find
    it. Non-publish moves have no utm and never receive observations."""
    if isinstance(move, Publish):
        position[move.utm_content] = index


def _decode_observation(raw: object) -> Observation | None:
    if raw is None:
        return None
    return Observation(
        utm_content=str(raw["utm_content"]),
        posted_at=str(raw["posted_at"]),
        observed_at=str(raw["observed_at"]),
        impressions=int(raw.get("impressions", 0)),
        reach=int(raw.get("reach", 0)),
        hook_rate=float(raw.get("hook_rate", 0.0)),
        hold_rate=float(raw.get("hold_rate", 0.0)),
        saves=int(raw.get("saves", 0)),
        shares=int(raw.get("shares", 0)),
        comments=int(raw.get("comments", 0)),
        profile_visits=int(raw.get("profile_visits", 0)),
        link_clicks=int(raw.get("link_clicks", 0)),
        attributed_conversions=int(raw.get("attributed_conversions", 0)),
        attribution_coverage=float(raw.get("attribution_coverage", 1.0)),
        incrementality=float(raw.get("incrementality", 1.0)),
        is_partial=bool(raw.get("is_partial", False)),
        dark_social_estimate=float(raw.get("dark_social_estimate", 0.0)),
    )
