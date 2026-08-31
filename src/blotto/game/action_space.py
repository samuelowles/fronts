"""The action space: structured moves in, opaque strings out.

``ActionKey`` is the only representation the synthesised world model ever
sees -- the paper defines ``Action = str`` and we honour that at the CWM
boundary. This module is the single crossing point (``blotto.game.types``
says so explicitly), which means every property the strings must have --
stability, determinism, collision freedom, human readability -- has to be
established here and nowhere else.

The encoding is pipe-delimited position for the enum dimensions and
``key=value`` for the free dimensions, e.g.::

    publish|tiktok|carousel|anti_hero_rant|anger_injustice|tribal_identity|
    pattern_interrupt|avatar=solo_founder|cta=start_trial|
    claim=first_party_proof|angle=launch_theater|utm=c0412
    scale|angle=launch_theater|factor=1.20
    kill|angle=launch_theater|reason=hook_burnout
    hold

Free-text fields may not contain ``|``: allowing it would make the encoding
ambiguous, and an ambiguous action space is a corrupted replay. Scale factors
are rounded to two decimals on encode, so ``1.2`` and ``1.20`` produce the
same key -- the key is canonical, and round-tripping is exact for any move
whose factor is already at two decimals.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeVar

from blotto.game.types import (
    ActionKey,
    Archetype,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    Hold,
    HookFamily,
    Kill,
    Move,
    MoveKind,
    Platform,
    Publish,
    Scale,
    SemanticTier,
)

__all__ = [
    "ActionDecodeError",
    "ActionCodec",
    "AngleRegistry",
    "enumerate_publishes",
    "utm_content_id",
    "restamp_unique",
]


class ActionDecodeError(ValueError):
    """Raised when a string cannot be decoded into a Move.

    A synthesised model will hand us malformed keys; failing loudly here is
    what keeps a typo from silently becoming a different legal move."""


# Free-text Publish/Scale/Kill fields, in encode order. ``avatar`` and
# ``angle`` are chosen by the planner, ``reason`` by the operator's policy.
_PUBLISH_PREFIX = 7
_PUBLISH_KEYS: tuple[str, ...] = ("avatar", "cta", "claim", "angle", "utm")
_SCALE_KEYS: tuple[str, ...] = ("angle", "factor")
_KILL_KEYS: tuple[str, ...] = ("angle", "reason")


def _check_free_text(field: str, value: str) -> None:
    if "|" in value:
        raise ValueError(
            f"{field} must not contain '|': {value!r} -- the action encoding "
            "is pipe-delimited and an embedded pipe would make it ambiguous"
        )


def _parse_kv(token: str, expected: Sequence[str]) -> dict[str, str]:
    parts = token.split("=", 1)
    # The KEY must be present and known; the VALUE may be empty. ``encode``
    # emits ``utm=`` for a Publish with an empty ``utm_content`` (only pipes
    # are forbidden in free text), so a decoder that rejected empty values
    # would refuse to read back what the codec itself writes -- and would
    # bury the ``TRACKED_OUTPUT`` refusal (the cited system invariant that
    # exists precisely for the untracked publish) behind a generic decode
    # error instead of the rule, reason and source an operator is owed.
    if len(parts) != 2 or not parts[0]:
        raise ActionDecodeError(f"malformed key=value token: {token!r}")
    key, value = parts
    if key not in expected:
        raise ActionDecodeError(f"unexpected field {key!r} in {token!r}")
    return {key: value}


_E = TypeVar("_E", bound=Enum)


def _enum_or_raise(enum_cls: type[_E], raw: str, what: str) -> _E:
    try:
        return enum_cls(raw)
    except ValueError as exc:
        raise ActionDecodeError(f"unknown {what}: {raw!r}") from exc


class ActionCodec:
    """The only thing permitted to convert between ``Move`` and ``ActionKey``.

    Stateless and deterministic: the same move always yields the same key on
    every platform and every run, which is what makes recorded trajectories
    replayable and unit-testable.
    """

    def encode(self, move: Move) -> ActionKey:
        """Return the canonical key for ``move``."""
        if isinstance(move, Publish):
            return self._encode_publish(move)
        if isinstance(move, Scale):
            return self._encode_scale(move)
        if isinstance(move, Kill):
            return self._encode_kill(move)
        if isinstance(move, Hold):
            return ActionKey(MoveKind.HOLD.value)
        raise TypeError(f"not a Move: {move!r}")

    def decode(self, key: ActionKey | str) -> Move:
        """Return the move encoded by ``key``.

        Raises ``ActionDecodeError`` on anything malformed -- wrong arity,
        unknown enum member, missing field, or unknown kind.
        """
        tokens = key.split("|")
        kind = tokens[0]
        if kind == MoveKind.PUBLISH.value:
            return self._decode_publish(tokens)
        if kind == MoveKind.SCALE.value:
            return self._decode_scale(tokens)
        if kind == MoveKind.KILL.value:
            return self._decode_kill(tokens)
        if kind == MoveKind.HOLD.value and len(tokens) == 1:
            return Hold()
        raise ActionDecodeError(f"unknown or malformed action kind: {key!r}")

    # -- Publish -----------------------------------------------------------

    def _encode_publish(self, move: Publish) -> ActionKey:
        for field, value in (
            ("avatar", move.avatar),
            ("angle", move.angle),
            ("utm_content", move.utm_content),
        ):
            _check_free_text(field, value)
        head = "|".join(
            (
                MoveKind.PUBLISH.value,
                move.platform.value,
                move.format.value,
                move.archetype.value,
                move.vector.value,
                move.semantic_tier.value,
                move.hook.value,
            )
        )
        tail = "|".join(
            (
                f"avatar={move.avatar}",
                f"cta={move.cta_mode.value}",
                f"claim={move.claim_class.value}",
                f"angle={move.angle}",
                f"utm={move.utm_content}",
            )
        )
        return ActionKey(f"{head}|{tail}")

    def _decode_publish(self, tokens: list[str]) -> Publish:
        if len(tokens) != _PUBLISH_PREFIX + len(_PUBLISH_KEYS):
            raise ActionDecodeError(
                f"publish key must have {_PUBLISH_PREFIX + len(_PUBLISH_KEYS)} "
                f"fields, got {len(tokens)}: {'|'.join(tokens)!r}"
            )
        fields: dict[str, str] = {}
        for token in tokens[_PUBLISH_PREFIX:]:
            entry = _parse_kv(token, _PUBLISH_KEYS)
            key = next(iter(entry))
            if key in fields:
                raise ActionDecodeError(f"duplicate field {key!r} in {token!r}")
            fields.update(entry)
        if tuple(fields) != _PUBLISH_KEYS:
            raise ActionDecodeError(
                f"publish fields out of order: expected {_PUBLISH_KEYS}"
            )
        return Publish(
            platform=_enum_or_raise(Platform, tokens[1], "platform"),
            format=_enum_or_raise(Format, tokens[2], "format"),
            archetype=_enum_or_raise(Archetype, tokens[3], "archetype"),
            vector=_enum_or_raise(EmotionalVector, tokens[4], "vector"),
            semantic_tier=_enum_or_raise(SemanticTier, tokens[5], "semantic tier"),
            hook=_enum_or_raise(HookFamily, tokens[6], "hook family"),
            avatar=fields["avatar"],
            cta_mode=_enum_or_raise(CtaMode, fields["cta"], "cta mode"),
            claim_class=_enum_or_raise(ClaimClass, fields["claim"], "claim class"),
            angle=fields["angle"],
            utm_content=fields["utm"],
        )

    # -- Scale / Kill / Hold -------------------------------------------------

    def _encode_scale(self, move: Scale) -> ActionKey:
        _check_free_text("angle", move.angle)
        # Two decimals keeps keys canonical: 1.2 and 1.20 are the same action.
        return ActionKey(
            f"{MoveKind.SCALE.value}|angle={move.angle}|factor={move.factor:.2f}"
        )

    def _decode_scale(self, tokens: list[str]) -> Scale:
        if len(tokens) != 1 + len(_SCALE_KEYS):
            raise ActionDecodeError(f"malformed scale key: {'|'.join(tokens)!r}")
        fields: dict[str, str] = {}
        for token in tokens[1:]:
            fields.update(_parse_kv(token, _SCALE_KEYS))
        if tuple(fields) != _SCALE_KEYS:
            raise ActionDecodeError(
                f"scale fields out of order: expected {_SCALE_KEYS}"
            )
        try:
            factor = float(fields["factor"])
        except ValueError as exc:
            raise ActionDecodeError(
                f"non-numeric scale factor: {fields['factor']!r}"
            ) from exc
        return Scale(angle=fields["angle"], factor=factor)

    def _encode_kill(self, move: Kill) -> ActionKey:
        _check_free_text("angle", move.angle)
        _check_free_text("reason", move.reason)
        return ActionKey(
            f"{MoveKind.KILL.value}|angle={move.angle}|reason={move.reason}"
        )

    def _decode_kill(self, tokens: list[str]) -> Kill:
        if len(tokens) != 1 + len(_KILL_KEYS):
            raise ActionDecodeError(f"malformed kill key: {'|'.join(tokens)!r}")
        fields: dict[str, str] = {}
        for token in tokens[1:]:
            fields.update(_parse_kv(token, _KILL_KEYS))
        if tuple(fields) != _KILL_KEYS:
            raise ActionDecodeError(
                f"kill fields out of order: expected {_KILL_KEYS}"
            )
        return Kill(angle=fields["angle"], reason=fields["reason"])


@dataclass(frozen=True, slots=True)
class AngleRecord:
    """One angle: what it is, and what class of claim it is allowed to carry.

    ``required_claim_class`` is the claim the angle *promises* the audience --
    launch theatre without first-party proof is just noise -- so the registry
    doubles as a coarse quality floor on the contested resource.
    """

    angle_id: str
    description: str
    required_claim_class: ClaimClass


class AngleRegistry:
    """Angle id -> record. Angles are minted at runtime by the planner, not
    seeded here, because which angles exist is a property of the campaign
    under way, not of the game.

    Congestion -- the field's share of an angle -- is computed per angle id,
    which is why the id must be a stable string rather than an object
    identity: it survives serialisation into an ``ActionKey`` and back.
    """

    def __init__(self) -> None:
        self._records: dict[str, AngleRecord] = {}

    def register(
        self,
        angle_id: str,
        description: str,
        required_claim_class: ClaimClass,
    ) -> AngleRecord:
        """Idempotently register an angle and return its record."""
        record = AngleRecord(angle_id, description, required_claim_class)
        self._records[angle_id] = record
        return record

    def get(self, angle_id: str) -> AngleRecord | None:
        return self._records.get(angle_id)

    def describe(self, angle_id: str) -> str:
        record = self._records.get(angle_id)
        return record.description if record else ""

    def requirement(self, angle_id: str) -> ClaimClass | None:
        record = self._records.get(angle_id)
        return record.required_claim_class if record else None

    def __contains__(self, angle_id: str) -> bool:
        return angle_id in self._records

    def __len__(self) -> int:
        return len(self._records)


def utm_content_id(move: Move, salt: str) -> str:
    """Deterministic 8-hex-char id from the canonical key plus a salt.

    This is the join key between a move and its eventual ``Observation``:
    analytics reports the id, the game records the move, and the two meet
    here. Short because platforms truncate utm_content; hashed because two
    creatives that differ only in hook must never share an id -- that would
    quietly merge their learning.
    """
    codec = ActionCodec()
    digest = hashlib.sha256(f"{codec.encode(move)}|{salt}".encode()).hexdigest()
    return digest[:8]


def restamp_unique(move: Publish, index: int) -> Publish:
    """Re-stamp a catalogue ``Publish`` with a per-step unique utm.

    The legal-action set is a fixed catalogue, so a policy that picks the
    same entry twice publishes two distinct posts under one utm -- and utm
    is the join key between a move and its observation. The posts then
    collapse to one row, later metrics overwrite earlier ones, and a
    transition test compares post A's prediction against post B's numbers.
    That read as a 3x modelling error and was a bookkeeping collision;
    every episode recorder calls this before appending a Publish.
    """
    return replace(move, utm_content=utm_content_id(move, salt=str(index)))


def enumerate_publishes(
    *,
    platforms: Iterable[Platform],
    formats: Iterable[Format],
    archetypes: Iterable[Archetype],
    vectors: Iterable[EmotionalVector],
    tiers: Iterable[SemanticTier],
    hooks: Iterable[HookFamily],
    avatars: Iterable[str],
    cta_modes: Iterable[CtaMode],
    claim_classes: Iterable[ClaimClass],
    angles: Iterable[str],
    limit: int,
    seed: int,
    salt: str = "enumerate_publishes",
) -> Iterator[Publish]:
    """Lazily yield up to ``limit`` candidate publishes from the cross product.

    The full product is combinatorially large (seven platforms x five formats
    x four archetypes x ... is already six figures before avatars), and a
    planner needs a sample, not a census. The generator never materialises
    more than ``limit`` moves: when the product fits inside ``limit`` it
    enumerates it in deterministic index order, otherwise it samples flat
    indices without replacement from ``random.Random(seed)`` -- same seed,
    same candidates, every run.

    Each yielded publish carries a utm id minted from its own (pre-utm) key
    and ``salt``, so candidate generation is itself reproducible.
    """
    dims: tuple[
        list[Platform],
        list[Format],
        list[Archetype],
        list[EmotionalVector],
        list[SemanticTier],
        list[HookFamily],
        list[str],
        list[CtaMode],
        list[ClaimClass],
        list[str],
    ] = (
        list(platforms),
        list(formats),
        list(archetypes),
        list(vectors),
        list(tiers),
        list(hooks),
        list(avatars),
        list(cta_modes),
        list(claim_classes),
        list(angles),
    )
    if limit <= 0 or any(len(dim) == 0 for dim in dims):
        return

    total = 1
    for dim in dims:
        total *= len(dim)

    def build(flat_index: int) -> Publish:
        """Decode a flat index into one candidate Publish.

        The flat encoding counts the LAST dimension fastest, so decoding
        peels dimensions off back-to-front; each pick is typed by its own
        dimension rather than flowing through one untyped list."""
        remaining = flat_index
        angle = dims[9][remaining % len(dims[9])]
        remaining //= len(dims[9])
        claim_class = dims[8][remaining % len(dims[8])]
        remaining //= len(dims[8])
        cta_mode = dims[7][remaining % len(dims[7])]
        remaining //= len(dims[7])
        avatar = dims[6][remaining % len(dims[6])]
        remaining //= len(dims[6])
        hook = dims[5][remaining % len(dims[5])]
        remaining //= len(dims[5])
        semantic_tier = dims[4][remaining % len(dims[4])]
        remaining //= len(dims[4])
        vector = dims[3][remaining % len(dims[3])]
        remaining //= len(dims[3])
        archetype = dims[2][remaining % len(dims[2])]
        remaining //= len(dims[2])
        format_ = dims[1][remaining % len(dims[1])]
        remaining //= len(dims[1])
        platform = dims[0][remaining % len(dims[0])]
        stub = Publish(
            platform=platform,
            format=format_,
            archetype=archetype,
            vector=vector,
            semantic_tier=semantic_tier,
            hook=hook,
            avatar=avatar,
            cta_mode=cta_mode,
            claim_class=claim_class,
            angle=angle,
            utm_content="",
        )
        return replace(stub, utm_content=utm_content_id(stub, salt))

    if total <= limit:
        for index in range(total):
            yield build(index)
        return

    rng = random.Random(seed)
    seen: set[int] = set()
    # Guard against pathological collision churn when total barely exceeds
    # limit; in practice sampling without replacement terminates long before.
    max_attempts = 1000 * limit + 10_000
    attempts = 0
    while len(seen) < limit and attempts < max_attempts:
        attempts += 1
        index = rng.randrange(total)
        if index in seen:
            continue
        seen.add(index)
        yield build(index)
