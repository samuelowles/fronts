"""Platform I/O through Composio's unified API, and nothing else.

THE DESIGN RULE, and the reason this module exists in this shape: every call
into Composio goes through ONE private method, ``_execute``. Nothing else in
this codebase may touch the SDK -- not the CLI, not the trajectory store, not
a test, not another adapter. Tool slugs drift between provider versions and
between Composio releases; when one renames ``TWITTER_CREATION_OF_A_POST``
out from under us, a single chokepoint makes that a one-line fix inside one
method, instead of archaeology across every call site that ever posted. If
you are adding a second entry point into the SDK, you are undoing the module.

The SDK is imported lazily, inside ``_execute``, for the same reason the LLM
clients in ``blotto.cwm.llm`` import lazily: the core must import and run on
a machine with no network and no extras installed, and an adapter that
breaks ``import blotto`` when ``composio`` is absent is an adapter nobody
can test offline.

Verified SDK surface (everything beyond this is UNVERIFIED, see
``ToolkitRegistry``)::

    from composio import Composio
    composio = Composio(api_key=...)          # falls back to COMPOSIO_API_KEY
    result = composio.tools.execute("SLUG", arguments={...}, user_id="user-123")

Two things this module refuses to do, on principle rather than by omission:

* It does not guess attribution coverage or incrementality. No platform API
  can measure either -- coverage is a property of the operator's tracking
  stack and incrementality of their holdout tests -- so both arrive as
  operator-supplied measurements and an ``Observation`` is refused (never
  defaulted) while they are unset. See ``MissingMeasurementError``.
* It never retries a publish. A retried read costs a round trip; a retried
  publish that had in fact landed is a duplicate post, which is an
  irreversible, outward-facing side effect. See ``ComposioAdapter.publish``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:  # Python 3.11+: the canonical spelling, and what ruff's target expects.
    from datetime import UTC
except ImportError:  # Python 3.10, which contributors run despite the floor
    # in pyproject -- the same accommodation the UP042 ignore documents.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 -- this module must import on 3.10 too

from blotto.game.priors import REPORTING_LAG_HOURS
from blotto.game.types import Observation, Platform, Publish

__all__ = [
    "ComposioConfig",
    "ToolkitEntry",
    "ToolkitRegistry",
    "ComposioAdapter",
    "DryRunAdapter",
    "RecordedAdapter",
    "PublishReceipt",
    "MissingMeasurementError",
    "stamp_utm",
]

logger = logging.getLogger(__name__)

_REPORTING_LAG = timedelta(hours=REPORTING_LAG_HOURS.high)
"""How long an observation stays partial. The HIGH end of the measured
24-72h band is the conservative reading: an observation is treated as
provisional until the longest measured lag has cleared, because peeking
inside the window is the specific failure the evidence gates exist to
prevent -- an observation marked settled one hour early can open a gate
that 49-conversion rule was closing."""

_BACKOFF_BASE_SECONDS = 0.5
"""Base for exponential backoff. Small because the first retry should be
near-immediate and each subsequent one doubles; a caller waiting through
three retries on a dead connection spends ~3.5s total, which is inside any
reasonable operator patience and outside any reasonable thundering-herd."""

# Substrings matched against the LOWERCASED exception text (and class name)
# to decide transient-vs-fatal. Deliberately conservative: anything not
# recognisably transient is treated as fatal, because a wrong retry on a
# non-idempotent call is worse than a slow failure on an idempotent one.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "rate limit",
    "ratelimit",
    "502",
    "503",
    "504",
    "unavailable",
)


class MissingMeasurementError(RuntimeError):
    """An Observation was requested before the operator measured the two
    numbers no API can supply.

    Attribution coverage and incrementality are properties of the operator's
    tracking stack and holdout tests respectively. Defaulting either to 1.0
    would silently re-base every reward the model ever learns on the
    assumption that the dashboard sees everything and causes everything --
    the exact confident-direction error docs/OPERATING.md opens with. The
    message points there deliberately.
    """


def stamp_utm(url: str, utm_content: str) -> str:
    """Return ``url`` with ``utm_content`` set to ``utm_content`` in its query.

    The utm id is the ONLY join key between a move and the observation that
    arrives days later (``Publish.utm_content``), so it is stamped at the
    last moment before the call leaves the process -- here, where nothing
    downstream can forget it. An existing ``utm_content`` parameter is
    replaced rather than duplicated (two values for one key is undefined by
    the analytics tools that read it); every other parameter is preserved in
    order.
    """
    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != "utm_content"
    ]
    query.append(("utm_content", utm_content))
    return urlunsplit(parts._replace(query=urlencode(query)))


def _require_measurements(
    attribution_coverage: float | None, incrementality: float | None
) -> None:
    """Refuse to build an Observation without both operator measurements.

    This check runs BEFORE any network call, so a misconfigured operator
    learns it at ingest time rather than after a round of API calls whose
    results could not be used anyway.
    """
    missing = [
        name
        for name, value in (
            ("attribution_coverage", attribution_coverage),
            ("incrementality", incrementality),
        )
        if value is None
    ]
    if missing:
        raise MissingMeasurementError(
            f"cannot build an Observation: {', '.join(missing)} is unset. "
            "Neither is measurable from a platform API -- coverage is a "
            "property of your tracking stack, incrementality of your holdout "
            "tests. Measure them (a holdout, or a post-purchase survey "
            "reconciled against analytics for a month) and set them under "
            "[measurement] in your config. See docs/OPERATING.md, 'The thing "
            "that will bite you'. A plausible default here would silently "
            "corrupt every reward downstream, which is why there is none."
        )


def _parse_timestamp(raw: str | datetime) -> datetime:
    """Parse an ISO timestamp into aware UTC.

    Accepts date-only strings (what the game layer records) and naive
    datetimes (treated as UTC) so comparisons never mix aware and naive --
    which raises rather than compares, and would surface as a mysterious
    ingest failure instead of a timezone bug.
    """
    moment = raw if isinstance(raw, datetime) else datetime.fromisoformat(raw)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _is_transient(exc: Exception) -> bool:
    """Best-effort transient-failure classification, deliberately pessimistic.

    Without importing SDK exception types (which would break the
    lazy-import rule) the only signals are the exception's stdlib base
    classes, its class name, and its message. Anything unrecognised is
    fatal: the cost of under-retrying is one failed call the operator
    re-runs, and the cost of over-retrying a non-idempotent call is a
    duplicate post.
    """
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if any(marker in name for marker in ("transient", "ratelimit", "timeout")):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


@dataclass(slots=True)
class ComposioConfig:
    """Connection settings for the Composio adapter.

    ``slug_overrides`` exists because the registry below is UNVERIFIED
    against a live account: when a real slug turns out to differ, the fix is
    a config entry (``{"publish:x": "X_CREATE_POST"}``), not a code change
    and a release. Keys are ``"<kind>:<platform value>"`` with kind one of
    ``publish`` / ``analytics``.
    """

    api_key: str | None = None
    """None reads ``COMPOSIO_API_KEY`` from the environment at call time, so
    a key never has to live in a config file that gets committed."""

    user_id: str = ""
    """Composio's connected-account identifier (``user_id`` on execute)."""

    timeout_seconds: float = 30.0
    max_retries: int = 3
    dry_run: bool = False
    slug_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolkitEntry:
    """One platform's toolkit name and its publish/analytics action slugs."""

    toolkit: str
    publish: str
    analytics: str


class ToolkitRegistry:
    """Platform -> Composio toolkit and action slugs.

    UNVERIFIED AGAINST A LIVE ACCOUNT. The names below are the author's best
    reading of Composio's naming conventions (``TWITTER_CREATION_OF_A_POST``
    is the convention every slug follows), written offline with no network
    and no connected account to check against. A confidently wrong slug is
    worse than a flagged one -- it fails at publish time with a provider
    error an operator will misread as a permissions problem -- so treat this
    table as a hypothesis to confirm on first live use, and correct via
    ``ComposioConfig.slug_overrides`` rather than by editing this class.
    Every slug is overridable; nothing here is load-bearing once a config
    supplies the real names.
    """

    ENTRIES: dict[Platform, ToolkitEntry] = {
        Platform.X: ToolkitEntry(
            toolkit="TWITTER",
            publish="TWITTER_CREATION_OF_A_POST",
            analytics="TWITTER_GET_TWEET_BY_ID",
        ),
        Platform.LINKEDIN: ToolkitEntry(
            toolkit="LINKEDIN",
            publish="LINKEDIN_CREATE_LINKED_IN_POST",
            analytics="LINKEDIN_GET_POST_ANALYTICS",
        ),
        Platform.REDDIT: ToolkitEntry(
            toolkit="REDDIT",
            publish="REDDIT_CREATE_A_POST",
            analytics="REDDIT_GET_POST_DETAILS",
        ),
        Platform.TIKTOK: ToolkitEntry(
            toolkit="TIKTOK",
            publish="TIKTOK_CREATE_POST",
            analytics="TIKTOK_GET_POST_METRICS",
        ),
        Platform.INSTAGRAM: ToolkitEntry(
            toolkit="INSTAGRAM",
            publish="INSTAGRAM_CREATE_MEDIA_ITEM",
            analytics="INSTAGRAM_GET_MEDIA_INSIGHTS",
        ),
        Platform.THREADS: ToolkitEntry(
            toolkit="THREADS",
            publish="THREADS_CREATE_POST",
            analytics="THREADS_GET_POST",
        ),
        Platform.YOUTUBE_SHORTS: ToolkitEntry(
            toolkit="YOUTUBE",
            publish="YOUTUBE_UPLOAD_VIDEO",
            analytics="YOUTUBE_GET_VIDEO_METRICS",
        ),
    }

    def entry(
        self,
        platform: Platform,
        overrides: Mapping[str, str] | None = None,
    ) -> ToolkitEntry:
        """Resolve one platform's slugs, applying ``overrides`` last.

        Override keys are ``publish:<platform.value>`` and
        ``analytics:<platform.value>`` -- the toolkit name is Composio's and
        slugs already encode it, so overriding the toolkit separately would
        be a third way to say the same thing.
        """
        base = self.ENTRIES[platform]
        if not overrides:
            return base
        publish = overrides.get(f"publish:{platform.value}", base.publish)
        analytics = overrides.get(f"analytics:{platform.value}", base.analytics)
        return ToolkitEntry(toolkit=base.toolkit, publish=publish, analytics=analytics)


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    """What came back from one publish attempt.

    ``url`` is the destination AFTER utm stamping, so the receipt is itself
    the record of which join key the post carries: if the observation never
    arrives, the receipt is where you check whether the stamp was there.
    """

    platform: Platform
    utm_content: str
    url: str
    action: str
    dry_run: bool
    response_id: str = ""


def _as_dict(result: Any) -> dict[str, Any]:
    """Normalise an SDK ExecuteResponse into a plain dict.

    The verified surface says only that ``tools.execute`` returns a result;
    Composio's client has historically returned response objects exposing
    ``.data`` or ``.to_dict()``. Anything dict-shaped or convertible is
    accepted, anything else fails loudly rather than being stringified --
    a metrics payload parsed out of a repr is worse than no payload.
    """
    if isinstance(result, dict):
        return result
    for attribute in ("data", "response"):
        value = getattr(result, attribute, None)
        if isinstance(value, dict):
            return value
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError(
        f"unrecognised Composio result shape {type(result).__name__}; "
        "expected a mapping or a response exposing .data/.to_dict()"
    )


def observation_from_metrics(
    utm_content: str,
    metrics: Mapping[str, Any],
    attribution_coverage: float | None,
    incrementality: float | None,
    now: datetime,
) -> Observation:
    """Map raw platform metrics into an ``Observation``.

    Metric keys are the ``Observation`` field names; missing keys read as
    zero rather than raising, because platform analytics endpoints return
    sparse objects (a text post has no hold rate) and a missing metric is
    genuinely absent data, not a malformed response. The two fields that are
    NOT metrics -- coverage and incrementality -- come only from the
    operator's measurements and refuse to be guessed
    (``MissingMeasurementError``).

    ``is_partial`` is set from the elapsed time between ``posted_at`` and
    ``now`` against the measured reporting lag's high end; see
    ``_REPORTING_LAG``.
    """
    _require_measurements(attribution_coverage, incrementality)
    posted = _parse_timestamp(str(metrics.get("posted_at", now.isoformat())))
    observed = _parse_timestamp(str(metrics.get("observed_at", now.isoformat())))
    return Observation(
        utm_content=utm_content,
        posted_at=posted.date().isoformat(),
        observed_at=observed.date().isoformat(),
        impressions=int(metrics.get("impressions", 0)),
        reach=int(metrics.get("reach", 0)),
        hook_rate=float(metrics.get("hook_rate", 0.0)),
        hold_rate=float(metrics.get("hold_rate", 0.0)),
        saves=int(metrics.get("saves", 0)),
        shares=int(metrics.get("shares", 0)),
        comments=int(metrics.get("comments", 0)),
        profile_visits=int(metrics.get("profile_visits", 0)),
        link_clicks=int(metrics.get("link_clicks", 0)),
        attributed_conversions=int(metrics.get("attributed_conversions", 0)),
        attribution_coverage=attribution_coverage,
        incrementality=incrementality,
        is_partial=(now - posted) < _REPORTING_LAG,
        dark_social_estimate=float(metrics.get("dark_social_estimate", 0.0)),
    )


class ComposioAdapter:
    """Live platform I/O. Every SDK call funnels through ``_execute``.

    The two operator measurements (``attribution_coverage``,
    ``incrementality``) are constructor arguments rather than derived from
    anything, because they cannot be derived from anything -- see
    ``MissingMeasurementError``. ``None`` (the default) keeps the adapter
    honest: it will publish, but ``fetch_analytics`` will refuse rather
    than emit observations denominated in guessed coverage.
    """

    def __init__(
        self,
        config: ComposioConfig | None = None,
        attribution_coverage: float | None = None,
        incrementality: float | None = None,
        registry: ToolkitRegistry | None = None,
    ) -> None:
        self.config = config if config is not None else ComposioConfig()
        self.attribution_coverage = attribution_coverage
        self.incrementality = incrementality
        self.registry = registry if registry is not None else ToolkitRegistry()
        self._client: Any | None = None

    # -- the chokepoint -----------------------------------------------------

    def _execute(
        self,
        slug: str,
        arguments: dict[str, Any],
        *,
        retryable: bool = True,
    ) -> dict[str, Any]:
        """Execute one Composio tool call. The ONLY SDK entry point.

        Every platform interaction this package performs -- publish,
        analytics, anything added later -- passes through here, so SDK
        version drift, credential resolution, and retry policy each have
        exactly one implementation and one place to fix.

        Retries: exponential backoff on transient failure, but ONLY when
        ``retryable``. Reads pass ``retryable=True`` (a duplicate read is
        free); ``publish`` passes False, because once a request has left the
        process an exception tells us nothing about whether the platform
        acted on it, and a duplicate post is irreversible. This is the
        module's second refusal-to-guess: better an operator re-runs a
        failed publish by hand than the adapter double-posts on their behalf.
        """
        if self._client is None:
            try:
                from composio import Composio
            except ImportError as exc:
                raise ImportError(
                    'the composio SDK is not installed; run '
                    'pip install "blotto[platforms]"'
                ) from exc
            api_key = self.config.api_key or os.environ.get("COMPOSIO_API_KEY")
            self._client = Composio(api_key=api_key)

        attempts = max(0, self.config.max_retries) + 1
        for attempt in range(attempts):
            try:
                result = self._client.tools.execute(
                    slug,
                    arguments=arguments,
                    user_id=self.config.user_id,
                )
            except Exception as exc:
                more = attempt < attempts - 1
                if retryable and more and _is_transient(exc):
                    delay = _BACKOFF_BASE_SECONDS * (2**attempt)
                    logger.warning(
                        "transient Composio failure on %s (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        slug,
                        attempt + 1,
                        attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                raise
            return _as_dict(result)

        # Unreachable: the loop returns or raises on its final attempt.
        raise RuntimeError(f"retry loop exited without a result on {slug!r}")

    # -- public interface ----------------------------------------------------

    def publish(self, move: Publish, content: Mapping[str, Any]) -> PublishReceipt:
        """Post one ``Publish`` move, its utm id stamped into the link.

        ``content`` is a brief as emitted by ``blotto brief``: free-form
        fields plus ``"url"``, the destination the utm is stamped into. Only
        ``Publish`` is postable -- a Scale commits budget, it does not ship
        an asset -- so anything else is a TypeError rather than a
        best-effort post of the wrong thing.

        NOT RETRIED, at any failure, ever (see ``_execute``): a timeout may
        mean the post landed and the response was lost, and two posts under
        one utm id would collapse into one row of analytics -- silently
        merging the learning of two distinct creatives.
        """
        if not isinstance(move, Publish):
            raise TypeError(f"publish posts Publish moves, not {type(move).__name__}")
        if "url" not in content:
            raise ValueError("content must carry a 'url' for utm stamping")
        url = stamp_utm(str(content["url"]), move.utm_content)
        entry = self.registry.entry(move.platform, self.config.slug_overrides)
        arguments: dict[str, Any] = {
            "text": str(content.get("text", "")),
            "link": url,
            "utm_content": move.utm_content,
        }
        result = self._execute(entry.publish, arguments, retryable=False)
        response_id = str(
            result.get("id", result.get("post_id", "")),
        )
        return PublishReceipt(
            platform=move.platform,
            utm_content=move.utm_content,
            url=url,
            action=entry.publish,
            dry_run=False,
            response_id=response_id,
        )

    def fetch_analytics(
        self,
        utm_contents: Sequence[str],
        since: datetime,
        platforms: Mapping[str, Platform] | None = None,
    ) -> list[Observation]:
        """Pull settled-and-partial metrics for each utm id, as Observations.

        ``platforms`` maps utm id -> the platform it was posted to. The live
        API is organised by toolkit, so without it the adapter cannot know
        which analytics slug owns a given id -- and guessing a toolkit is
        the same class of error as guessing a slug. The caller who owns the
        trajectory store owns this mapping; a missing map (or a missing
        entry) is a ValueError rather than a query against the wrong
        platform. ``RecordedAdapter`` ignores the mapping entirely.

        Fresh observations -- inside the measured reporting lag -- are
        returned with ``is_partial=True`` so the evidence gates can refuse
        them; the reference model and the legality engine agree that a
        partial observation may be read but must not open a gate.
        """
        _require_measurements(self.attribution_coverage, self.incrementality)
        since = _parse_timestamp(since)
        if platforms is None:
            raise ValueError(
                "fetch_analytics requires platforms= (utm -> Platform); the "
                "analytics toolkit cannot be guessed from the utm alone"
            )
        now = datetime.now(UTC)
        observations: list[Observation] = []
        for utm in utm_contents:
            if utm not in platforms:
                raise ValueError(
                    f"no platform known for utm {utm!r}; pass platforms= from "
                    "the trajectory store -- the toolkit cannot be guessed"
                )
            platform = platforms[utm]
            entry = self.registry.entry(platform, self.config.slug_overrides)
            result = self._execute(
                entry.analytics,
                {"utm_content": utm, "since": since.date().isoformat()},
            )
            metrics = result.get("metrics", result)
            if not metrics:
                continue
            observation = observation_from_metrics(
                utm,
                metrics,
                self.attribution_coverage,
                self.incrementality,
                now,
            )
            if _parse_timestamp(observation.posted_at) >= since:
                observations.append(observation)
        return observations


class DryRunAdapter:
    """Same interface as ``ComposioAdapter``, zero I/O.

    The CLI's default: an operator should be able to see exactly what WOULD
    be posted, under which slugs, with which utm stamps, before any
    credential exists. Intended calls are recorded in ``calls`` (machine)
    and logged (human); ``fetch_analytics`` returns an empty list because a
    dry run has no data and must not invent any -- fabricating plausible
    metrics here would be the single easiest way to train a model on
    fiction.
    """

    def __init__(
        self,
        attribution_coverage: float | None = None,
        incrementality: float | None = None,
    ) -> None:
        self.attribution_coverage = attribution_coverage
        self.incrementality = incrementality
        self.calls: list[dict[str, Any]] = []

    def publish(self, move: Publish, content: Mapping[str, Any]) -> PublishReceipt:
        """Record the intended call and return a dry-run receipt."""
        if not isinstance(move, Publish):
            raise TypeError(f"publish posts Publish moves, not {type(move).__name__}")
        if "url" not in content:
            raise ValueError("content must carry a 'url' for utm stamping")
        url = stamp_utm(str(content["url"]), move.utm_content)
        self.calls.append(
            {
                "action": "publish",
                "platform": move.platform.value,
                "utm_content": move.utm_content,
                "url": url,
                "text": str(content.get("text", "")),
            }
        )
        logger.info("[dry-run] publish %s on %s -> %s", move.utm_content, move.platform.value, url)
        return PublishReceipt(
            platform=move.platform,
            utm_content=move.utm_content,
            url=url,
            action="(dry-run)",
            dry_run=True,
        )

    def fetch_analytics(
        self,
        utm_contents: Sequence[str],
        since: datetime,
        platforms: Mapping[str, Platform] | None = None,
    ) -> list[Observation]:
        """Record the intended reads; observe nothing and invent nothing."""
        self.calls.append(
            {
                "action": "fetch_analytics",
                "utm_contents": list(utm_contents),
                "since": since.date().isoformat(),
            }
        )
        logger.info(
            "[dry-run] fetch_analytics for %d utm ids since %s",
            len(utm_contents),
            since.date().isoformat(),
        )
        return []


class RecordedAdapter:
    """Replay recorded Composio responses from a JSONL fixture.

    For offline tests, in the same spirit as ``RecordedClient`` in
    ``blotto.cwm.llm``: a recorded response carries the shape a live one
    actually had, which a mock asserts only what its author believed. Each
    fixture line is ``{"utm_content": ..., "response": {...}}``; a utm with
    no recorded entry is SKIPPED rather than synthesised, so a fixture that
    stops mid-history reads as "no data yet" -- the same reading the live
    adapter gives a post the platform has not indexed.
    """

    def __init__(
        self,
        path: Path,
        attribution_coverage: float | None = None,
        incrementality: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.attribution_coverage = attribution_coverage
        self.incrementality = incrementality
        self._responses: dict[str, dict[str, Any]] = {}
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._responses[entry["utm_content"]] = entry["response"]

    def publish(self, move: Publish, content: Mapping[str, Any]) -> PublishReceipt:
        """Return the recorded receipt for this utm, utm still stamped."""
        if not isinstance(move, Publish):
            raise TypeError(f"publish posts Publish moves, not {type(move).__name__}")
        if "url" not in content:
            raise ValueError("content must carry a 'url' for utm stamping")
        url = stamp_utm(str(content["url"]), move.utm_content)
        recorded = self._responses.get(move.utm_content, {})
        return PublishReceipt(
            platform=move.platform,
            utm_content=move.utm_content,
            url=url,
            action="(recorded)",
            dry_run=False,
            response_id=str(recorded.get("id", "")),
        )

    def fetch_analytics(
        self,
        utm_contents: Sequence[str],
        since: datetime,
        platforms: Mapping[str, Platform] | None = None,
    ) -> list[Observation]:
        """Replay recorded metrics through the same mapping the live adapter
        uses -- including the measurement refusal, so a test fixture cannot
        quietly bypass the rule the live path enforces."""
        _require_measurements(self.attribution_coverage, self.incrementality)
        since = _parse_timestamp(since)
        now = datetime.now(UTC)
        observations: list[Observation] = []
        for utm in utm_contents:
            response = self._responses.get(utm)
            if not response:
                continue
            # Unwrap exactly as the live adapter does, so a fixture line's
            # "response" is byte-compatible with what _execute would return
            # for the same call -- one response shape, two consumers.
            metrics = response.get("metrics", response)
            observation = observation_from_metrics(
                utm,
                metrics,
                self.attribution_coverage,
                self.incrementality,
                now,
            )
            if _parse_timestamp(observation.posted_at) >= since:
                observations.append(observation)
        return observations
