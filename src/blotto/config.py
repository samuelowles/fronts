"""Operator configuration: TOML in, one validated object out.

Everything the game and solver layers need that is not a corpus constant is
an operator input, and every operator input lives here -- unit economics,
legality thresholds, platform credentials, file locations, and the two
MEASURED attribution numbers nothing else in the system is permitted to
guess. A config file is the right home for all of it because the honest
answer to "why is your creative-judgement gate at 50?" is a citation, and
the honest answer to "why is your attribution coverage 0.72?" is "I
measured it, on this date, with this method" -- both belong in a file the
operator owns, not in code.

The parser is ``tomllib``, standard since Python 3.11. On an older
interpreter ``load`` raises an error naming the required version rather
than falling back to a hand-rolled parser: a bespoke TOML subset is a
second parser to test, and a config silently mis-read (inline tables,
dates, dotted keys) is far worse than a clear "upgrade Python".

Note on the ``tomli`` backport: it is accepted as a drop-in when already
present (any environment running pytest on Python 3.10 has it, since pytest
itself depends on it there), which keeps the CLI usable for 3.10
contributors without adding a dependency -- the package still imports, and
still requires nothing installed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from blotto.adapters.composio_io import ComposioConfig
from blotto.game.legality import OperatorPolicy
from blotto.game.payoff import Economics
from blotto.game.types import Format

__all__ = ["Paths", "BlottoConfig", "load", "example", "default_config"]


@dataclass(frozen=True, slots=True)
class Paths:
    """Where the loop's artefacts live. Relative paths resolve against the
    working directory the command runs from -- deliberately simple, because
    a config that silently rebased paths against the config file's own
    location would work until the first cron job ran from elsewhere."""

    history: Path = Path("data/trajectories.jsonl")
    """Recorded moves and observations; what ``synth`` learns from."""

    rules: Path = Path("rules.md")
    """Platform policy in prose; the synthesis prompt's rules section."""

    model: Path = Path("data/model.py")
    """The synthesised world model source; ``plan`` loads and runs it."""

    plan: Path = Path("data/plan.json")
    """``plan`` writes here; ``brief`` and ``publish`` read here."""

    briefs: Path = Path("data/briefs.json")
    """``brief`` writes here: the plan as creative briefs."""


@dataclass(slots=True)
class BlottoConfig:
    """The whole operator configuration in one object.

    ``attribution_coverage`` and ``incrementality`` are MEASUREMENTS, not
    settings: the fraction of true conversions your tracking can see, and
    the fraction of attributed conversions you actually caused. Neither is
    derivable from any API, both default to ``None``, and the adapter
    refuses to build an ``Observation`` while either is unset -- see
    ``blotto.adapters.composio_io.MissingMeasurementError`` and
    docs/OPERATING.md before setting them by anything other than a holdout
    or a survey reconciliation.
    """

    economics: Economics
    policy: OperatorPolicy = field(default_factory=OperatorPolicy)
    composio: ComposioConfig = field(default_factory=ComposioConfig)
    paths: Paths = field(default_factory=Paths)
    attribution_coverage: float | None = None
    incrementality: float | None = None
    destination_url: str | None = None
    """Landing page publishes stamp their utm into; what ``publish`` links to.
    None means the operator has not wired a destination yet, and publishing
    fails with a clear error rather than posting a dead link."""


def default_config() -> BlottoConfig:
    """A config with no file: the repo's worked-example economics and every
    corpus-default policy knob.

    The economics are the worked example from ``blotto.game.payoff``'s tests
    (ARPU 100, margin 0.8, churn 0.08 -- LTV 1000), carried in
    ``ReferenceConfig`` for the same reason: they reproduce a documented
    example, they are not a claim about any real business. The two
    measurements stay ``None``; there is no worked example for a number you
    have to have measured yourself.
    """
    return BlottoConfig(
        economics=Economics(
            arpu_monthly=100.0,
            gross_margin=0.8,
            monthly_churn=0.08,
            cogs_share=0.2,
            fixed_cost_per_post=50.0,
        )
    )


def _toml_parser() -> Any:
    """Resolve tomllib (3.11+) or the tomli backport, lazily.

    Imported inside this function, not at module level, so importing
    ``blotto.config`` -- and everything downstream, including the CLI --
    succeeds on a machine with no TOML library installed at all; only
    actually loading a file requires one.
    """
    try:
        import tomllib

        return tomllib
    except ImportError:
        pass
    try:
        import tomli

        return tomli
    except ImportError as exc:
        raise RuntimeError(
            "loading a blotto config requires the standard library's TOML "
            "parser, which arrived in Python 3.11; this interpreter is "
            f"Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Install Python 3.11 or newer (or the 'tomli' backport) and try "
            "again."
        ) from exc


def _economics(table: dict[str, Any]) -> Economics:
    known = {f.name for f in fields(Economics)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"unknown [economics] keys: {unknown}")
    return Economics(**table)


def _policy(table: dict[str, Any]) -> OperatorPolicy:
    """Map the ``[policy]`` table onto ``OperatorPolicy``.

    Format keys arrive as TOML strings and become ``Format`` members; an
    unknown format is an error rather than a silently dropped cap, because
    a cap that failed to load is a cap that is not enforced. A PARTIAL
    ``[policy.max_per_day_by_format]`` table merges over the defaults
    rather than replacing them: setting one cap must not quietly un-set
    the others, or the day's cadence constraint dies by config typo.
    """
    policy = OperatorPolicy()
    known = {f.name for f in fields(OperatorPolicy)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"unknown [policy] keys: {unknown}")
    for key, value in table.items():
        if key == "max_per_day_by_format":
            caps = dict(policy.max_per_day_by_format)
            for name, cap in value.items():
                caps[Format(name)] = int(cap)
            policy.max_per_day_by_format = caps
        else:
            setattr(policy, key, value)
    return policy


def _composio(table: dict[str, Any]) -> ComposioConfig:
    known = {f.name for f in fields(ComposioConfig)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"unknown [composio] keys: {unknown}")
    return ComposioConfig(**table)


def _paths(table: dict[str, Any]) -> Paths:
    known = {f.name for f in fields(Paths)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"unknown [paths] keys: {unknown}")
    return Paths(**{key: Path(value) for key, value in table.items()})


def _measurement(name: str, value: Any) -> float | None:
    """Validate one operator measurement: absent, or a fraction in [0, 1].

    Out-of-range values raise rather than clamp: a coverage of 1.4 is not
    "close enough to 1.0", it is evidence the operator entered a percentage
    where a fraction belongs, and clamping it would silently re-base every
    reward on a number nobody intended.
    """
    if value is None:
        return None
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{name} must be a fraction in [0, 1] (got {fraction}); if you "
            "measured 72%, that is 0.72"
        )
    return fraction


def load(path: str | Path) -> BlottoConfig:
    """Load and validate a config file.

    Every section is optional; anything absent keeps its default, and the
    two measurements default to unset -- a config that loads cleanly with
    no ``[measurement]`` section is a config whose ingest will refuse to
    guess, which is the designed behaviour and not a misconfiguration.
    """
    parser = _toml_parser()
    with open(path, "rb") as handle:
        data = parser.load(handle)

    config = default_config()
    if "economics" in data:
        config.economics = _economics(data["economics"])
    if "policy" in data:
        config.policy = _policy(data["policy"])
    if "composio" in data:
        config.composio = _composio(data["composio"])
    if "paths" in data:
        config.paths = _paths(data["paths"])
    measurement = data.get("measurement", {})
    unknown = sorted(set(measurement) - {"attribution_coverage", "incrementality"})
    if unknown:
        raise ValueError(f"unknown [measurement] keys: {unknown}")
    config.attribution_coverage = _measurement(
        "attribution_coverage", measurement.get("attribution_coverage")
    )
    config.incrementality = _measurement(
        "incrementality", measurement.get("incrementality")
    )
    publish = data.get("publish", {})
    unknown = sorted(set(publish) - {"destination_url"})
    if unknown:
        raise ValueError(f"unknown [publish] keys: {unknown}")
    config.destination_url = (
        str(publish["destination_url"]) if "destination_url" in publish else None
    )
    return config


def example() -> str:
    """A fully commented example config, byte-identical to the repo's
    ``blotto.example.toml`` (a test asserts the identity, so the shipped
    file can never drift from what ``load`` accepts).

    The measurement section is commented OUT, on purpose: shipping active
    plausible values would be inventing measurements, and the single most
    damaging thing this file could do is make an unmeasured operator look
    configured.
    """
    return """\
# blotto configuration. Copy to blotto.toml and edit.
# Every section is optional; anything absent keeps its default.
# `blotto --config PATH ...` selects a non-default file.

[paths]
# Where the loop's artefacts live. Relative paths resolve against the
# directory you run `blotto` from.
history  = "data/trajectories.jsonl"   # recorded moves + observations
rules    = "rules.md"                  # platform policy, in prose
model    = "data/model.py"             # the synthesised world model
plan     = "data/plan.json"            # written by `plan`, read by `publish`
briefs   = "data/briefs.json"          # written by `brief`

[economics]
# YOUR unit economics, per paying user per month. The reward the planner
# maximises is contribution margin computed from these -- see
# src/blotto/game/payoff.py for what each field does to it.
arpu_monthly      = 100.0    # revenue per paying user per month
gross_margin      = 0.8      # after cost of service
monthly_churn     = 0.08     # fraction of paying users lost per month
cogs_share        = 0.2      # cost of goods as a share of revenue (margin + this <= 1)
fixed_cost_per_post = 50.0   # what one published item costs you to produce
# allowable_cac_share = 0.30 # ceiling on CAC as a share of LTV (corpus default)

[policy]
# Legality thresholds. The defaults are corpus figures with citations in
# src/blotto/game/legality.py; change them when your risk tolerance differs,
# not to make a refused move pass.
creative_judgement_min_conversions = 50   # settled conversions / 7d to call a winner
creative_judgement_window_days     = 7    # the trailing window those settle in
spend_commitment_min_conversions   = 300  # settled conversions / variant over 14d
spend_commitment_min_days          = 14
spend_commitment_factor            = 1.20 # Scale factor that counts as budget
attribution_coverage_floor         = 0.5  # below this an angle is unjudgeable
max_allocation_increase_48h        = 0.20 # velocity cap per rolling 48h
acquisition_ratio_target           = 0.70 # day's acquisition/service split
acquisition_ratio_tolerance        = 0.10
comparative_allowed                = false
drawdown_spend_limit               = 1000.0
standing_floor                     = 0.5

[policy.max_per_day_by_format]
# Per-format daily caps. Production capacity, not aspiration.
carousel    = 10
text_thread = 5

[composio]
# Platform I/O via Composio (https://composio.dev). The api key is read
# from COMPOSIO_API_KEY when api_key is unset -- prefer the environment;
# keys in files get committed.
# api_key         = "cs_..."
user_id          = "user-123"
max_retries      = 3
# Action slugs are UNVERIFIED against a live account (see
# ToolkitRegistry). When a real slug differs, override it here -- no code
# change needed. Keys are "publish:<platform>" / "analytics:<platform>".
# slug_overrides = { "publish:x" = "X_CREATE_POST" }

[publish]
# Landing page publishes stamp their utm_content into. Must exist before
# `blotto publish --live` will ship anything.
destination_url = "https://example.com/pricing"

[measurement]
# THE TWO NUMBERS NOTHING ELSE CAN SUPPLY. Leave commented out until you
# have measured them -- docs/OPERATING.md, "The thing that will bite you".
# Until they are set, ingest refuses to build observations rather than
# guessing, and the legality engine keeps every angle unjudgeable.
#
# attribution_coverage = 0.72  # fraction of true conversions your tracking sees
# incrementality       = 0.70  # fraction of attributed conversions YOU caused
#
# How to measure: attribution_coverage from a post-purchase "how did you
# hear about us" survey reconciled against analytics for a month;
# incrementality from a holdout (pause a channel, watch what does not drop).
# The GTM corpus records the usual finding: the dashboard says direct, and
# 40% of respondents name a channel it cannot see.
"""
