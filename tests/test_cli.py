"""Unit tests for the CLI and the config loader behind it.

The CLI's contract with an operator's shell script is threefold, and each
part is asserted here: the commands parse exactly as docs/OPERATING.md and
README.md name them; ``publish`` touches nothing live without ``--live``; a
legality refusal is exit code 2 and prints the rule, the reason AND the
source, so the operator can see which rule stopped them and where it came
from.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from blotto.adapters.trajectory import TrajectoryStore
from blotto.cli import _build_parser, app
from blotto.config import example, load
from blotto.game.action_space import ActionCodec
from blotto.game.types import (
    Archetype,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    HookFamily,
    Observation,
    Platform,
    Publish,
    Scale,
    SemanticTier,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CODEC = ActionCodec()
_TODAY = date.today()


def _publish(**overrides: object) -> Publish:
    fields: dict[str, object] = {
        "platform": Platform.X,
        "format": Format.TEXT_THREAD,
        "archetype": Archetype.FOUNDER_TRAUMA,
        "vector": EmotionalVector.EXHAUSTION_RELIEF,
        "semantic_tier": SemanticTier.TRIBAL_IDENTITY,
        "hook": HookFamily.CURIOSITY_GAP,
        "avatar": "solo_founder",
        "cta_mode": CtaMode.START_TRIAL,
        "claim_class": ClaimClass.FIRST_PARTY_PROOF,
        "angle": "launch_theater",
        "utm_content": "c0412ab9",
    }
    fields.update(overrides)
    return Publish(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """\
[paths]
history = "{history}"
rules    = "{rules}"
model    = "{model}"
plan     = "{plan}"
briefs   = "{briefs}"

[economics]
arpu_monthly       = 100.0
gross_margin       = 0.8
monthly_churn      = 0.08
cogs_share         = 0.2
fixed_cost_per_post = 50.0

[composio]
user_id = "user-123"

[publish]
destination_url = "https://owles.works/pricing"

[measurement]
attribution_coverage = 0.72
incrementality = 0.70
"""


def _write_config(tmp_path: Path, *, measurements: bool = True) -> Path:
    """A config pointing every path at tmp_path. Measurements included by
    default because a publish-refusal fixture needs an above-floor coverage
    for the evidence gate -- not the coverage gate -- to fire."""
    text = _CONFIG_TEMPLATE.format(
        history=(tmp_path / "history.jsonl").as_posix(),
        rules=(tmp_path / "rules.md").as_posix(),
        model=(tmp_path / "model.py").as_posix(),
        plan=(tmp_path / "plan.json").as_posix(),
        briefs=(tmp_path / "briefs.json").as_posix(),
    )
    if not measurements:
        marker = text.index("[measurement]")
        text = text[:marker]
    path = tmp_path / "blotto.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _write_plan(tmp_path: Path, moves: list[str]) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"moves": moves}), encoding="utf-8")
    return plan


def _write_history_with_conversions(
    tmp_path: Path, conversions: int, *, utm: str = "c0412ab9"
) -> Path:
    """A one-post history whose settled observation carries ``conversions``
    conversions posted five days ago -- inside the 7d window, outside the
    reporting lag."""
    store = TrajectoryStore(tmp_path / "history.jsonl")
    store.append_move(
        _publish(utm_content=utm), posted_at=(_TODAY - timedelta(days=5)).isoformat()
    )
    store.attach_observation(
        Observation(
            utm_content=utm,
            posted_at=(_TODAY - timedelta(days=5)).isoformat(),
            observed_at=_TODAY.isoformat(),
            reach=2000,
            attributed_conversions=conversions,
            attribution_coverage=0.72,
            incrementality=0.70,
        )
    )
    return store.path


# ---------------------------------------------------------------------------
# Parsing: the command surface is the documented surface.
# ---------------------------------------------------------------------------

_Documented_COMMANDS: list[list[str]] = [
    ["synth", "--history", "data/trajectories.jsonl", "--rules", "rules.md"],
    ["synth", "--history", "h.jsonl", "--rules", "r.md", "--out", "m.py",
     "--max-calls", "5"],
    ["accuracy"],
    ["accuracy", "--model", "data/model.py"],
    ["plan", "--days", "7", "--sims", "1000"],
    ["plan"],
    ["arena"],
    ["ingest"],
    ["ingest", "--since", "2026-08-01"],
    ["brief"],
    ["publish"],
    ["publish", "--live"],
    ["stats"],
]


@pytest.mark.parametrize("argv", _Documented_COMMANDS, ids=lambda argv: " ".join(argv))
def test_every_documented_subcommand_parses(argv: list[str]) -> None:
    args = _build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_global_flags_parse_before_and_after_the_subcommand() -> None:
    before = _build_parser().parse_args(["--config", "c.toml", "--seed", "5", "--json",
                                         "stats"])
    after = _build_parser().parse_args(["stats", "--config", "c.toml", "--seed", "5",
                                        "--json"])
    for args in (before, after):
        assert args.command == "stats"
        assert args.config == "c.toml"
        assert args.seed == 5
        assert args.json is True
    bare = _build_parser().parse_args(["stats"])
    assert getattr(bare, "seed", None) is None, "absent flags leave defaults unset"
    assert not getattr(bare, "json", False)


def test_no_subcommand_prints_help_and_exits_nonzero(capsys: pytest.CaptureFixture) -> None:
    assert app([]) == 1
    assert "plan" in capsys.readouterr().out


def test_cli_uses_argparse_only() -> None:
    """typer and rich appear nowhere, by design: a CLI that needs a wheel
    before it will talk is a CLI nobody runs on the box that matters."""
    source = Path(sys.modules["blotto.cli"].__file__).read_text(encoding="utf-8")
    assert "typer" not in source.lower()
    assert "rich" not in source.lower()
    assert "argparse" in source
    offenders = [name for name in sys.modules if name.split(".")[0] in ("typer", "rich")]
    assert not offenders


# ---------------------------------------------------------------------------
# publish: dry by default; refusal is exit 2 with rule, reason and source.
# ---------------------------------------------------------------------------


def test_publish_without_live_performs_no_live_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    config = _write_config(tmp_path)
    _write_plan(tmp_path, [_CODEC.encode(_publish())])
    _write_history_with_conversions(tmp_path, conversions=60)

    def no_live_calls(self: object, slug: str, arguments: dict) -> dict:
        raise AssertionError("a dry-run publish reached the Composio SDK")

    monkeypatch.setattr(
        "blotto.adapters.composio_io.ComposioAdapter._execute", no_live_calls
    )
    code = app(["publish", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "DRY RUN" in out
    assert "utm_content=c0412ab9" in out, "the receipt shows the stamped join key"
    assert not [name for name in sys.modules if name.split(".")[0] == "composio"]
    recorded = TrajectoryStore(tmp_path / "history.jsonl").load()
    assert len(recorded.steps) == 2, "the published move joins the history"


def test_publish_live_uses_the_composio_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """--live is the only path that constructs the live adapter; the
    chokepoint is stubbed so even a constructed one never reaches the
    (uninstalled) SDK."""
    config = _write_config(tmp_path)
    _write_plan(tmp_path, [_CODEC.encode(_publish())])
    _write_history_with_conversions(tmp_path, conversions=60)
    monkeypatch.setattr(
        "blotto.adapters.composio_io.ComposioAdapter._execute",
        lambda self, slug, arguments, **_: {"id": "tw-1"},
    )
    code = app(["publish", "--config", str(config), "--live"])
    out = capsys.readouterr().out
    assert code == 0
    assert "LIVE" in out
    assert "utm_content=c0412ab9" in out


def test_publish_refusal_exits_two_and_prints_rule_reason_source(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """12 settled conversions against a 50-conversion gate: the exact move
    the corpora say kills accounts, refused with its citation."""
    config = _write_config(tmp_path)
    _write_plan(tmp_path, [_CODEC.encode(Scale("launch_theater", 1.10))])
    history = _write_history_with_conversions(tmp_path, conversions=12)
    history_before = history.read_bytes()

    code = app(["publish", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 2, "refused-by-legality is its own exit code, not an error"
    assert "REFUSED" in out
    assert "CREATIVE_JUDGEMENT" in out
    assert "12 settled conversions" in out
    assert "17_A_B_Testing_Story_Arcs" in out, "the source is part of the refusal"
    assert history.read_bytes() == history_before, (
        "a refused plan ships NOTHING -- not even the legal-looking parts"
    )


def test_publish_refuses_when_unmeasured_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """With coverage unmeasured the angle is unjudgeable: the refusal names
    the coverage rule, not the evidence gate."""
    config = _write_config(tmp_path, measurements=False)
    _write_plan(tmp_path, [_CODEC.encode(Scale("launch_theater", 1.10))])
    _write_history_with_conversions(tmp_path, conversions=400)
    code = app(["publish", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 2
    assert "ATTRIBUTION_COVERAGE" in out


def test_publish_without_a_plan_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path)
    assert app(["publish", "--config", str(config)]) == 1
    assert "no plan" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# brief / stats / ingest
# ---------------------------------------------------------------------------


def test_brief_emits_the_plan_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path)
    _write_plan(
        tmp_path,
        [
            _CODEC.encode(_publish()),
            _CODEC.encode(Scale("launch_theater", 1.10)),
        ],
    )
    assert app(["brief", "--config", str(config)]) == 0
    briefs = json.loads(capsys.readouterr().out)
    assert len(briefs) == 1, "scales are decisions, not creative briefs"
    assert briefs[0]["utm_content"] == "c0412ab9"
    assert briefs[0]["angle"] == "launch_theater"
    assert (tmp_path / "briefs.json").exists()


def test_stats_reports_cold_start_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path)
    _write_history_with_conversions(tmp_path, conversions=12)
    assert app(["stats", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "NOT cleared" in out


def test_stats_json_flag_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path)
    _write_history_with_conversions(tmp_path, conversions=12)
    assert app(["stats", "--config", str(config), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settled_observations"] == 1
    assert payload["ready_for_synthesis"] is False


def test_ingest_dry_runs_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    config = _write_config(tmp_path, measurements=False)
    store = TrajectoryStore(tmp_path / "history.jsonl")
    store.append_move(
        _publish(utm_content="c0412ab9"), posted_at=_TODAY.isoformat()
    )
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)

    def no_sdk(self: object, slug: str, arguments: dict) -> dict:
        raise AssertionError("ingest without credentials must not call the SDK")

    monkeypatch.setattr(
        "blotto.adapters.composio_io.ComposioAdapter._execute", no_sdk
    )
    assert app(["ingest", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out


def test_ingest_live_refuses_unmeasured_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """A live ingest with no measurement on file refuses with the pointer
    to docs/OPERATING.md rather than guessing -- exit 1, not a traceback."""
    config = _write_config(tmp_path, measurements=False)
    store = TrajectoryStore(tmp_path / "history.jsonl")
    store.append_move(
        _publish(utm_content="c0412ab9"), posted_at=_TODAY.isoformat()
    )
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    monkeypatch.setenv("COMPOSIO_API_KEY", "cs-live")
    code = app(["ingest", "--config", str(config)])
    err = capsys.readouterr().err
    assert code == 1
    assert "docs/OPERATING.md" in err


def test_plan_without_a_model_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path)
    assert app(["plan", "--config", str(config), "--days", "3"]) == 1
    assert "world model" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_example_toml_at_repo_root_parses_and_matches_example() -> None:
    shipped = _REPO_ROOT / "blotto.example.toml"
    assert shipped.exists(), "blotto.example.toml ships at the repo root"
    text = shipped.read_text(encoding="utf-8")
    assert text == example(), "the shipped file and example() must not drift"
    config = load(shipped)
    assert config.economics.arpu_monthly == 100.0
    assert config.destination_url is not None
    assert config.attribution_coverage is None, (
        "the shipped example must NOT contain invented measurements"
    )
    assert config.incrementality is None


def test_load_applies_policy_and_measurement_tables(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        """\
[economics]
arpu_monthly = 250.0
gross_margin = 0.9
monthly_churn = 0.05
cogs_share = 0.1
fixed_cost_per_post = 75.0

[policy]
creative_judgement_min_conversions = 80

[policy.max_per_day_by_format]
text_thread = 3

[measurement]
attribution_coverage = 0.81
incrementality = 0.65
""",
        encoding="utf-8",
    )
    config = load(path)
    assert config.economics.arpu_monthly == 250.0
    assert config.policy.creative_judgement_min_conversions == 80
    assert config.policy.max_per_day_by_format[Format.TEXT_THREAD] == 3
    assert config.policy.max_per_day_by_format[Format.CAROUSEL] == 10, (
        "unset caps keep the corpus defaults"
    )
    assert config.attribution_coverage == 0.81
    assert config.incrementality == 0.65


def test_load_rejects_unknown_keys_and_range_errors(tmp_path: Path) -> None:
    bad_key = tmp_path / "a.toml"
    bad_key.write_text("[economics]\narp = 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load(bad_key)

    bad_range = tmp_path / "b.toml"
    bad_range.write_text("[measurement]\nattribution_coverage = 72\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fraction"):
        load(bad_range)


def test_load_names_the_required_python_version_when_no_toml_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an interpreter with neither tomllib nor tomli, the error names
    Python 3.11 rather than surfacing an ImportError from the stdlib."""
    config = tmp_path / "c.toml"
    config.write_text("[economics]\narpu_monthly = 1.0\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    with pytest.raises(RuntimeError, match="3.11"):
        load(config)
