"""Unit tests for the adapter layer: Composio I/O and the trajectory store.

The adapter is where the system meets the outside world, so the tests here
are mostly about refusal and survival rather than happy paths: no SDK import
without a live call, no I/O in a dry run, no guessed attribution numbers, no
half-written history after a crash. An adapter that is wrong in any of these
ways poisons quietly rather than failing loudly, which is worse.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from fronts.adapters.composio_io import (
    ComposioAdapter,
    ComposioConfig,
    DryRunAdapter,
    MissingMeasurementError,
    RecordedAdapter,
    ToolkitRegistry,
    stamp_utm,
)
from fronts.adapters.trajectory import (
    COLD_START_MIN_ARCHETYPES,
    COLD_START_MIN_PLATFORMS,
    COLD_START_MIN_SETTLED,
    TrajectoryStore,
)
from fronts.game.types import (
    ActionKey,
    Archetype,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    Hold,
    HookFamily,
    Kill,
    Observation,
    Platform,
    Publish,
    Scale,
    SemanticTier,
    Step,
    Trajectory,
)

pytestmark = pytest.mark.unit

_NOW = datetime.now()
"""Wall clock for fixture timestamps. Naive on purpose: the store's
``_normalise`` and the adapter's ``_parse_timestamp`` each put timestamps
into one frame before comparing, so fixture times never need to be aware."""


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


def _observation(**overrides: object) -> Observation:
    fields: dict[str, object] = {
        "utm_content": "c0412ab9",
        "posted_at": (_NOW - timedelta(days=5)).date().isoformat(),
        "observed_at": _NOW.date().isoformat(),
        "reach": 1000,
        "impressions": 1500,
        "hook_rate": 0.33,
        "hold_rate": 0.38,
        "attributed_conversions": 12,
        "attribution_coverage": 0.72,
        "incrementality": 0.70,
    }
    fields.update(overrides)
    return Observation(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module hygiene: the SDK may not be imported at import time.
# ---------------------------------------------------------------------------


def test_importing_every_fronts_module_never_imports_the_sdk() -> None:
    """The core must import on a machine with no composio SDK installed:
    importing every module in the package is the check."""
    import fronts

    sys.modules.pop("composio", None)
    for module_info in _walk(fronts):
        importlib.import_module(module_info)
    offenders = [
        name for name in sys.modules if name == "composio" or name.startswith("composio.")
    ]
    assert not offenders, f"module-level SDK import leaked: {offenders}"


def _walk(package: types.ModuleType) -> list[str]:
    import pkgutil

    return [
        name
        for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    ]


def test_sdk_import_is_inside_execute_not_at_module_scope() -> None:
    """The lazy import lives in the one chokepoint; a grep-shaped check that
    the module source contains no top-level SDK import."""
    source = Path(sys.modules["fronts.adapters.composio_io"].__file__).read_text(
        encoding="utf-8"
    )
    body = source.split('"""', 2)[2]  # skip the module docstring
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("import composio") or stripped.startswith("from composio"):
            # Allowed only when indented inside a function (the lazy import).
            assert line.startswith((" ", "\t")), f"module-level SDK import: {line}"


# ---------------------------------------------------------------------------
# A fake composio module, so the live adapter's full path runs offline.
# ---------------------------------------------------------------------------


class _FakeExecuteResponse:
    """Mimics an SDK result object: not a dict, exposes .data, the shape
    _as_dict exists to normalise."""

    def __init__(self, data: dict) -> None:
        self.data = data


class _FakeTools:
    def __init__(self, behavior) -> None:
        self._behavior = behavior
        self.calls: list[tuple[str, dict, str]] = []

    def execute(self, slug: str, arguments: dict, user_id: str):
        self.calls.append((slug, arguments, user_id))
        return self._behavior(slug)


class _FakeComposio:
    instances: list[_FakeComposio] = []

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.behavior = lambda slug: {"id": "post-1"}
        self.tools = _FakeTools(lambda slug: {"id": "post-1"})
        _FakeComposio.instances.append(self)


def _install_fake_composio(monkeypatch: pytest.MonkeyPatch) -> list[_FakeComposio]:
    """Put a fake 'composio' module in sys.modules; `from composio import
    Composio` inside _execute resolves to it without any install."""
    _FakeComposio.instances = []
    fake_module = types.ModuleType("composio")
    fake_module.Composio = _FakeComposio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "composio", fake_module)
    return _FakeComposio.instances


# ---------------------------------------------------------------------------
# ComposioAdapter
# ---------------------------------------------------------------------------


def test_publish_goes_through_execute_and_stamps_utm(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_composio(monkeypatch)
    adapter = ComposioAdapter(ComposioConfig(api_key="cs-key", user_id="user-123"))
    receipt = adapter.publish(
        _publish(), {"text": "the thread", "url": "https://owles.works/pricing?utm_source=x"}
    )
    assert len(instances) == 1
    assert instances[0].api_key == "cs-key"
    slug, arguments, user_id = instances[0].tools.calls[0]
    assert slug == "TWITTER_CREATION_OF_A_POST"
    assert user_id == "user-123"
    assert arguments["utm_content"] == "c0412ab9"
    assert "utm_content=c0412ab9" in arguments["link"]
    assert "utm_source=x" in arguments["link"], "existing params must survive stamping"
    assert not receipt.dry_run
    assert receipt.response_id == "post-1"
    assert receipt.url == arguments["link"]


def test_publish_is_never_retried_even_on_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retried publish that had landed is a duplicate post; an exception
    after the request left the process says nothing about whether it did."""
    _install_fake_composio(monkeypatch)

    def always_timeouts(slug: str):
        raise TimeoutError("connection timed out")

    adapter = ComposioAdapter(ComposioConfig(api_key="k", max_retries=3))
    adapter._client = _FakeComposio()
    adapter._client.tools = _FakeTools(always_timeouts)
    monkeypatch.setattr("fronts.adapters.composio_io.time.sleep", lambda _: None)
    with pytest.raises(TimeoutError):
        adapter.publish(_publish(), {"text": "t", "url": "https://owles.works/"})
    assert len(adapter._client.tools.calls) == 1, "a publish must not be retried"


def test_reads_retry_with_backoff_on_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analytics is a read: a duplicate request costs a round trip and
    nothing else, so transient failures are retried with backoff."""
    _install_fake_composio(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(
        "fronts.adapters.composio_io.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    attempts: list[int] = []

    def flaky(slug: str):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("connection reset")
        return {
            "metrics": {
                "posted_at": (_NOW - timedelta(days=5)).isoformat(),
                "reach": 500,
                "attributed_conversions": 4,
            }
        }

    adapter = ComposioAdapter(
        ComposioConfig(api_key="k", max_retries=3),
        attribution_coverage=0.72,
        incrementality=0.70,
    )
    adapter._client = _FakeComposio()
    adapter._client.tools = _FakeTools(flaky)
    observations = adapter.fetch_analytics(
        ["c0412ab9"], _NOW - timedelta(days=7), platforms={"c0412ab9": Platform.X}
    )
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0], "exponential backoff, doubling from the base"
    assert observations[0].reach == 500


def test_fetch_analytics_refuses_when_coverage_unmeasured() -> None:
    """The refusal fires before any network call: a misconfigured operator
    learns it at ingest time, not after a round of calls whose results
    could not be used anyway."""
    adapter = ComposioAdapter(ComposioConfig())
    with pytest.raises(MissingMeasurementError, match="docs/OPERATING.md"):
        adapter.fetch_analytics(["c0412ab9"], _NOW, platforms={"c0412ab9": Platform.X})


def test_fetch_analytics_refuses_when_incrementality_unmeasured() -> None:
    adapter = ComposioAdapter(
        ComposioConfig(), attribution_coverage=0.72, incrementality=None
    )
    with pytest.raises(MissingMeasurementError) as excinfo:
        adapter.fetch_analytics(["c0412ab9"], _NOW, platforms={"c0412ab9": Platform.X})
    assert "incrementality" in str(excinfo.value)
    assert "docs/OPERATING.md" in str(excinfo.value)


def test_fetch_analytics_never_defaults_measurements_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the operator measured is what lands on the Observation --
    not 1.0, and not anything the adapter invented."""
    _install_fake_composio(monkeypatch)

    def metrics(slug: str):
        return {
            "metrics": {
                "posted_at": (_NOW - timedelta(days=5)).isoformat(),
                "reach": 1000,
                "attributed_conversions": 10,
            }
        }

    adapter = ComposioAdapter(
        ComposioConfig(api_key="k"),
        attribution_coverage=0.72,
        incrementality=0.70,
    )
    adapter._client = _FakeComposio()
    adapter._client.tools = _FakeTools(metrics)
    (observation,) = adapter.fetch_analytics(
        ["c0412ab9"], _NOW - timedelta(days=7), platforms={"c0412ab9": Platform.X}
    )
    assert observation.attribution_coverage == 0.72
    assert observation.incrementality == 0.70


def test_fetch_analytics_marks_fresh_observations_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the measured reporting lag an observation is partial, and a
    partial observation must never open an evidence gate."""
    _install_fake_composio(monkeypatch)
    fresh = (_NOW - timedelta(hours=2)).isoformat()
    settled = (_NOW - timedelta(days=5)).isoformat()

    def stub_for(when: str):
        def behavior(slug: str):
            return {"metrics": {"posted_at": when, "reach": 10}}

        return behavior

    adapter = ComposioAdapter(
        ComposioConfig(api_key="k"), attribution_coverage=0.72, incrementality=0.7
    )
    adapter._client = _FakeComposio()
    adapter._client.tools = _FakeTools(stub_for(fresh))
    (partial,) = adapter.fetch_analytics(
        ["fresh1"], _NOW - timedelta(days=7), platforms={"fresh1": Platform.X}
    )
    assert partial.is_partial

    adapter._client = _FakeComposio()
    adapter._client.tools = _FakeTools(stub_for(settled))
    (done,) = adapter.fetch_analytics(
        ["settled1"], _NOW - timedelta(days=7), platforms={"settled1": Platform.X}
    )
    assert not done.is_partial


def test_fetch_analytics_requires_platform_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The analytics toolkit cannot be guessed from a utm id; refusing the
    call is safer than querying the wrong platform."""
    _install_fake_composio(monkeypatch)
    adapter = ComposioAdapter(
        ComposioConfig(api_key="k"), attribution_coverage=0.72, incrementality=0.7
    )
    with pytest.raises(ValueError, match="platforms"):
        adapter.fetch_analytics(["c0412ab9"], _NOW)
    with pytest.raises(ValueError, match="no platform known"):
        adapter.fetch_analytics(["c0412ab9"], _NOW, platforms={"other": Platform.X})


def test_non_publish_moves_are_a_typeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_composio(monkeypatch)
    adapter = ComposioAdapter(ComposioConfig(api_key="k"))
    with pytest.raises(TypeError):
        adapter.publish(Scale("launch_theater", 1.1), {"url": "https://owles.works/"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DryRunAdapter().publish(Hold(), {"url": "u"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ToolkitRegistry
# ---------------------------------------------------------------------------


def test_registry_is_marked_unverified_and_complete() -> None:
    """A slug table nobody has checked against a live account must say so in
    its own docstring, or the flag lives in a commit message instead."""
    assert "UNVERIFIED" in (ToolkitRegistry.__doc__ or "")
    assert set(ToolkitRegistry.ENTRIES) == set(Platform)


def test_every_slug_is_overridable() -> None:
    registry = ToolkitRegistry()
    overrides = {
        f"{kind}:{platform.value}": f"CUSTOM_{kind.upper()}_{platform.value.upper()}"
        for platform in Platform
        for kind in ("publish", "analytics")
    }
    for platform in Platform:
        entry = registry.entry(platform, overrides)
        assert entry.publish == overrides[f"publish:{platform.value}"]
        assert entry.analytics == overrides[f"analytics:{platform.value}"]
    assert registry.entry(Platform.X).publish == "TWITTER_CREATION_OF_A_POST"


# ---------------------------------------------------------------------------
# stamp_utm
# ---------------------------------------------------------------------------


def test_stamp_utm_replaces_and_preserves() -> None:
    assert (
        stamp_utm("https://owles.works/", "c0412ab9")
        == "https://owles.works/?utm_content=c0412ab9"
    )
    assert (
        stamp_utm("https://owles.works/p?utm_source=newsletter&id=7", "deadbeef")
        == "https://owles.works/p?utm_source=newsletter&id=7&utm_content=deadbeef"
    )
    assert (
        stamp_utm("https://owles.works/p?utm_content=old", "new1")
        == "https://owles.works/p?utm_content=new1"
    ), "two values for one key is undefined downstream; replace, never append"


# ---------------------------------------------------------------------------
# DryRunAdapter
# ---------------------------------------------------------------------------


def test_dry_run_adapter_performs_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """With open() and the socket module poisoned, every adapter method must
    still work: a dry run that touches the network or disk is not a dry run."""

    def refuses(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run attempted I/O")

    monkeypatch.setattr("builtins.open", refuses)
    monkeypatch.setattr("socket.socket", refuses)
    adapter = DryRunAdapter()
    receipt = adapter.publish(_publish(), {"text": "t", "url": "https://owles.works/"})
    assert receipt.dry_run
    assert receipt.action == "(dry-run)"
    assert receipt.url.endswith("utm_content=c0412ab9")
    assert adapter.calls and adapter.calls[0]["action"] == "publish"
    assert (
        adapter.fetch_analytics(["c0412ab9"], _NOW, platforms={"c0412ab9": Platform.X})
        == []
    ), "a dry run observes nothing and invents nothing"
    assert adapter.calls[-1]["action"] == "fetch_analytics"


# ---------------------------------------------------------------------------
# RecordedAdapter
# ---------------------------------------------------------------------------


def _write_fixture(path: Path) -> None:
    entries = [
        {
            "utm_content": "c0412ab9",
            "response": {
                "metrics": {
                    "posted_at": (_NOW - timedelta(days=5)).isoformat(),
                    "reach": 1200,
                    "hook_rate": 0.31,
                    "attributed_conversions": 9,
                }
            },
        },
        {
            "utm_content": "c07e33aa",
            "response": {
                "metrics": {
                    "posted_at": (_NOW - timedelta(hours=3)).isoformat(),
                    "reach": 80,
                    "attributed_conversions": 1,
                }
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_recorded_adapter_replays_a_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "composio.jsonl"
    _write_fixture(fixture)
    adapter = RecordedAdapter(
        fixture, attribution_coverage=0.72, incrementality=0.70
    )
    observations = adapter.fetch_analytics(
        ["c0412ab9", "c07e33aa", "missing"], _NOW - timedelta(days=7)
    )
    by_utm = {observation.utm_content: observation for observation in observations}
    assert by_utm["c0412ab9"].reach == 1200
    assert by_utm["c0412ab9"].attribution_coverage == 0.72
    assert not by_utm["c0412ab9"].is_partial
    assert by_utm["c07e33aa"].is_partial, "three hours old is inside the lag"
    assert "missing" not in by_utm, "no recorded entry means no data, not fiction"


def test_recorded_adapter_enforces_the_measurement_refusal(tmp_path: Path) -> None:
    fixture = tmp_path / "composio.jsonl"
    _write_fixture(fixture)
    adapter = RecordedAdapter(fixture)
    with pytest.raises(MissingMeasurementError, match="docs/OPERATING.md"):
        adapter.fetch_analytics(["c0412ab9"], _NOW)


def test_recorded_adapter_publish_replays_recorded_id(tmp_path: Path) -> None:
    fixture = tmp_path / "composio.jsonl"
    with open(fixture, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"utm_content": "c0412ab9", "response": {"id": "tw-99"}}) + "\n")
    adapter = RecordedAdapter(fixture)
    receipt = adapter.publish(_publish(), {"url": "https://owles.works/"})
    assert receipt.response_id == "tw-99"
    assert "utm_content=c0412ab9" in receipt.url


# ---------------------------------------------------------------------------
# TrajectoryStore
# ---------------------------------------------------------------------------


def _trajectory_with_moves() -> Trajectory:
    trajectory = Trajectory(account="test", notes="unit")
    trajectory.chance = [
        ActionKey("chance|response=mid|drift=flat"),
        ActionKey("chance|response=high|drift=up"),
    ]
    trajectory.steps = [
        Step(move=_publish(), observation=_observation()),
        Step(move=Scale("launch_theater", 1.10), observation=None),
        Step(move=Kill("enemy_of_slop", "burnout"), observation=None),
        Step(move=Hold(), observation=None),
    ]
    return trajectory


def test_moves_round_trip_through_the_store(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    original = _trajectory_with_moves()
    store.save(original)
    reloaded = store.load()
    assert reloaded.account == "test"
    assert reloaded.notes == "unit"
    assert reloaded.steps == original.steps


def test_chance_survives_save_and_load(tmp_path: Path) -> None:
    """Losing the recorded chance sequence turns every transition test into
    a coin flip: the reference model scored 0.20 against its own
    trajectories when replay re-rolled the dice."""
    store = TrajectoryStore(tmp_path / "t.jsonl")
    original = _trajectory_with_moves()
    store.save(original)
    assert store.load().chance == original.chance
    assert all(isinstance(key, str) for key in store.load().chance)


def test_attach_observation_joins_on_utm(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="c0412ab9"), posted_at="2026-08-01")
    store.attach_observation(
        _observation(utm_content="c0412ab9", attributed_conversions=12)
    )
    trajectory = store.load()
    assert trajectory.steps[0].observation is not None
    assert trajectory.steps[0].observation.attributed_conversions == 12


def test_latest_observation_for_a_utm_wins(tmp_path: Path) -> None:
    """Analytics pulls refresh numbers; the later line is the fresher
    reading, and a stale one must not overwrite it."""
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="c0412ab9"), posted_at="2026-08-01")
    store.attach_observation(_observation(attributed_conversions=3, reach=100))
    store.attach_observation(_observation(attributed_conversions=9, reach=140))
    observation = store.load().steps[0].observation
    assert observation is not None
    assert observation.attributed_conversions == 9


def test_unmatched_observations_are_counted_not_attached(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="c0412ab9"), posted_at="2026-08-01")
    store.attach_observation(_observation(utm_content="orphan01"))
    trajectory = store.load()
    assert trajectory.steps[0].observation is None
    assert store.stats().unmatched_utm == 1


def test_crash_mid_rewrite_leaves_previous_file_intact(
    tmp_path: Path,
) -> None:
    """The reason atomicity is tested rather than trusted: a truncated
    history is indistinguishable from a real one once a model is trained on
    it. Simulate death inside the fsync, before the replace."""
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.save(_trajectory_with_moves())
    before = store.path.read_bytes()

    with mock.patch("fronts.adapters.trajectory.os.fsync") as broken_fsync:
        broken_fsync.side_effect = OSError("simulated crash mid-write")
        with pytest.raises(OSError):
            store.save(_trajectory_with_moves())
    assert store.path.read_bytes() == before, "previous history must survive intact"
    assert not list(tmp_path.glob("*.tmp")), "no half-written temp litter"


def test_crash_before_replace_leaves_previous_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.save(_trajectory_with_moves())
    before = store.path.read_bytes()
    with mock.patch(
        "fronts.adapters.trajectory.os.replace", side_effect=OSError("power loss")
    ), pytest.raises(OSError):
        store.save(_trajectory_with_moves())
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_truncated_final_line_is_skipped(tmp_path: Path) -> None:
    """A crash mid-append leaves a cut line at the end; the reader skips
    exactly that line and keeps everything before it."""
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="c0412ab9"), posted_at="2026-08-01")
    with open(store.path, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "move", "act')  # torn write
    assert len(store.load().steps) == 1


def test_mark_settled_respects_the_lag(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="old00001"), posted_at="2026-07-01")
    store.append_move(_publish(utm_content="new00002"), posted_at="2026-08-30")
    old_posted = (_NOW - timedelta(days=5)).date().isoformat()
    fresh_posted = (_NOW - timedelta(hours=2)).date().isoformat()
    store.attach_observation(
        _observation(utm_content="old00001", posted_at=old_posted, is_partial=True)
    )
    store.attach_observation(
        _observation(utm_content="new00002", posted_at=fresh_posted, is_partial=True)
    )
    flipped = store.mark_settled(_NOW)
    assert flipped == 1, "only the observation past the longest measured lag settles"
    by_utm = {
        step.move.utm_content: step.observation
        for step in store.load().steps
        if isinstance(step.move, Publish)
    }
    assert by_utm["old00001"] is not None and not by_utm["old00001"].is_partial
    assert by_utm["new00002"] is not None and by_utm["new00002"].is_partial


def test_mark_settled_is_idempotent(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(), posted_at="2026-07-01")
    store.attach_observation(
        _observation(posted_at=(_NOW - timedelta(days=10)).date().isoformat())
    )
    assert store.mark_settled(_NOW) == 0, "already settled; nothing to rewrite"


def test_settled_only_drops_partial_and_unobserved(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="aaa11111"), posted_at="2026-08-01")
    store.append_move(_publish(utm_content="bbb22222"), posted_at="2026-08-02")
    store.append_move(Scale("launch_theater", 1.1), posted_at="2026-08-02")
    store.attach_observation(_observation(utm_content="aaa11111", is_partial=False))
    store.attach_observation(_observation(utm_content="bbb22222", is_partial=True))
    settled = store.settled_only()
    assert [step.move.utm_content for step in settled.steps] == ["aaa11111"]


def test_stats_reports_the_cold_start_floor(tmp_path: Path) -> None:
    """The floor from docs/OPERATING.md: 30 settled items spanning at least
    two archetypes and two platforms."""
    store = TrajectoryStore(tmp_path / "t.jsonl")
    trajectory = Trajectory(account="floor")
    for index in range(COLD_START_MIN_SETTLED):
        platform = (Platform.X, Platform.LINKEDIN)[index % 2]
        archetype = (Archetype.FOUNDER_TRAUMA, Archetype.UGC_REVIEW)[index % 2]
        utm = f"{index:08x}"
        trajectory.steps.append(
            Step(
                move=_publish(utm_content=utm, platform=platform, archetype=archetype),
                observation=_observation(
                    utm_content=utm,
                    posted_at=(_NOW - timedelta(days=5)).date().isoformat(),
                ),
            )
        )
    store.save(trajectory)
    stats = store.stats()
    assert stats.moves == COLD_START_MIN_SETTLED
    assert stats.settled_observations == COLD_START_MIN_SETTLED
    assert stats.archetypes == COLD_START_MIN_ARCHETYPES
    assert stats.platforms == COLD_START_MIN_PLATFORMS
    assert stats.ready_for_synthesis

    narrow = Trajectory(account="narrow")
    for index in range(COLD_START_MIN_SETTLED):
        utm = f"{index:08x}"
        narrow.steps.append(
            Step(
                move=_publish(utm_content=utm),  # one archetype, one platform
                observation=_observation(utm_content=utm),
            )
        )
    store.save(narrow)
    assert not store.stats().ready_for_synthesis, "30 items on one archetype is a narrow game"


def test_stats_counts_partials_separately(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append_move(_publish(utm_content="par11111"), posted_at="2026-08-01")
    store.append_move(_publish(utm_content="set11112"), posted_at="2026-08-01")
    store.attach_observation(_observation(utm_content="par11111", is_partial=True))
    store.attach_observation(_observation(utm_content="set11112", is_partial=False))
    stats = store.stats()
    assert stats.partial_observations == 1
    assert stats.settled_observations == 1
    assert not stats.ready_for_synthesis


def test_load_on_missing_file_is_an_empty_trajectory(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "absent.jsonl")
    assert store.load().steps == []
    stats = store.stats()
    assert stats.moves == 0
    assert not stats.ready_for_synthesis
