# Contributing

The short version: read `AGENTS.md`. It is written for coding agents but the
rules are the same for humans, and several of them exist because breaking them
produced bugs that passed the entire test suite.

## Setup

```bash
git clone https://github.com/owles-works/blotto
cd blotto
python -m pip install -e ".[dev]"
```

The core has no runtime dependencies, so `pip install -e .` with no extras is a
real test — if anything under `src/` grows a third-party import, that install
still succeeds and CI's import job is what catches it.

## Before you open a PR

```bash
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python scripts/selfcheck.py
PYTHONPATH=src python examples/walkthrough.py
ruff check .
mypy src/blotto
```

## What gets merged quickly

**A sandbox escape with a proof of concept.** `src/blotto/cwm/sandbox.py`
executes code an LLM wrote. Its first version had nine working escapes, found by
review, and the fix was architectural rather than a patch — see `SECURITY.md`.
If you can defeat the current one, a failing test in the style of
`test_sandbox_refuses_known_module_graph_escapes` is the most valuable thing you
can send, and it lands with credit.

**A benchmark with a citation.** Every number in `game/priors.py` names the file
it came from. If you have measured figures for one of the three emotional
vectors currently set to `None`, that is a genuine contribution — provided the
source is real and named. A plausible-looking guess is worse than the `None`,
because the `None` is honest.

**A failure case.** `docs/GAME.md` §10 lists what the model is known to get
wrong. Additions to that list are as welcome as fixes.

**A solver.** `solvers/` holds general algorithms with their literature origin
named in the docstring. Repeated-game / folk-theorem treatment of audience trust
is the obvious missing one, and `docs/GAME.md` §10 says why.

## What will get pushback

**Widening a tolerance to make a test pass.** If the reference model cannot
score 1.00 against its own trajectories, the bug is in the model or the replay,
never in the threshold. This exact shortcut hid a broken measurement through the
whole first draft.

**Turning an evidence gate into a penalty or a warning.** A gate returns
`legal=False`. A penalty is a term in an objective, and a sufficiently confident
planner will simply pay it.

**Adding reach, engagement, or attributed conversions to the reward function.**
They are evidence, not payoff. The omission is deliberate and commented.

**Importing `game/priors`, `legality`, `payoff` or `action_space` into a
solver.** CI fails on it. A solver is a general algorithm; the constants belong
to the caller.

**An empirical claim without a source**, anywhere.

**An LLM call inside a planning loop.** Synthesis is where the model is used.
Planning is search. Mixing them dissolves the only property that makes this
architecture worth having.

## Style

Docstrings explain *why*, cite where a source exists, and state plainly what a
guarantee does not cover. Do not write comments that restate the line below
them. Full type annotations on public signatures. Line length 100.

Prose in docs and docstrings should be specific enough to be wrong. "Improves
performance" is not a claim; "0.20 to 1.00 against its own trajectories at
tolerances tightened from 0.25 to 0.05" is.

## Licence

MIT. By contributing you agree your work ships under it.
