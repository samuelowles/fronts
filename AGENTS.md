# AGENTS.md

Orientation for coding agents working in this repository. Read this before
changing anything. Several of the rules below exist because breaking them
produced bugs that passed every test.

## What this is in one paragraph

`fronts` models organic content distribution as a finite-horizon
imperfect-information game and plans inside a world model that an LLM writes
from posting history. The LLM's job is to produce the simulator, not the
policy. Search does the playing. If you find yourself adding an LLM call
inside a planning loop, stop: that is the architecture collapsing back into
the thing it replaced.

## Layer map and the dependency rule

```
protocols.py          the contract between our code and synthesised code
game/                 what moves exist, what they cost, which are forbidden
solvers/              general algorithms: ISMCTS, Blotto, EXP3, congestion, ...
cwm/                  synthesis, refinement, inference, arena, reporting
adapters/             Composio I/O and the trajectory store
```

`solvers/` must not import from `game/` beyond `types.py`. No `priors`, no
`legality`, no `payoff`, no `action_space`. A solver is a general algorithm,
and marketing constants arrive through an injected config dataclass. CI
enforces this with a grep. It is not a style preference: it is what keeps the
solvers testable against toy games and publishable as algorithms.

## Rules that are load-bearing

Never widen a tolerance to make a test pass. `Tolerance` defaults are 0.05
for counts and 0.02 for rates. They used to be 0.25 and 0.10, and the slack
was hiding a broken measurement: replay re-rolled the chance player instead of
replaying the recording, so a perfect model scored 0.20 against its own
trajectories. If the reference model cannot score 1.00, the bug is in the
model or the replay. It is never in the threshold.

`test_reference_model_scores_one_against_its_own_trajectories` is the canary.
It must read exactly 1.00 at horizons 10, 20 and 30 in both decks. A measuring
instrument has to read zero on a known-zero input before any reading it gives
you means anything. Watch it whenever you touch `cwm/tests_from_traj.py` or
`cwm/reference.py`.

Evidence gates return `legal=False`. They are never penalties. A penalty is a
term in an objective, and a confident planner will pay it. A gate removes the
move from the legal set, so the planner cannot select it at any confidence. If
you are tempted to soften a gate into a score, you have misunderstood what it
is for.

Partial observations count for nothing. An observation inside the 24-72h
reporting window never opens an evidence gate and never becomes a unit test.
Testing against provisional numbers teaches the model to predict noise.

Do not invent an empirical number. Every entry in `game/priors.py` carries a
`source` naming the corpus file it came from, and `scripts/selfcheck.py`
asserts that programmatically. If a value is not measured, it is `None` with a
comment saying so. Three of the five emotional vectors are `None` for this
reason.

No module objects in the sandbox namespace. See `SECURITY.md`. Adding a
convenience import for synthesised code reopens nine known escapes.

`utm_content` must be unique per published item. It is the join key between a
move and the observation that lands days later. Two posts sharing one collapse
into a single row and read as a 3x prediction error.

Reach is not in the reward function. It appears in `Observation` because it is
evidence about hidden state. It does not appear in `payoff.reward` because it
is not the thing being bought. The omission is deliberate and commented. Do
not "fix" it.

## Verification

```bash
PYTHONPATH=src python -m pytest tests/ -q      # the suite
PYTHONPATH=src python scripts/selfcheck.py     # stdlib-only, no pytest needed
ruff check .
PYTHONPATH=src python examples/walkthrough.py  # end-to-end, no API keys
```

`scripts/selfcheck.py` exists so the core guarantees survive in an environment
with nothing installed. Append checks via the `@check` decorator. Do not
restructure the runner.

## Style

Docstrings explain why, cite a source where one exists, and say plainly what
a guarantee does not cover. Do not write comments that restate the line below
them. Full type annotations on every public parameter and return. Line length
100. `from __future__ import annotations` at the top of every module.

Where a design decision came from the reference paper (arXiv:2510.04542, see
`docs/PAPER.md`), quote the paper rather than paraphrasing it. Where the
decision is ours, say so, and rate how well it is justified. `docs/GAME.md`
section 9 does this per solver, and the signalling module openly admits it
rests on an analogy.

## Where the thinking lives

- `docs/GAME.md`: the formal game, and section 10 lists what the model gets
  wrong
- `docs/PAPER.md`: what transfers from the paper, what does not, two errata
- `docs/OPERATING.md`: the daily loop and the cold-start floor
- `SECURITY.md`: the sandbox, including the escape class it originally had

If a change makes one of those documents wrong, the change is not finished.
