# Running the loop

What you actually do on a Monday.

`blotto` is not a content generator with a planner bolted on. It is a planner
that happens to need content as its action space. The distinction shows up in
the daily routine: most of the work is maintaining a world model that predicts
your results, and the plan falls out of that almost for free.

---

## The cold start problem, stated honestly

**You cannot synthesise a world model from nothing.** The reference paper works
from five trajectories per game, but a game trajectory is complete and settled,
whereas a distribution trajectory is partial for the first 24–72 hours and
partially unobservable forever.

A workable floor is **30 published items with settled metrics**, spanning at
least two archetypes and two platforms. Below that, synthesis will produce a
model that fits your history and predicts nothing, and `blotto accuracy` will
show it: high train accuracy, poor test accuracy. That gap is the diagnostic,
which is exactly why the report separates them.

If you are below the floor, the honest move is to run without a model: use
`solvers/blotto.py` for allocation and `solvers/exp3.py` for angle selection —
neither needs a world model — and start recording trajectories. Come back when
you have history worth learning from.

---

## Daily

```
blotto ingest      # pull yesterday's metrics; mark anything inside the lag partial
blotto plan        # ISMCTS over the next N days, inside the current world model
blotto brief       # emit the plan as content briefs
                   # -- you or your pipeline produce the actual creative --
blotto publish     # ship via Composio, stamping each item's utm_content
```

Two things to notice.

**`plan` never calls a language model.** Synthesis happens weekly; planning is
pure search. If your plan step is making API calls, the architecture has
collapsed back into prompting and you have lost the property that made it worth
building.

**`publish` refuses illegal moves.** Not warns — refuses. If the planner asked
for something the legality engine forbids, that is a bug in the planner and you
want it loud rather than shipped.

---

## Weekly

```
blotto synth       # re-synthesise the world model from updated history
blotto accuracy    # train / test / online, transition and inference
blotto arena       # candidate strategies play inside the models
```

`synth` writes two artefacts side by side: the world model and an
`inference.py` sampler that `plan` determinizes with. If the sampler fails to
synthesise or load, `plan` says `open-loop` in its output — a weaker search,
not a broken one, and worth re-running `synth` to fix.

Re-synthesis cadence is a genuine open question and weekly is a guess. The
reference paper synthesises once, offline, and names online learning as future
work. Distribution moves faster than a board game's rules, so more often is
probably better, and the cost is LLM calls. Watch the drift: if online accuracy
falls well below test accuracy, the world has moved and the model has not.

**Read `accuracy` before trusting `plan`.** A model that cannot predict last
month will not predict next month, and a confident plan from a bad model is
worse than no plan, because it is actionable.

---

## Before you trust any of these numbers

There is one test in the suite worth knowing about by name:
`test_reference_model_scores_one_against_its_own_trajectories`.

It takes the hand-written reference model, generates trajectories from it, turns
those into unit tests, and runs them back against the same model. The score must
be exactly 1.00. A measuring instrument has to read zero on a known-zero input
before any reading it gives you means anything.

It did not, at first. It read 0.20. Two bookkeeping mistakes — replay re-rolling
the chance player instead of replaying what was recorded, and two posts sharing
a `utm_content` — meant a *perfect* model could not score above roughly 0.35.
Every downstream number inherited it: refinement could never reach its early
stop, so a run would burn its whole call budget, and accuracy sat on a scale
whose maximum nobody knew.

Neither mistake was visible from the outputs. Both were visible immediately from
this test, once it existed. If you fork this and change anything in
`cwm/tests_from_traj.py` or `cwm/reference.py`, that test is the one to watch.

## What good looks like

| Metric | Healthy | Worrying |
|---|---|---|
| Transition accuracy, test | > 0.90 | < 0.75 — you are in Gin rummy territory |
| Train minus test gap | < 0.05 | > 0.15 — memorising, not modelling |
| Test minus online gap | < 0.05 | > 0.10 — the world moved since synthesis |
| Inference accuracy | > 0.85 | < 0.60 — determinizations are fiction, so is the plan |
| LLM calls to converge | < 50 | approaching 500 — the domain is resisting synthesis |

The paper's own worst case sat at 0.78 train and 0.75 test after exhausting 500
calls. That is what failure looks like, and it is a legitimate result to get.
The system is built to show it to you rather than to route around it.

---

## What to do when the model is bad

In order of preference.

1. **Add trajectories.** Most synthesis failures are data-poverty failures.
2. **Narrow the game.** One platform, one avatar. The paper's accuracy is near
   perfect on small action spaces and degrades on large ones; ours is no
   different.
3. **Shorten the horizon.** Thirty days of compounding cohort effects is a lot
   to ask of code written from a few dozen examples. Seven is easier.
4. **Fall back to the model-free solvers.** Blotto allocation and EXP3 angle
   selection need no world model and still beat picking by feel.
5. **Do not lower the tolerance until the tests pass.** That produces a model
   that reports success and predicts nothing, which is strictly worse than a
   model that reports failure.

---

## The thing that will bite you

**Attribution coverage.** Everything downstream depends on it, and it is the
number operators are least likely to have measured.

If you do not know your coverage, you do not know your conversions, and every
reward the model learns from is wrong by an unknown factor. Measure it before
anything else: run a holdout, or reconcile a post-purchase "how did you hear
about us" survey against your analytics for a month. The GTM corpus records the
usual result — the dashboard says direct, and 40% of respondents name a channel
the dashboard cannot see.

Until then the legality engine will keep refusing to scale or kill anything
below the coverage floor, and it is right to. Unjudgeable is not the same as
bad, and an operator who overrides that rule is not gaining information — only
confidence.
