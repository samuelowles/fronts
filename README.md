# blotto

**Plan content distribution as an imperfect-information game — inside a world
model a language model wrote for you.**

Most AI distribution tooling asks a model to *be* a marketer: here is my
product, write me ten hooks. `blotto` asks it to do something narrower and far
more useful — read your posting history and **write the simulator**. Then it
plans inside that simulator with Information Set MCTS, the same way an engine
plays poker.

Based on [*Code World Models for General Game Playing*](https://arxiv.org/abs/2510.04542)
(Lehrach et al., Google DeepMind, 2025), applied to a domain the paper does not
discuss.

---

## The argument in three paragraphs

**Asking a language model for content is asking it to be a policy.** The paper's
objection is that this "relies on the model's implicit fragile pattern-matching
capabilities, leading to frequent illegal moves and strategically shallow play."
In board games an illegal move forfeits. In distribution it is an
unsubstantiated FTC claim, an undisclosed AI face on TikTok, or a budget
committed on five conversions of evidence. Shallow play is posting today with no
model of what today does to tomorrow's reach.

**So change the model's job.** Give it your platform policies in prose and your
last few dozen posts with the metrics that came back, and have it emit a Python
simulator of your distribution environment: a transition function, a legal-action
function, an observation function with real attribution loss in it, and a reward
function denominated in paying users. Then never call the language model again
during planning. Search does the playing, and search converts compute into
strength in a way prompting does not.

**And take the game theory seriously, because the paper doesn't.** It uses
extensive-form games purely as planning scaffolding — no equilibria, no
mechanism design, no signalling. But distribution genuinely is strategic: your
daily output is a fixed force allocated across contested fronts (Colonel
Blotto), angles decay in value as rivals crowd them (congestion games), the
ranking algorithm drifts adversarially (EXP3, not UCB), it commits before you
move and never tells you what to (Stackelberg), and a claim is believable only
when it is expensive to fake (Spence). Those five solvers are ours.

---

## What makes it different

**The reward is paying users.** Reach does not appear in the objective function
at all. It appears in the *observation*, because a hook rate below 25% tells you
the platform stopped serving your asset — but never in the reward. The corpus
this is built from records the reason: the cheapest acquisition vector
($30 CAC) produced 45% three-month churn and $250 LTV, while the most expensive
($65 CAC) produced 8% churn and $1,200 LTV. **Anything minimising CAC picks the
wrong one, and gets more confident with more data.**

**Scaling too early is impossible, not discouraged.** `Scale` is a first-class
move with an evidence gate in front of it: 50 settled conversions in a trailing
7-day window before an angle may be called a winner, 300 conversions and 14 days
before budget follows. Observations inside the 24–72 hour reporting lag count
for zero. This is a *gate*, not a penalty — the planner cannot select the move
at any confidence, because it never appears in the legal action set.

**Data you mostly cannot see makes an angle unjudgeable, not noisy.** Below the
attribution-coverage floor, `Scale` and `Kill` are *both* illegal. You may not
conclude an angle is winning and you may not conclude it is losing. This is the
least intuitive rule in the system and the one operators most reliably get wrong
in the confident direction.

**Every number carries a citation.** `game/priors.py` holds the empirical bands,
and each one has a `source` field naming the file it came from. A prior without
a source is not permitted to exist, and the self-check enforces that
programmatically rather than by inspection.

---

## Architecture

```
  platform policy (prose)  ─┐
  your posting history     ─┼──►  cwm/synth  ──►  candidate world models
  benchmark priors         ─┘         │              (Python, executable)
                                      │
                          unit tests generated from
                          your real trajectories
                                      │
                          Thompson-sampled refinement
                          (REx tree search, C = 5.0)
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  Code World Model │  apply_action
                            │  dict state       │  get_legal_actions  ◄── legality/
                            │  OpenSpiel-shaped │  get_observations   ◄── attribution loss
                            └──────────────────┘  get_rewards        ◄── contribution margin
                                      │
        cwm/inference ────────────────┤  resample_state: sample a ranking-weight
        (closed deck)                 │  vector your results do not contradict
                                      ▼
                            ┌──────────────────┐
                            │  solvers/ismcts   │  plan, never prompt
                            └──────────────────┘
                                      │
      blotto · exp3 · congestion · signalling · stackelberg
                                      │
                                      ▼
                          adapters/composio → post, pull analytics
                                      │
                              new trajectories ──┐
                                      └──────────┘
```

---

## Status

The game, world-model and solver layers are built and tested — 165 tests, no
runtime dependencies, `ruff` clean. **The Composio adapter and the CLI are not
built yet**, so the commands below describe the intended surface rather than
something you can run today; `examples/` and `blotto plan` are the next
milestone. Everything under `src/blotto/` works and is exercised by the suite.

The sandbox that executes synthesised code was independently reviewed and found
fully escapable in its first form — an allowlist of *modules* cannot hold,
because module objects form a reachable graph and `random._os` is the real `os`.
It now refuses imports outright and binds values instead, with the nine verified
escapes kept as regression tests. Anything running genuinely untrusted synthesis
output should still use `SubprocessSandbox` or a container.

## Install

```bash
pip install blotto              # core: zero dependencies
pip install "blotto[all]"       # + LLM providers, Composio, CLI
```

The game, world-model and solver layers depend on nothing outside the standard
library. That is deliberate: a planner that needs numpy is a planner you cannot
drop into a Lambda, and a repository that needs a wheel built before it will
tell you anything is a repository nobody evaluates.

---

## Quickstart

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
export COMPOSIO_API_KEY=...       # platform connections

blotto synth --history data/trajectories.jsonl --rules rules.md
blotto accuracy                   # transition + inference, train/test/online
blotto plan --days 7 --sims 1000
blotto arena                      # strategies play inside the models, before you spend
```

`blotto arena` is the one to notice. Following the paper's bad-sample rejection,
it synthesises several world models, has the resulting content strategies play
each other using one model as host in place of ground truth you do not have, and
discards any that lose by more than 10% of the observed utility range — **before
you commit a day of real output.**

---

## Honesty about what this is

The reference paper's worst result is Gin rummy: 0.78 train and 0.75 test
transition accuracy after exhausting a 500-call synthesis budget, and a heavy
loss to a ground-truth opponent. Its diagnosis is that games with "intricate,
multi-step procedural subroutines" resist synthesis.

Attribution windows, 24–72 hour reporting lag, and compounding cohort effects
are exactly that kind of subroutine. **Expect distribution to behave more like
Gin rummy than like tic-tac-toe.** That is why `blotto accuracy` reports
transition and inference accuracy separately for train, test and online, and why
you should look at it before trusting a plan. A world model that cannot predict
your last month is not going to predict your next one.

`docs/GAME.md` §10 lists what the model is known to get wrong, and §9 rates each
solution concept for how well it is actually justified. The signalling module in
particular is an analogy resting on assumptions that do not cleanly hold, and
says so in its own docstring.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/GAME.md`](docs/GAME.md) | The formal game: players, hidden state, observation model, payoffs, solution concepts, known failures |
| [`docs/PAPER.md`](docs/PAPER.md) | The Code World Models translation — what transfers, what doesn't, and two errata in the original |
| [`docs/OPERATING.md`](docs/OPERATING.md) | The daily and weekly loop, the cold-start floor, and what to do when the model is bad |

---

## Licence

MIT. Built by [owles.works](https://owles.works).

Never automates spam, evasion, fake engagement, scraping behind a login, or
posting to accounts you do not own. The legality engine exists to make several
of those structurally impossible rather than merely discouraged.
