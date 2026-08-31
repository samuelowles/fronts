<h1 align="center">blotto</h1>

<p align="center">
  <strong>Plan content distribution as an imperfect-information game —<br>
  inside a world model a language model wrote for you.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-blue.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Zero runtime dependencies" src="https://img.shields.io/badge/runtime%20deps-0-brightgreen.svg">
  <a href="docs/GAME.md"><img alt="Formal game spec" src="https://img.shields.io/badge/docs-formal%20game%20spec-8957e5.svg"></a>
  <a href="https://arxiv.org/abs/2510.04542"><img alt="arXiv 2510.04542" src="https://img.shields.io/badge/arXiv-2510.04542-b31b1b.svg"></a>
</p>

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

**Every number carries a citation — and the citations are unverifiable, which
you should know.** `game/priors.py` holds the empirical bands, each with a
`source` field naming the file it came from, and a prior without one cannot
exist — the self-check enforces that programmatically. But those files are two
private operator corpora. You cannot open them. So the citations buy provenance,
not independent verification, and these are one operator's measurements in their
verticals at a point in time, not constants.

Treat them as a **starting prior** and replace them with your own once you have
settled history. The mechanism in this repository is verifiable from the
repository; the priors are not. Judge them separately.

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

> [!WARNING]
> **`blotto` executes Python that a language model wrote.** The sandbox was
> independently reviewed and found *fully escapable* in its first form — an
> allowlist of modules cannot hold, because module objects form a reachable
> graph and `random._os` is the real `os`. It now refuses imports outright and
> binds values instead, with the nine verified escapes kept as regression
> tests. Anything running genuinely untrusted synthesis output should still use
> `SubprocessSandbox` or a container. [`SECURITY.md`](SECURITY.md) has the full
> account, including the escapes.

> [!IMPORTANT]
> **You need roughly 30 published items with settled metrics before synthesis
> is worth attempting.** Below that floor you will get a model that fits your
> history and predicts nothing, and `blotto accuracy` will show it as a wide
> train/test gap. Run the model-free solvers — Blotto allocation and EXP3 angle
> selection — until you have history worth learning from.
> [`docs/OPERATING.md`](docs/OPERATING.md) covers the cold start.

Alpha. The API will move. Everything under `src/blotto/` is exercised by the
test suite, `ruff` and `mypy` are clean, and the core has zero runtime
dependencies.

## Install

```bash
git clone https://github.com/owles-works/blotto && cd blotto
pip install -e .                       # core: zero dependencies
python examples/walkthrough.py         # the whole loop, seconds, no keys needed
```

Add providers only when you want a real synthesis run:

```bash
pip install -e ".[all]"                # + Anthropic / OpenAI / Composio
```

The game, world-model and solver layers depend on nothing outside the standard
library. That is deliberate: a planner that needs numpy is a planner you cannot
drop into a Lambda, and a repository that needs a wheel built before it will
tell you anything is a repository nobody evaluates.

---

## See it run

No keys, no network, no setup. `examples/walkthrough.py` drives the whole loop
against a reference model in under half a minute, deterministically — the same
numbers on every run. Three excerpts, verbatim.

**The instrument reads zero on a known-zero input.** Tests generated from a
model's own history, run back against it:

```
3. The ground-truth test: pass rate must be exactly 1.00
==============================================================================
  tests generated from history: 243
  passed against the model that wrote it: 243
  pass rate: 1.00

  1.00, exactly. Tolerances are the tightened ones (counts 0.05,
  rates 0.02) -- wide tolerances here would only ever conceal a broken
  instrument. A synthesised model is now measured against this scale,
  whose maximum is finally KNOWN to be 1.00.
```

**Search, not prompting.** The concentration of visits *is* the confidence:

```
  action                                                         visits         value
  publish|tiktok|text_thread|founder_trauma|aspiration_status|tr    125       311,143
  publish|tiktok|carousel|founder_trauma|aspiration_status|triba      2       173,277
  hold                                                                1        70,513
```

**A refusal you can audit.** Not a warning, not a penalty — the move is absent
from the legal set:

```
  ATTEMPT: scale 'launch_theater' by 1.10x on 12 settled conversions
  ----------------------------------------------------------------------------
  REFUSED
    rule:   CREATIVE_JUDGEMENT
    reason: angle 'launch_theater' has 12 settled conversions in the trailing 7d; declaring a winner requires 50
    source: Storytelling Engineer/17_A_B_Testing_Story_Arcs_Statistical_Significance_in_Emotion.md
  ----------------------------------------------------------------------------

  And peeking does not help: 100 conversions inside the reporting
  lag plus 12 settled counts as 12 -- partials are dropped
  entirely, not prorated.
```

Every refusal carries the rule, the reason, and the file the threshold came
from. The CLI exits `2` on one, so a pipeline can tell "refused" from "broke".

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

Two more gaps, stated plainly. The shipped `blotto plan` determinizes
uniformly — the closed-deck state-inference sampler (`cwm/inference`) is
exercised by the test suite and available through the API, but not yet wired
into the CLI. And `blotto accuracy` reports the inference column as `n/a`
rather than inventing a number for it, for the same reason.

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
