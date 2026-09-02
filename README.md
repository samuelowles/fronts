<h1 align="center">fronts</h1>

<p align="center">
  <strong>Plan content distribution as an imperfect-information game,<br>
  inside a world model a language model wrote from your posting history.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-blue.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Zero runtime dependencies" src="https://img.shields.io/badge/runtime%20deps-0-brightgreen.svg">
  <a href="docs/GAME.md"><img alt="Formal game spec" src="https://img.shields.io/badge/docs-formal%20game%20spec-8957e5.svg"></a>
  <a href="https://arxiv.org/abs/2510.04542"><img alt="arXiv 2510.04542" src="https://img.shields.io/badge/arXiv-2510.04542-b31b1b.svg"></a>
</p>

Most AI marketing tools ask a language model to write your content. `fronts`
gives it a narrower job: read your posting history and write a simulator of
your distribution environment. Planning then happens inside that simulator
with Information Set MCTS, the same family of search that plays poker. The
language model is never called while a plan is being computed.

The method comes from [*Code World Models for General Game
Playing*](https://arxiv.org/abs/2510.04542) (Lehrach et al., Google DeepMind,
2025). This repository applies it to a domain the paper does not cover.

---

## Why

Used as a policy, a language model picks moves by pattern-matching, and the
paper is blunt about how that goes: "frequent illegal moves and strategically
shallow play." In a board game an illegal move loses the game. In distribution
it is an unsubstantiated FTC claim, an undisclosed AI face on TikTok, or budget
committed on five conversions of evidence.

So the model gets a different job. You hand it your platform policies in prose
and a few dozen posts with the metrics that came back. It writes a Python
simulator: a transition function, a legal-action function, an observation
function that models real attribution loss, and a reward measured in paying
users. After that, search does the playing.

Distribution is also a real game, and the repository treats it as one. Your
daily output is a fixed force spread across contested fronts (Colonel Blotto).
Angles lose value as rivals crowd them (congestion games). The ranking
algorithm drifts (EXP3 rather than UCB). The platform commits before you move
(Stackelberg). A claim persuades only when it is expensive to fake (Spence).
All five solvers ship with the repository and run with no model at all.

## The rules that matter

The reward is paying users, not reach. Reach shows up in the observation
because a hook rate under 25% means the platform stopped serving your asset,
but it never enters the reward. The corpus behind the priors records why this
matters: a $30-CAC channel produced 45% three-month churn at $250 LTV, while a
$65-CAC channel produced 8% churn at $1,200 LTV. A system built to minimise
CAC picks the wrong channel and grows more confident as the data piles up.

Premature scaling is illegal rather than penalised. Calling an angle a winner
takes 50 settled conversions in a trailing 7-day window. Moving budget takes
300 conversions over 14 days. Anything still inside the 24-72 hour reporting
lag counts for zero. The planner cannot pick a gated move at any level of
confidence, because the move never enters the legal action set.

Low attribution coverage makes an angle unjudgeable. Below the coverage floor,
`Scale` and `Kill` are both illegal: when most of the data is missing you may
not conclude an angle is winning, and you may not conclude it is losing
either. Operators get this one wrong in the confident direction.

Every threshold carries a citation. `game/priors.py` names the source file for
each number, and the self-check fails if one is missing. The sources are two
private operator corpora, so a citation gives you provenance rather than
independent verification. Treat the numbers as a starting prior and replace
them once you have settled history of your own.

---

## Architecture

```
  platform policy (prose)  --+
  your posting history     --+-->  cwm/synth  -->  candidate world models
  benchmark priors         --+         |             (Python, executable)
                                       |
                            unit tests generated from
                            your real trajectories
                                       |
                            Thompson-sampled refinement
                            (REx tree search, C = 5.0)
                                       |
                                       v
                            +-------------------+
                            | Code World Model  |  apply_action
                            | dict state        |  get_legal_actions  <-- legality/
                            | OpenSpiel-shaped  |  get_observations   <-- attribution loss
                            +-------------------+  get_rewards        <-- contribution margin
                                       |
        cwm/inference -----------------+  resample_state: sample a ranking-weight
        (closed deck)                  |  vector your results do not contradict
                                       v
                            +-------------------+
                            |  solvers/ismcts   |  plan, never prompt
                            +-------------------+
                                       |
        blotto | exp3 | congestion | signalling | stackelberg
                                       |
                                       v
                          adapters/composio: post, pull analytics
                                       |
                                       +-->  new trajectories, back to the top
```

---

## Status

> [!WARNING]
> **`fronts` executes Python that a language model wrote.** The first sandbox
> was independently reviewed and found fully escapable: an allowlist of
> modules cannot hold, because module objects form a reachable graph and
> `random._os` is the real `os`. The current sandbox refuses imports outright
> and binds values instead, and the nine verified escapes are kept as
> regression tests. If you run synthesis output you do not trust, use
> `SubprocessSandbox` or a container. [`SECURITY.md`](SECURITY.md) has the
> full account, including the escapes.

> [!IMPORTANT]
> **Synthesis needs roughly 30 published items with settled metrics.** Below
> that floor the model fits your history and predicts nothing, and
> `fronts accuracy` shows it as a wide train/test gap. Until then, run the
> model-free solvers for allocation and angle selection.
> [`docs/OPERATING.md`](docs/OPERATING.md) covers the cold start.

Alpha. The API will move. Everything under `src/fronts/` is exercised by the
test suite, `ruff` and `mypy --strict` are clean, and the core has zero
runtime dependencies.

## Install

```bash
git clone https://github.com/samuelowles/fronts && cd fronts
pip install -e .                       # core: zero dependencies
python examples/walkthrough.py         # the whole loop, no keys needed
```

Provider SDKs are optional extras, only needed for a real synthesis run:

```bash
pip install -e ".[all]"                # Anthropic / OpenAI / Composio
```

The game, world-model and solver layers use only the standard library, so the
planner runs anywhere Python runs.

---

## See it run

The walkthrough needs no API keys and no network. It finishes in under half a
minute and prints the same numbers on every run. Three excerpts from its
output, quoted verbatim.

Tests generated from a model's own history score a perfect 1.00 against that
model. That check is what makes every later accuracy reading meaningful:

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

Search output, with visit counts. You can read the planner's confidence
directly from how the visits concentrate:

```
  action                                                         visits         value
  publish|tiktok|text_thread|founder_trauma|aspiration_status|tr    125       311,143
  publish|tiktok|carousel|founder_trauma|aspiration_status|triba      2       173,277
  hold                                                                1        70,513
```

A refusal, with the rule and the source file the threshold came from:

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

Refusals exit with code 2, so a pipeline can tell a refusal from a crash.

## Quickstart

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
export COMPOSIO_API_KEY=...       # platform connections

fronts synth --history data/trajectories.jsonl --rules rules.md
fronts accuracy                   # transition + inference, train/test/online
fronts plan --days 7 --sims 1000
fronts arena                      # strategies play inside the models
```

`fronts arena` synthesises several world models, has candidate strategies play
each other inside them, and rejects any strategy that loses by more than 10%
of the observed utility range. That filtering happens before you commit a day
of real output to a plan.

---

## Limitations

The reference paper's worst result is Gin rummy: 0.78 train and 0.75 test
transition accuracy after a 500-call synthesis budget, and a heavy loss to a
ground-truth opponent. Its diagnosis is that games with intricate multi-step
procedures resist synthesis. Attribution windows, reporting lag and
compounding cohort effects are that kind of procedure, so expect distribution
to land closer to Gin rummy than to tic-tac-toe. `fronts accuracy` reports
train, test and online accuracy separately. Read it before trusting a plan.

`docs/GAME.md` section 10 lists what the model is known to get wrong, and
section 9 rates how well each solution concept is justified. The signalling
module rests on an analogy that does not fully hold, and says so in its own
docstring.

One more caveat. `fronts synth` writes a state-inference sampler beside the
world model, `fronts plan` determinizes with it when it loads (and prints
`open-loop` when it does not), and `fronts accuracy` scores it. That score
certifies the samples are consistent with your observations, not that they
are drawn from the right distribution. A plan under the sampler is better
informed, not clairvoyant.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/GAME.md`](docs/GAME.md) | The formal game: players, hidden state, observation model, payoffs, solution concepts, known failures |
| [`docs/PAPER.md`](docs/PAPER.md) | What transfers from the Code World Models paper, what does not, and two errata in the original |
| [`docs/OPERATING.md`](docs/OPERATING.md) | The daily and weekly loop, the cold-start floor, and what to do when the model is bad |

---

## Licence

MIT.
