# From Code World Models to distribution

How `blotto` applies **Code World Models for General Game Playing**
(Lehrach, Hennes, Lázaro-Gredilla et al., Google DeepMind, arXiv:2510.04542) to
organic content distribution — and, equally important, where the paper stops and
we are on our own.

---

## 1. What the paper actually does

The dominant way to use a language model in a sequential decision problem is as
a policy: show it the history, ask it for the next move. The paper's objection
is precise.

> "It relies on the model's implicit fragile pattern-matching capabilities,
> leading to frequent illegal moves and strategically shallow play."

Their alternative is to change the LLM's job. Instead of asking it to *play*,
they ask it to *write the game*: given natural-language rules and a handful of
observed trajectories, emit a Python simulator conforming to the OpenSpiel API.
Then run classical planners — MCTS for perfect information, Information Set MCTS
for imperfect — inside the synthesised model.

> "We shift the burden on the LLM from producing a good policy to producing a
> good world model, which in turn enables planning methods to turn compute into
> playing performance."

The results are worth stating with their caveats intact. Across ten games the
CWM agent beats Gemini 2.5 Pro used as a policy — but on Backgammon, Gemini
forfeited 100 out of 100 games, and on Gin rummy 99–100 out of 100. Much of the
headline margin is an opponent that cannot produce a legal move. The cleanest
win is Generalized tic-tac-toe: 0.89 and 0.93 win rates with **zero** forfeits
on either side. And the paper's own worst case is honest and instructive — Gin
rummy synthesis reaches 0.78 train and 0.75 test transition accuracy after
exhausting a 500-call budget, and the agent loses badly to a ground-truth
opponent.

---

## 2. Why this transfers

Distribution has the exact shape the method is built for, and one property that
makes the fit unusually good.

**You have rules in natural language.** Platform policies, FTC guidance, your
own brand constraints, the operating doctrine in your playbooks. Prose, not
code.

**You have trajectories.** Every post you have shipped, with the metrics that
came back. Action, then observation. Small in number — which is the regime the
paper targets, since it synthesises from five trajectories.

**You do not have a simulator.** Nobody publishes one. Building one by hand
means encoding beliefs about a ranking function you cannot inspect.

**And you cannot see the state.** This is the property that makes the fit good
rather than merely plausible, and it maps onto the paper's hardest setting.

---

## 3. Closed deck

The paper distinguishes two regimes:

- **Open deck** — offline trajectories reveal hidden state and other players'
  actions after the fact. The assumption prior CWM work makes.
- **Closed deck** — "the agent can only ever access its own observations and
  actions." The paper notes this "has not been addressed in prior CWM work."

Organic distribution is closed deck, unambiguously. You see your analytics. You
do not see the ranking function's current weights. You do not see which angles
your competitors tested and killed last week. You do not even see all of your
own results — 25–40% of conversion events are blocked client-side before they
are recorded.

The paper's closed-deck solution is elegant and it is what `cwm/` implements. It
drops every unit test requiring hidden state and keeps only
observation → latent → observation reconstruction plus no-crash tests, producing

> "a kind of autoencoder, where the inference function acts as an encoder … and
> the CWM acts as a decoder … Instead of a bottleneck, or a regularization term,
> the game rules and the required OpenSpiel API (used in the unit tests)
> introduced in the context of the LLM act as regularizers to prevent trivial
> latent spaces from being discovered."

This is why `protocols.py` is narrow and why `State` is `dict[str, Any]` rather
than a schema of ours. The API *is* the regulariser. Constrain the latent space
and you remove the freedom that makes the method work — the paper's Hand of war
result, where the closed-deck agent outperformed the open-deck one, is
attributed to exactly "the freedom to synthesize simpler state spaces."

---

## 4. The mapping

| Paper | Here |
|---|---|
| Game rules in natural language | Platform policy, FTC guidance, brand constraints, operator doctrine |
| Observed trajectories | Your posted content and the metrics that came back |
| Hidden state | Ranking weights `θ`, account standing, angle saturation, audience belief |
| Chance player | Audience arrival, virality lottery, attribution noise |
| Illegal move → forfeit | Unsubstantiated claim, undisclosed AI face, premature scale |
| Transition accuracy | Does the model predict the reach and conversions you actually got |
| Inference accuracy | Does the inferred hidden state reproduce your observations |
| Value function tournament | Which state-value heuristic to trust with no ground truth |
| Host-CWM arena | Pick a content strategy before spending a day of output on it |
| Open deck | Not available. You never get to see behind the curtain |
| Closed deck | The only regime that exists here |

---

## 5. What a unit test is, here

In the paper, a test is a recorded `(state, action) → state'` transition. Ours
is a recorded `(context, published content) → observed metrics`, drawn from real
posting history, and it comes with a complication the paper does not have:
**our observations settle late.**

A metric read inside the 24–72 hour reporting window is provisional. Testing a
synthesised model against provisional numbers teaches it to predict noise. So
`Observation.is_partial` is part of the type, and partial observations are
excluded from test generation — the same discipline the legality engine applies
when refusing to open an evidence gate on unsettled data.

The refinement loop follows the paper's tree search: several candidate models
held simultaneously, Thompson sampling choosing which to refine, with a Beta
prior of `α = 1 + C·h` and `β = 1 + (1−h)·C` at `C = 5.0`, where `h` is the
average unit-test pass rate. Failed tests come back as stack traces in context.

---

## 6. Inference as code

ISMCTS needs to sample from the belief over hidden states. Exact inference is
exponential, so the paper has the LLM synthesise an approximate sampler instead.

The guarantee this buys is narrower than it first appears and stronger than it
looks. Replaying a sampled history through the CWM must reproduce every
observation actually seen, which places the sample inside the posterior's
support without placing it at the right density. The paper's argument for why
that suffices carries over intact:

> "Although this does not guarantee that s̃_t is correctly distributed, the
> correct support is already very informative, given the extremely sparse
> support of state posteriors in games."

Here that means: sample a `θ` and a saturation map consistent with the results
you actually observed. You will not get the right distribution over ranking
weights. You will get weights that are not *contradicted* by your data, which is
enough to plan against and is strictly more than an operator reasoning by feel
has available.

Because we are closed deck, `resample_state` is the primary path. The paper is
candid that it is the weaker of the two — it "cannot guarantee that the produced
sample s̃_t belongs to the support of the posterior, nor that it constitutes a
valid CWM hidden state, because it ignores the dependency between consecutive
states." We use it anyway, because the alternative does not exist.

---

## 7. The operational payoff

Two mechanisms in the paper are worth more here than they are in board games.

**Legality by construction.** In Backgammon an illegal move costs you a game. In
distribution it costs you an FTC action, a platform ban, or a budget committed
on five conversions of evidence. `get_legal_actions` enumerating only permitted
moves means the planner *cannot* select those — not "is penalised for selecting
them", cannot. That is a stronger safety property than any prompt.

**Deciding before spending.** The paper synthesises five candidate models, has
their agents play tournaments against each other using one agent's own model as
the host in place of unavailable ground truth, and rejects any agent "worse than
the best scoring agent by more than 10% of the observed utility range."

Transposed: synthesise several world models from your history, have the
resulting content strategies play each other inside those models, and discard
the ones that lose — **before** committing a day of real output. That is a
selection procedure operating on strategies rather than on posts, and it costs
compute rather than reach.

---

## 8. What the paper does not give us

Being clear about this matters more than the mapping.

**No game theory.** The paper uses extensive-form games as scaffolding for
planning. There is no mechanism design, no equilibrium computation, no auction,
no signalling. It cites equilibrium work only as related work. Every solution
concept in `solvers/` beyond ISMCTS — Blotto, EXP3, congestion, signalling,
Stackelberg — is ours, and `docs/GAME.md` §9 rates each for how well it is
actually justified rather than presenting them as uniformly load-bearing.

**No online learning.** The model is synthesised once, offline. The authors flag
this as future work: *"we hope to extend our method to enable active and online
learning of the world model."* Distribution is non-stationary on a timescale of
days, so this limitation bites harder here than in a board game whose rules do
not move. Re-synthesis cadence is a real open question in this repository too.

**No guarantee it works on procedurally complex domains.** The Gin rummy
failure is diagnosed as "multi-stage scoring … difficult for the LLM to capture
perfectly in code from a small number of trajectories," and named as "a key
frontier for CWM synthesis: mastering games with intricate, multi-step
procedural subroutines." Attribution windows, reporting lag, and compounding
cohort effects are exactly that kind of subroutine. **We should expect
distribution to be closer to Gin rummy than to tic-tac-toe**, and the honest
consequence is that transition and inference accuracy are reported for train,
test and online separately, so a bad model shows up as a bad model rather than
as inexplicably poor results.

---

## 9. Two things to know if you cite the numbers

Reading the paper closely turns up two internal inconsistencies worth flagging:

1. The §5.1.2 prose gives Gin rummy accuracy as 84% train / 79% test. Table 1
   gives 0.7816 / 0.7455. The prose figures match neither Table 1 nor the
   hidden-state inference numbers in Table 6 (1.0000 / 0.9513); where they
   came from is not identifiable from the paper.
2. §5.2.1 claims CWM-MCTS and ground-truth MCTS are at parity, "without either
   of them clearly winning in any of the games." Table 7's Backgammon row shows
   0.08 win / 0.92 loss and 0.07 / 0.93. That is not parity.

Neither undermines the method. Both are the sort of thing worth knowing before
quoting a figure in a README.

---

## Reference

Lehrach, W., Hennes, D., Lázaro-Gredilla, M., Lou, X., Wendelken, C., Li, Z.,
Dedieu, A., Grau-Moya, J., Lanctot, M., Iscen, A., Schultz, J., Chiam, M.,
Gemp, I., Zielinski, P., Singh, S., Murphy, K. P. (2025).
*Code World Models for General Game Playing.* arXiv:2510.04542.
