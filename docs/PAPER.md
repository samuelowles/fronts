# From Code World Models to distribution

How `fronts` applies *Code World Models for General Game Playing* (Lehrach,
Hennes, Lázaro-Gredilla et al., Google DeepMind, arXiv:2510.04542) to organic
content distribution, and where the paper stops and this repository is on its
own.

---

## 1. What the paper actually does

The dominant way to use a language model in a sequential decision problem is
as a policy: show it the history, ask it for the next move. The paper's
objection is precise.

> "It relies on the model's implicit fragile pattern-matching capabilities,
> leading to frequent illegal moves and strategically shallow play."

Their alternative changes the LLM's job. Instead of asking it to play, they
ask it to write the game: given natural-language rules and a handful of
observed trajectories, emit a Python simulator conforming to the OpenSpiel
API. Classical planners then run inside the synthesised model: MCTS for
perfect information, Information Set MCTS for imperfect.

> "We shift the burden on the LLM from producing a good policy to producing a
> good world model, which in turn enables planning methods to turn compute
> into playing performance."

The results deserve their caveats. Across ten games the CWM agent beats Gemini
2.5 Pro used as a policy, but on Backgammon, Gemini forfeited 100 out of 100
games, and on Gin rummy 99-100 out of 100, so much of the headline margin is
an opponent that cannot produce a legal move. The cleanest win is Generalized
tic-tac-toe: 0.89 and 0.93 win rates with zero forfeits on either side. The
paper's worst case is instructive too. Gin rummy synthesis reaches 0.78 train
and 0.75 test transition accuracy after exhausting a 500-call budget, and the
agent loses badly to a ground-truth opponent.

---

## 2. Why this transfers

Distribution has the shape the method is built for.

You have rules in natural language: platform policies, FTC guidance, your own
brand constraints, the operating doctrine in your playbooks. Prose, not code.

You have trajectories: every post you have shipped, with the metrics that came
back. Action, then observation. Few in number, which is the regime the paper
targets, since it synthesises from five trajectories per game.

You do not have a simulator. Nobody publishes one, and building one by hand
means encoding beliefs about a ranking function you cannot inspect.

And you cannot see the state. This last property is what makes the fit close
rather than merely plausible, because it maps onto the paper's hardest
setting.

---

## 3. Closed deck

The paper distinguishes two regimes:

- Open deck: offline trajectories reveal hidden state and other players'
  actions after the fact. Prior CWM work assumes this.
- Closed deck: "the agent can only ever access its own observations and
  actions." The paper notes this "has not been addressed in prior CWM work."

Organic distribution is closed deck, unambiguously. You see your analytics.
You do not see the ranking function's current weights. You do not see which
angles your competitors tested and killed last week. You do not even see all
of your own results, since 25-40% of conversion events are blocked client-side
before they are recorded.

The paper's closed-deck solution is what `cwm/` implements. It drops every
unit test requiring hidden state and keeps observation-to-latent-to-observation
reconstruction plus no-crash tests, producing

> "a kind of autoencoder, where the inference function acts as an encoder ...
> and the CWM acts as a decoder ... Instead of a bottleneck, or a
> regularization term, the game rules and the required OpenSpiel API (used in
> the unit tests) introduced in the context of the LLM act as regularizers to
> prevent trivial latent spaces from being discovered."

This is why `protocols.py` is narrow and why `State` is `dict[str, Any]`
rather than a schema of ours. The API is the regulariser. Constrain the latent
space further and you remove the freedom that makes the method work. The
paper's Hand of War result, where the closed-deck agent outperformed the
open-deck one, is attributed to "the freedom to synthesize simpler state
spaces."

---

## 4. The mapping

| Paper | Here |
|---|---|
| Game rules in natural language | Platform policy, FTC guidance, brand constraints, operator doctrine |
| Observed trajectories | Your posted content and the metrics that came back |
| Hidden state | Ranking weights theta, account standing, angle saturation, audience belief |
| Chance player | Audience arrival, virality lottery, attribution noise |
| Illegal move leads to forfeit | Unsubstantiated claim, undisclosed AI face, premature scale |
| Transition accuracy | Does the model predict the reach and conversions you actually got |
| Inference accuracy | Does the inferred hidden state reproduce your observations |
| Value function tournament | Which state-value heuristic to trust with no ground truth |
| Host-CWM arena | Pick a content strategy before spending a day of output on it |
| Open deck | Not available. You never see behind the curtain |
| Closed deck | The only regime that exists here |

---

## 5. What a unit test is, here

In the paper, a test is a recorded `(state, action) -> state'` transition.
Ours is a recorded `(context, published content) -> observed metrics`, drawn
from real posting history, and it carries a complication the paper does not
have: our observations settle late.

A metric read inside the 24-72 hour reporting window is provisional. Testing a
synthesised model against provisional numbers teaches it to predict noise. So
`Observation.is_partial` is part of the type, and partial observations are
excluded from test generation, the same discipline the legality engine applies
when it refuses to open an evidence gate on unsettled data.

The refinement loop follows the paper's tree search: several candidate models
held simultaneously, Thompson sampling choosing which to refine, with a Beta
prior of `alpha = 1 + C*h` and `beta = 1 + (1-h)*C` at `C = 5.0`, where `h` is
the average unit-test pass rate. Failed tests come back as stack traces in the
next prompt's context.

---

## 6. Inference as code

ISMCTS needs to sample from the belief over hidden states. Exact inference is
exponential, so the paper has the LLM synthesise an approximate sampler
instead.

The guarantee this buys is narrow but real. Replaying a sampled history
through the CWM must reproduce every observation actually seen, which places
the sample inside the posterior's support without placing it at the right
density. The paper's argument for why support membership is enough carries
over:

> "Although this does not guarantee that s_t is correctly distributed, the
> correct support is already very informative, given the extremely sparse
> support of state posteriors in games."

Here that means: sample a theta and a saturation map consistent with the
results you observed. You will not get the right distribution over ranking
weights. You will get weights that are not contradicted by your data, which is
enough to plan against, and more than an operator reasoning by feel has.

Because we are closed deck, `resample_state` is the primary path. The paper is
candid that it is the weaker of the two variants: it "cannot guarantee that
the produced sample s_t belongs to the support of the posterior, nor that it
constitutes a valid CWM hidden state, because it ignores the dependency
between consecutive states." We use it anyway, because the alternative does
not exist in this regime.

---

## 7. The operational payoff

Two mechanisms in the paper matter more here than they do in board games.

Legality by construction. In Backgammon an illegal move costs you a game. In
distribution it costs you an FTC action, a platform ban, or budget committed
on five conversions of evidence. With `get_legal_actions` enumerating only
permitted moves, the planner cannot select an illegal one. That is a stronger
property than penalising illegal moves, and stronger than anything a prompt
provides.

Deciding before spending. The paper synthesises five candidate models, has
their agents play tournaments against each other using one agent's own model
as the host in place of unavailable ground truth, and rejects any agent "worse
than the best scoring agent by more than 10% of the observed utility range."

Transposed: synthesise several world models from your history, have the
resulting content strategies play each other inside those models, and discard
the ones that lose, before committing a day of real output. That is selection
operating on strategies rather than on posts, and it costs compute rather than
reach.

---

## 8. What the paper does not give us

No game theory. The paper uses extensive-form games as scaffolding for
planning. There is no mechanism design, no equilibrium computation, no
auction, no signalling; it cites equilibrium work only as related work. Every
solution concept in `solvers/` beyond ISMCTS (Blotto, EXP3, congestion,
signalling, Stackelberg) is this repository's addition, and `docs/GAME.md`
section 9 rates each one for how well it is justified instead of presenting
them as uniformly load-bearing.

No online learning. The model is synthesised once, offline. The authors flag
this as future work: "we hope to extend our method to enable active and online
learning of the world model." Distribution is non-stationary on a timescale of
days, so the limitation bites harder here than in a board game whose rules do
not move. Re-synthesis cadence is an open question in this repository too.

No guarantee on procedurally complex domains. The Gin rummy failure is
diagnosed as "multi-stage scoring ... difficult for the LLM to capture
perfectly in code from a small number of trajectories," and named as "a key
frontier for CWM synthesis: mastering games with intricate, multi-step
procedural subroutines." Attribution windows, reporting lag, and compounding
cohort effects are that kind of subroutine. Expect distribution to land closer
to Gin rummy than to tic-tac-toe. The practical consequence: transition and
inference accuracy are reported for train, test and online separately, so a
bad model shows up as a bad model rather than as inexplicably poor results.

---

## 9. Two things to know if you cite the numbers

Reading the paper closely turns up two internal inconsistencies worth
flagging.

1. The section 5.1.2 prose gives Gin rummy accuracy as 84% train / 79% test.
   Table 1 gives 0.7816 / 0.7455. The prose figures match neither Table 1 nor
   the hidden-state inference numbers in Table 6 (1.0000 / 0.9513); where they
   came from is not identifiable from the paper.
2. Section 5.2.1 claims CWM-MCTS and ground-truth MCTS are at parity, "without
   either of them clearly winning in any of the games." Table 7's Backgammon
   row shows 0.08 win / 0.92 loss and 0.07 / 0.93. That is not parity.

Neither undermines the method. Both are worth knowing before quoting a figure
in a README.

---

## Reference

Lehrach, W., Hennes, D., Lázaro-Gredilla, M., Lou, X., Wendelken, C., Li, Z.,
Dedieu, A., Grau-Moya, J., Lanctot, M., Iscen, A., Schultz, J., Chiam, M.,
Gemp, I., Zielinski, P., Singh, S., Murphy, K. P. (2025).
*Code World Models for General Game Playing.* arXiv:2510.04542.
