# The distribution game

A formal specification of the environment `fronts` plans inside.

This document defines a game. It does not claim the game is true. It claims
the game is less wrong than the alternative, which is to treat distribution as
a supervised learning problem with a content-shaped input and an
engagement-shaped label. That framing has a specific failure mode, described
below.

---

## 1. Why a game

Three properties of organic distribution break the supervised framing, and
more data does not fix any of them.

The reward is not the label. Engagement is observable, immediate and abundant.
Paying users are partly unobservable, delayed by days, and rare. A system
trained on the abundant signal learns to produce the abundant signal. The DTC
corpus records a concrete case: the cheapest acquisition vector
(Aspiration/Status at $30 CAC) produced 45% three-month churn and a $250 LTV,
while the most expensive (Exhaustion/Relief at $65 CAC) produced 8% churn and
a $1,200 LTV. An objective that minimises CAC picks the wrong one with high
confidence, and gets more confident with more data.

The environment contains other optimisers. An angle that works stops working,
and usually not because it decayed on its own. The platform reweighted, or
eleven other people started running it. Both corpora treat this as the central
dynamic. Ads "fatigue because the visual signature has saturated the specific
micro-cluster of audience that Facebook mapped it to." A polarising creative
drops CPM by up to 60% because rival loyalists argue in the comments and
subsidise the reach. These are strategic interactions. A model without other
agents in it can only watch the numbers go down and call it drift.

You cannot see the state. The ranking function is not published. Competitor
test schedules are not published. Even your own results are incomplete: ad
blockers and privacy controls remove 25-40% of events before they are
recorded, 20-40% of what remains would have converted anyway, and in B2B,
60-80% of the sharing that drives everything happens in DMs and Slack and
shows up, if at all, as direct traffic.

A model with hidden state, multiple optimising agents, and a reward that
differs from the observation is a game with imperfect information. Any other
name for it discards the three properties that matter most.

---

## 2. The formal object

    G = < N, H, Z, tau, A, rho, u, I >

a finite-horizon extensive-form game with imperfect information and chance.

| Symbol | Meaning |
|---|---|
| `N` | Players: operator, platform, field, plus chance |
| `H` | Histories; `Z` (a subset of `H`) the terminal ones, reached at horizon `T` |
| `tau(h)` | The player to act at history `h` |
| `A(h)` | Actions legal at `h`. Section 7 explains why this set does the real work |
| `rho` | Chance distribution at chance nodes |
| `u_i(z)` | Player `i`'s payoff at terminal `z` |
| `I_i` | Player `i`'s information partition: histories `i` cannot tell apart |

The operator's information partition is coarse and stays coarse. That is the
whole difficulty. Section 6 quantifies it.

---

## 3. Players

### P0: Operator

You, and the only player whose policy we choose. The objective is cumulative
contribution margin from paying users over the horizon, defined in section 8.

### P1: Platform

The ranking algorithm. It is not your adversary. It is misaligned with you,
which is a different and more tractable problem: it maximises session time and
retention while you maximise paying users. The objectives overlap, since
engaging content serves both, and they diverge sharply at the point where you
ask someone to leave the feed. That divergence is why external links carry a
ranking penalty, and why the penalty is a structural feature of the game
rather than a bug to route around.

Two properties make P1 unusual as a player.

It commits first. The ranking rule is fixed before you choose today's content,
and it does not adapt to you specifically. This makes P1 a Stackelberg leader
and you a follower, so your optimal play is a best response to the commitment
rather than an equilibrium negotiated with it.

The commitment is unobserved and drifts. You never see `theta`. You infer it,
badly, from your own results. `solvers/stackelberg.py` computes the follower
best response, and its `value_of_information` measures what the ignorance
costs.

### P2: Field

The aggregate of every other creator competing for the same attention.
Modelled as a population playing a congestion strategy over the angle space
rather than as a named opponent, because you do not know who they are and it
does not matter. What matters is occupancy: how much of the field currently
runs the angle you are considering.

The field is why the benchmark tables in `game/priors.py` must not be read as
a ranking. The Anti-Hero/Enemy Rant archetype posts the highest measured hook
rate in the corpus, 40-48%. That number was measured when few people were
running the archetype. It describes an uncrowded angle, and treating it as a
constant is the most expensive mistake available in this domain.

### Chance

Audience arrival, the virality lottery, and attribution noise. Following the
reference paper, all other transitions are deterministic and randomness enters
only here. The constraint is practical rather than stylistic: a model with
hidden nondeterminism cannot be unit-tested against a recorded trajectory, and
unit-testing against recorded trajectories is the mechanism by which the world
model gets built.

---

## 4. Hidden state

Never observed by P0. Not observed with noise; not observed at all.

| Component | Meaning |
|---|---|
| `theta` | Platform ranking weights over hook rate, hold rate, saves, shares, comments, dwell, follow-through, external-link penalty. Drifts. |
| `q` | Account standing in [0,1]. Degrades on compliance strikes. |
| `sigma[a]` | Field occupancy of angle `a`. Drives congestion. |
| `beta[v]` | Per-avatar audience belief: exposure count and credence in your central claim. |
| `f` | Per-asset fatigue clock. |

`beta` deserves a note. Attention within a cohort is not renewable on the
timescale of a campaign. The corpus puts it plainly: "You just end up showing
the exact same ad to the exact same guy for the 40th time. He didn't buy the
first 39 times." A model whose audience resets each step will happily
recommend showing him the 41st.

---

## 5. Action space

An operator move is one of four things, and keeping them separate is the
modelling decision this repository exists to make.

`Publish` ships one item into one slot. It is structured rather than free
text:

    (platform, format, archetype, vector, semantic_tier, hook,
     avatar, cta_mode, claim_class, angle, utm_content)

Each dimension is an enumerated taxonomy drawn from the source corpora, and
several carry measured priors. See `game/priors.py`, where every number has a
`source` field and a prior without one is not permitted to exist.

`utm_content` is not metadata. It is the join key between a move and the
observation that arrives 24-72 hours later. A `Publish` without one is illegal
(section 7): an untracked post generates no learning signal, so it is strictly
dominated by the same post with tracking.

`Scale` commits more budget or volume behind an angle already in flight.

`Kill` retires an angle.

`Hold` spends a slot on nothing. It is legal and sometimes correct: an
operator with no uncrowded angle and no substantiated claim does better silent
than shipping generic slop at $3.50 CPC and 0.8% CVR. A planner that cannot
represent inaction will always find a reason to post.

The structural point is that `Publish` and `Scale` are different kinds of
decision. Publishing is cheap and reversible. Scaling commits real money
behind a belief. Collapsing the two into "produce content" is what lets a
system quietly bet the budget on five conversions' worth of evidence.

---

## 6. Observation model

What P0 actually receives. The degradation is stated numerically because every
figure below is measured in the source corpora rather than assumed.

| Property | Value | Source |
|---|---|---|
| Client-side event loss | 25-40% of events blocked | GTM 09 |
| Server-side recovery | returns 20-30% of the lost | GTM 09 |
| Non-incremental share | 20-40% of attributed conversions would have happened anyway | GTM 09 |
| Dark social (B2B) | 60-80% of content sharing is untrackable | GTM 03 |
| Reporting lag | 24-72 hours before figures settle | DTC 21 |

Two consequences are enforced in code.

Coverage and incrementality pull in opposite directions. True conversions are
`attributed / coverage * incrementality`. Dividing by coverage grosses up for
what you could not see; multiplying by incrementality discounts what you would
have got anyway. Applying only one correction, which is the common practice in
both directions, produces a biased estimate that looks like a careful one.
`game/payoff.py` applies both.

Coverage below the floor makes an angle unjudgeable rather than noisy. Below
the floor, `Scale` and `Kill` are both illegal. You may not conclude an angle
is winning, and you may not conclude it is losing, because you cannot see most
of what it did. Operators reliably get this rule wrong in the confident
direction.

---

## 7. Legality is structural, not advisory

`A(h)`, the legal action set, is where this design earns its keep.

The reference paper's central practical claim is that a synthesised world
model serves as a formal specification, "allowing planners to algorithmically
enumerate valid actions and avoid illegal moves." In board games that prevents
forfeits. Here the illegal moves are:

- an unsubstantiated outcome claim (FTC: every claim must be provable);
- an AI-generated testimonial presented as a human one;
- undisclosed AI-generated faces on TikTok;
- before/after imagery on Meta;
- a budget increase above +20% in a rolling 48 hours, which resets the
  learning phase;
- a `Scale` move on an angle that has not cleared its evidence gate.

The last one is the point. Both corpora arrive independently at the conclusion
that the dominant failure in distribution is premature scaling rather than bad
creative, and both state it numerically. The DTC side: never declare a winner
under 50 conversions in a trailing 7-day window; call it at five and CPA "will
frequently explode to $150" when you scale from $100/day to $1,000/day. The
GTM side, for spend commitments: 300+ conversions per variant, minimum 14
days, one variable, no peeking before the minimum sample.

Both thresholds are implemented, because they license different decisions.
50/7d permits a creative judgement. 300/14d permits a budget commitment.

The distinction that matters is between a penalty and a gate. A penalty is a
term in the objective, and a sufficiently confident planner will pay it. A
gate removes the move from `A(h)` entirely, so the planner cannot select it at
any confidence. Observations still inside the reporting lag contribute zero
toward a gate, which is what makes peeking structurally impossible instead of
merely discouraged.

---

## 8. Payoff

    u_0 = sum over t of [ true_conversions(o_t) * contribution_per_user - cost_t ]

where

    true_conversions = attributed / max(coverage, eps) * incrementality
    contribution_per_user = LTV * gross_margin
    LTV = ARPU * gross_margin * (1 / monthly_churn)

Health bands, from GTM 02: LTV:CAC below 1.0, stop; 1.0-2.0, marginal; 3.0,
golden; above 5.0 you are underspending on growth.

Reach does not appear in this function. Neither do impressions, likes, follows
or saves. They appear in the observation, because they are evidence about
hidden state: a hook rate below 25% tells you the platform has stopped serving
the asset, which is worth knowing. They do not appear in the reward, because
they are not the thing being bought. The omission is deliberate, and
`payoff.py` says so in a comment so that no future reader mistakes it for an
oversight and adds an engagement term.

---

## 9. Solution concepts, and how well each is justified

Different sub-problems take different tools. Saying which is which is the
difference between applied game theory and game-theoretic vocabulary.

| Sub-problem | Concept | Module | Standing |
|---|---|---|---|
| Sequential play under hidden state | Information Set MCTS | `solvers/ismcts.py` | Sound. Cowling et al. 2012, well understood. |
| Daily allocation across contested fronts | Colonel Blotto mixed equilibrium | `solvers/blotto.py` | Sound for the symmetric continuous case (Roberson 2006). Our discretisation to integer units is an approximation. |
| Angle choice under drift | EXP3 adversarial bandit | `solvers/exp3.py` | Sound. Auer et al. 2002. Regret bounds hold against an adaptive adversary, which is why not UCB. |
| Angle crowding | Congestion game, Rosenthal potential | `solvers/congestion.py` | Sound as a model; pure Nash exists and best-response dynamics converge. Whether real creators play the equilibrium is untested. |
| Claim credibility | Spence separating equilibrium | `solvers/signalling.py` | An analogy. Load-bearing assumptions stated in-module. |
| Responding to the ranking rule | Stackelberg follower best response | `solvers/stackelberg.py` | Sound structurally. Depends entirely on the quality of the theta estimate. |

The signalling module is the one to be sceptical of. Spence's model assumes a
Bayesian receiver and an imitation cost that is common knowledge, and neither
holds cleanly for an audience scrolling a feed. It is included because it
makes a useful and otherwise unavailable prediction: that "radical
transparency" works because it is expensive to fake, not because audiences
prefer honesty. That prediction is falsifiable, which is the standard it is
held to. It is not included because it is proven.

---

## 10. What this model gets wrong

An unfalsifiable model is not a model. Here is where this one is known to be
weak.

The field is a population, not agents. Real competitors observe you and
respond. Congestion modelling captures crowding, but not deliberate
counter-positioning, and it cannot represent a rival who moves because you
moved.

Theta drift is a random walk. Real ranking changes are discrete, correlated
across accounts, and occasionally enormous. A random walk understates tail
risk in the situations where tail risk matters most.

The horizon is finite and short. Brand equity, reputation and audience trust
compound over years, and a 30-day horizon systematically undervalues them,
which biases the planner toward extraction. A repeated-game correction, where
the folk theorem makes honest play sustainable at a high enough discount
factor, is the obvious next module. It is not built.

Priors are point-in-time. Benchmarks from 2026 corpora will age. Every number
in `priors.py` carries a `source` so its provenance and staleness stay visible
instead of dissolving into the code.

Synthesis can fail. The reference paper's worst case is Gin rummy: 0.78 train,
0.75 test transition accuracy, 500 LLM calls, budget exhausted. Its diagnosis,
that games with "intricate, multi-step procedural subroutines" resist
synthesis, applies directly to a domain with attribution windows, lag, and
compounding cohort effects. `cwm/` reports transition and inference accuracy
for train, test and online separately, so a bad model is visible as a bad
model rather than as mysteriously poor results.

---

## References

- Lehrach, Hennes, Lázaro-Gredilla et al. (2025). *Code World Models for
  General Game Playing.* arXiv:2510.04542. The architecture this repository
  applies.
- Cowling, Powley, Whitehouse (2012). *Information Set Monte Carlo Tree Search.*
- Roberson (2006). *The Colonel Blotto Game.* Economic Theory 29(1).
- Auer, Cesa-Bianchi, Freund, Schapire (2002). *The Nonstochastic Multiarmed
  Bandit Problem.* SIAM J. Comput. 32(1).
- Rosenthal (1973). *A Class of Games Possessing Pure-Strategy Nash Equilibria.*
- Spence (1973). *Job Market Signaling.* QJE 87(3).
- Kuhn (1953). *Extensive Games and the Problem of Information.*
