# Decision memo: per-step model routing and what it does to the leaderboard

**For:** advisor review
**Status:** decision required. This memo does not pick.
**Scope:** what routing changes about the leaderboard's meaning, and the two
options. The engineering question — does it work — is separate and is answered
by measurement, not by this memo.

---

## The mechanism, in one paragraph

A decision pipeline is N sequential LLM calls. Step 1 receives the market
snapshot; each later step receives every earlier step's JSON output. Today all
N calls go to the entrant's configured model. Routing sends the mechanical
steps — extraction, formatting — to a cheap model and keeps the expensive one
for the decision step. Cost per call spans **193x** across the seven measured
models ($0.000447 Nemotron to $0.086262 GPT-5.5), so on a 3-step pipeline
routing steps 1–2 to the cheap end removes most of the bill.

## What changes about what the leaderboard measures

The leaderboard exists to compare models. Its current claim is:

> *These twelve entrants ran the same strategy over the same window. The
> differences in return are differences between the models.*

If steps 1–2 run Nemotron for every entrant, that claim becomes:

> *These twelve entrants shared an extraction and signal-generation front end.
> The differences in return are differences between the models **at the
> decision step**.*

Both are defensible sentences. They are not the same sentence, and the second
one is not what the leaderboard currently advertises.

**The change is not a degradation — it is a change of variable.** Routing holds
steps 1..N-1 constant across entrants, which is what a controlled experiment
normally wants. Today those steps vary with the entrant, so a model that writes
a better intermediate summary scores better partly through that summary rather
than through its trading judgement. Whether that is signal or confound depends
on what the leaderboard is FOR:

| If the leaderboard is asking… | then routing… |
|---|---|
| "which model makes better trading decisions?" | **improves** it — it removes a confound |
| "which model runs this agent best, end to end?" | **invalidates** it — the pipeline is no longer the entrant's |

Both readings currently have adherents in how the board is described.

## The three things that are true regardless

1. **It is not a small edit to the meaning.** Four of seven models on the board
   are closed-weight and differ substantially in verbosity; the intermediate
   summaries they produce today are part of what is being scored.

2. **A partial rollout is the worst option.** Routing some entrants and not
   others makes the board incomparable in a way that is invisible in the
   results table. Whatever is chosen must apply to every entrant or to none.

3. **The failure mode is not "slightly worse".** `pipeline_runner` aborts the
   entire decision when any step returns unparseable JSON, with no retry. The
   caller then either raises or **silently falls back to rule-based trading**.
   So a cheap model that fails to parse does not produce a worse LLM decision —
   it produces a decision that is not an LLM decision at all, recorded on the
   board under the entrant's name. Any routing on the leaderboard needs the
   parse-reliability numbers first, and arguably needs the silent fallback
   changed to a hard failure before it is safe at all.

## The two options

### Option A — route everywhere, including the leaderboard

Every entrant shares the same cheap front end; only the decision step varies.

- Cost falls on the largest line item immediately.
- The comparison becomes cleaner in the "decision quality" reading: one
  variable instead of N.
- The board's public description must be rewritten. Historical entries become
  incomparable to new ones, so the current window either ends or is re-run.
- Requires the parse-reliability result first, and a decision about the
  rule-based fallback, because a routed abort silently contaminates a
  scoreboard.

### Option B — route only outside the leaderboard

The leaderboard keeps every step on the entrant's own model. Routing applies to
user agents, backtests, and internal runs.

- The board's existing claim and its history stay intact; nothing to rewrite.
- The saving is real but smaller — it misses whatever share of spend the
  leaderboard represents, and that share should be quantified before choosing.
- Two code paths to keep honest. The config already supports this (routing is
  per-agent and off by default), but "leaderboard entrants must not carry a
  routing config" becomes an invariant someone has to enforce and test.
- Leaves the confound in place: the board keeps scoring intermediate-summary
  quality as part of trading skill, without saying so.

## A finding from the first measured run that changes the ordering

The reliability harness has now been run ($0.0349, 90 calls — see
`step_reliability_first_run.md`). It surfaced something that outranks this
memo's question:

**A well-formed "make no trades this hour" — `{"orders": []}` — aborts the
decision**, because `pipeline_output_to_decision` requires a *non-empty* list.
The caller then either raises or silently falls back to rule-based trading.
Verified with no model involved. "Trade nothing" is the correct decision most
hours and the pipeline cannot express it.

On the leaderboard this is worse than an error, because the fallback is silent:
an entry keeps running under the model's name while its decisions are
rule-based. **That is a live contamination risk in the current board, with no
routing involved at all**, and it should be settled before the routing question
is worth deciding.

The same run showed both cheap models parsing steps 1–2 at 30/30 cleanly, so
the mechanical steps are not obviously the risk that motivated measuring first.

## What would settle it

Not an argument — three numbers, of which one now partly exists:

1. **Parse reliability per model per step.** If the cheap models are not
   near-perfect on the mechanical steps, both options collapse and the question
   is moot. First run done: steps 1–2 are 30/30 on both models tested; step 3
   is not yet measured against a realistic upstream context, so it remains open.
2. **The leaderboard's share of total spend.** If it is small, Option B costs
   little and the question is much less interesting.
3. **Whether the front-end summary actually differentiates the models.** If
   entrants produce near-identical step-1 output, routing removes a confound
   that was not doing anything, and Option A is nearly free of meaning-change.

## What is being asked

A choice between A and B, or a direction on which of the three numbers to get
first. The routing configuration and both measurement harnesses are built and
tested; nothing is enabled. **The default in code is current behaviour, and no
run changes unless a config is explicitly switched on.**
