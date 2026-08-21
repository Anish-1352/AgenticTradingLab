# Decision memo: should routing apply to the leaderboard?

**For:** advisor review
**Status:** decision required. This memo does not pick.
**Supersedes:** the earlier memo on `feature/per-step-model-routing`, written
before the decision step was measured.

---

## What changed since the last memo

Three things, and two of them weaken the case for routing.

**Routing's end-to-end saving is `35.6%`, not `76.2%`** [DERIVED]. The earlier
figure was the saving on the mechanical steps alone and was labelled as such.
End to end it roughly halves, because those steps are only `49.4%` of pipeline
cost — and it falls to `15.5%` if the decision step keeps a mid-tier model
rather than a cheap one [DERIVED]. Routing is strongest precisely when the
thing it protects is cheap.

**The cheap models fail where judgement is hardest.** Across four market
regimes, every decision-step failure for the two cheaper models landed in the
flat, directionless regime — `1/4` and `2/4` there against `4/4` in every
other regime [MEASURED]. The mid-tier model handled the same regime `4/4`
[MEASURED]. Failures that cluster on flat days are worse than the pooled rate
suggests: on a directionless day an entrant could lose most of its decisions
rather than a fifth of them.

**The empty-decision result reversed.** The earlier round measured
`{"orders": []}` in every attempt [MEASURED] and it looked like a serious
defect exposure. Against real upstream
context the empty-decision rate is `0` [MEASURED]. The earlier result was an
artifact of a placeholder fixture. The underlying defect at
`portfolio_manager.py:432-446` is still real — a well-formed empty decision is
still silently replaced by a rule-based one — but this probe found no live
instance of it firing.

## What routing would change about the board's meaning

The leaderboard exists to compare models. Its current claim is:

> These entrants ran the same strategy over the same window. The differences
> are differences between the models.

With the mechanical steps on a shared cheap model, it becomes:

> These entrants shared an extraction and signal front end. The differences
> are differences between the models **at the decision step**.

Both are defensible. They are not the same sentence, and the second is not what
the board currently advertises.

### Reading A — routing makes it better science

Today each entrant's intermediate summaries are produced by its own model, so a
model that writes a better summary scores better partly through that summary
rather than through its trading judgement. Routing holds the mechanical steps
constant and isolates the variable the board claims to measure. That is what a controlled
experiment normally wants.

### Reading B — routing invalidates the comparison

The pipeline is part of what an entrant *is*. A model that reads noisy
extraction well is a better trading agent, and controlling that away measures
something narrower than "which model runs this agent best". The board would be
comparing decision-step behaviour and reporting it as end-to-end performance.

Which reading is right depends on what the board is for, and that has not been
written down anywhere this project could find.

## The cost of each option

| option | saving on the board's spend | what it costs | Tier |
|---|---:|---|---|
| A — route everywhere including the board | up to `35.6%` of multi-step pipeline spend | the board's public claim must be rewritten; historical entries become incomparable | DERIVED |
| B — route everywhere except the board | `0%` of board spend | two code paths; the confound stays and stays unstated | DERIVED |

Both numbers are smaller than they look, for a reason that applies to either
option: **the board's entrants are single-prompt agents.** The seed runs
measured `1.000` calls per decision [MEASURED] — one call, no mechanical steps,
nothing to route. Routing would have saved the measured leaderboard runs
exactly `$0` [DERIVED]. It becomes relevant only if the board moves to
multi-step pipelines.

That reframes the decision considerably. This is not "should we trade the
board's meaning for a saving". It is "if the board ever adopts
multi-step pipelines, should routing apply to it" — a question that can be
answered before the situation arises, and cheaply.

## The prior question: the board cannot currently attribute its own decisions

Independent of routing, earlier work established that the platform cannot tell
whether a leaderboard decision came from the model or from the rule-based
fallback [MEASURED]. `PortfolioManager.llm_decisions` counts model-driven steps
correctly, `llm_agent.py` copies it onto the strategy object, the publish
guard reads it — and then it is dropped. `insert_run` has no such parameter, so
the number existed in memory during every run and was never written.

`llm_calls` cannot substitute: the call is billed before the response is
parsed, so a fallback consumes its call too. The observed `1.000` calls/bar is
consistent with every decision being the model's and with none of them being
[MEASURED].

**So the board's attribution is already weaker than it appears.** Routing would
add a second thing the board cannot see, on top of one it already cannot. The
ordering that follows is not a recommendation about routing — it is an
observation that one of these is cheap and unambiguous and the other is neither:

```
ALTER TABLE agent_runs ADD COLUMN llm_decisions INTEGER;
```

one column and one argument through `insert_run`, and the question "was this
entry actually the model" becomes a `SELECT`.

## What is being asked

A choice between A and B — or, more usefully, a decision on whether the board
is heading for multi-step pipelines at all, since that is what makes the
question live. Nothing is enabled: the routing config defaults to current
behaviour and no run changes unless a config is explicitly switched on
[MEASURED].

## What this memo does not know

- **Whether routing changes which decisions get made** [NOT MEASURED]. This
  round measured that the decision step still parses. It says nothing about
  whether the orders are the same or as good — that needs multiple evaluation
  windows and a definition of "same".
- **Whether routing changes the decision step's reliability** [NOT MEASURED].
  Every decision-step attempt read upstream produced by the cheap model, so
  there is no measurement of the unrouted case to compare against.
- **What fraction of platform spend runs through multi-step pipelines**
  [NOT MEASURED]. No telemetry records it, which is why no saving is quoted
  against a whole bill anywhere in this work.
