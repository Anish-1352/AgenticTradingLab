# Escalate the output ceiling on the first retry, not the fifth

**Branch:** `fix/escalate-ceiling-on-first-retry` (from `origin/main` @ `4388762f`)
**Size:** ~50 lines in the retry loop, one env flag, 17 tests. Off by default.

## The problem

An empty reply — `content types: ['thinking']` — is what a reply looks like when
reasoning ran out the output ceiling before any text was emitted. Measured
across four runs: **every** such failure returned at exactly `max_tokens`, and
**none** returned below it.

Asking again at the ceiling that just failed asks for the same failure. The
single-prompt loop in `portfolio_manager` does that four times, then raises the
ceiling on the fifth attempt and usually succeeds.

`pipeline_runner` already disagrees with this. An empty pipeline response is
retried **once**, immediately at `RECOVERY_MAX_OUTPUT_TOKENS`. So the identical
failure costs one extra call on that path and five on this one. This PR makes
them agree.

## What changes

When `LLM_ESCALATE_CEILING_ON_RETRY=1`, retries after an empty reply use the
recovery ceiling instead of the one that just failed.

**The first request is untouched** — in call shape, not merely in resolved
value. `max_tokens` is passed only when it is actually being raised, so an
unescalated attempt is the identical call it is today. That is why the existing
`test_portfolio_final_no_text_retry_preserves_reasoning_and_increases_budget`
passes unmodified, and it is what keeps this a cost change rather than a
behaviour change.

Also sets `recovery_spent` when the escalated retry is what produced the text,
so the post-parse truncation retry does not re-issue an identical request —
which the existing comment says has "nothing different left to ask for".

Behind `LLM_ESCALATE_CEILING_ON_RETRY`, default off — nothing changes until
enabled. Happy to make it the default if preferred; the change only affects
requests that already failed.

## Measured: three interleaved pairs, 42 decisions a side

Paired because a single run cannot see this. The **same** configuration on the
same window produced 2.21, 2.43 and 3.36 attempts per decision — a single-run
A/B would be measuring the day.

| | off | on | delta |
|---|---:|---:|---:|
| attempts/decision | 2.67 | 1.88 | −29.5% |
| retries | 70 | 37 | −47.1% |
| **decisions needing ≥4 attempts** | **14 / 42** | **0 / 42** | Fisher *p* < 0.001 |
| input tokens | 224,255 | 150,130 | −33.1% |
| output tokens | 241,848 | 190,003 | −21.4% |
| wall seconds | 1531 | 1427 | −6.8% |
| cost | $0.0596 | $0.0455 | −23.6% |

Attempt distribution:

| attempts | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| off | 13 | 11 | 4 | 5 | 9 |
| on | 10 | 27 | 5 | **0** | **0** |

No decision exceeded three attempts with the flag on, across 42 decisions.

It also removes most of the run-to-run variance: the flag-off arm ranged
2.21–3.36 attempts per decision; the flag-on arm came out 1.93, 1.93, 1.79.

## Did it change the trading?

| arm | returns | mean |
|---|---|---:|
| off | −0.0092, −0.0053, −0.0010 | −0.0052 |
| on | −0.0037, −0.0032, +0.0017 | −0.0017 |

About one standard deviation on three runs a side, which **settles nothing**.
What matters is that it did not get worse. Raising the default output ceiling —
the other candidate for this failure — changes every request and made the same
window's return worse (−0.19% → −0.71%). This changes only requests that had
already failed, and returns moved slightly the other way.

That asymmetry is the argument for this lever over that one.

## Why the tail matters more than the cost

A decision that exhausts all five attempts raises `LLMDecisionError` and falls
back to the rule-based agent, with nothing persisted to say so. Those decisions
were 14 of 42 without this change and 0 with it. The cost saving is welcome;
removing the silent fallbacks is the point.

## Not in this PR

- Raising the default ceiling. Cheaper still, but it changes every request and
  degraded returns on the one window measured. It needs several windows before
  anyone should default it on.
- Persisting which attempt a call was — that is `feat/llm-call-usage`, and it
  is what would let this be verified from stored data rather than from a probe.
