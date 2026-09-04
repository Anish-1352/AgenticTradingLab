# Fit the thinking budget under `max_tokens` (opt-in)

**Branch:** `fix/reasoning-budget-fits-output-cap` (from `origin/main` @ `4388762f`)
**Status:** measured, and it **does not fix the thing it targets**. Read the results section before merging.

## The defect

The shipped defaults ask for more thinking than the reply is allowed to be:

| setting | value |
|---|---|
| `reasoning.max_tokens` (medium effort) | **2048** |
| `max_tokens` (`DEFAULT_MAX_OUTPUT_TOKENS`) | **2000** |

OpenRouter documents the invariant this violates: *"max_tokens must be strictly
higher than the reasoning budget to ensure there are tokens available for the
final response after thinking."* Reasoning tokens are billed as output tokens,
so a model that uses its whole budget is cut off inside the thinking block and
returns `content types: ['thinking']` with no text — the sole condition that
advances the no-text retry loop, at up to four extra billed calls each.

`anthropic_thinking_kwarg` already says in its own docstring that the budget
exists *"so thinking cannot consume the entire max_tokens ceiling (Nemotron
otherwise often returns only thinking/redacted_thinking)"*. It just never
compared the two numbers.

## What this does

Compares them in `_OpenRouterMessages.create`, the first point where both the
per-request ceiling and the budget are in scope, and clamps the budget to leave
512 tokens for an answer.

Off by default. With `OPENROUTER_FIT_REASONING_TO_MAX_TOKENS` unset the request
is byte-for-byte what it is today — asserted by
`test_disabled_by_default_reproduces_current_behaviour`.

## Results — the honest part

**It does not work for nemotron.** With the clamp verified applied
(`reasoning.max_tokens=1488`, `thinking.budget_tokens=1488`), every failure
still came back at exactly 2000 output tokens. The model reasoned past its
budget and consumed the whole ceiling anyway.

| arm | ceiling | attempts/decision | retries |
|---|---:|---:|---:|
| upstream default | 2000 | 2.93 | 27 |
| this change | 2000 | 2.29 | 18 |
| raise the ceiling instead | 4096 | **1.14** | **2** |

2.93 → 2.29 across single runs does not clear run-to-run variance, and the
failure signature is unchanged. **Nemotron via OpenRouter treats the reasoning
budget as advisory.**

What does work on the same window is raising the ceiling — and it is *cheaper*,
because a truncated attempt is billed in full, thrown away, and drags a fresh
copy of the whole prompt with it: −61% input tokens, −28% wall time, −56% cost.
That needs no code; `LLM_MAX_OUTPUT_TOKENS` already exists.

**But do not just raise the default.** On the one window measured, the higher
ceiling produced *worse* trading results (return −0.19% → −0.71%). Neither arm
fell back to the rule-based agent, so that is not fallback contamination. One
3-day window cannot settle it, and the project's own gate refuses to
generalise from a single window.

## Recommendation

- Merge this only if you want the invariant enforced for providers that honour
  the budget. It is inert for nemotron and off by default.
- Do **not** rely on it as the remedy for the retry amplification.
- The lever worth building next is different and not in this PR: upstream
  already escalates to a larger ceiling on attempt five. Moving that escalation
  to the **first** retry leaves the first request unchanged — so it does not
  alter results the way raising the default ceiling does — while removing most
  of the wasted calls.
