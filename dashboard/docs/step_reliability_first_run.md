# Parse reliability, first measured run

**Run:** `pipeline-analyst` (the shipped 3-step template), 15 attempts per model
per step, `--mode isolated`, via OpenRouter.
**Spend:** $0.0349 actual against a $0.1049 ceiling, 90 calls.
**Command:** `dashboard/scripts/measure_step_reliability.py --integration openrouter --models nvidia/nemotron-3-nano-30b-a3b deepseek/deepseek-v4-pro --attempts 15`

## What came back

| model | step | passed | mean out tok | categories |
|---|---|---:|---:|---|
| nemotron-3-nano-30b | Information Gathering | 15/15 | 1458 | `ok` ×15 |
| nemotron-3-nano-30b | Information to Signal | 15/15 | 327 | `ok` ×15 |
| nemotron-3-nano-30b | Signal to Execution | **0/15** | — | `empty_actions_no_trade` ×15 |
| deepseek-v4-pro | Information Gathering | 15/15 | 1197 | `ok` ×15 |
| deepseek-v4-pro | Information to Signal | 15/15 | 388 | `ok` ×15 |
| deepseek-v4-pro | Signal to Execution | **0/15** | — | `empty_actions_no_trade` ×15 |

## The headline is not about the models

**A well-formed "make no trades this hour" aborts the decision.** Both models
returned exactly what the template asks for:

```json
{ "orders": [] }
```

`pipeline_output_to_decision` returns `None` for that, because it requires a
**non-empty** list. `run_pipeline_decision` treats `None` as a failed step and
returns no decision. `portfolio_manager` then either raises `LLMDecisionError`
or **silently falls back to rule-based trading**.

Verified with no model involved:

```
{"orders": []}                                        -> None
{"actions": []}                                       -> None
{"orders": [{"symbol":"AAPL","side":"hold","qty":0}]} -> {'actions': [...]}
```

"Trade nothing" is the correct decision most hours, and the pipeline cannot
express it. A model must emit an explicit `hold` order to say what an empty
list says more naturally. This is independent of routing and was not a known
issue — the first two steps' 30/30 clean parses show the models are not the
problem.

**On the leaderboard this is worse than an error**, because the fallback is
silent: the entry keeps running under the model's name while the decisions are
rule-based. How often that fires in practice is not measured here and is the
first thing worth checking against real run logs.

## What this run does NOT establish

**It is not a verdict on either model at step 3.** The isolated mode feeds each
step a frozen upstream context built from the template's own `outputFormat`
placeholders — right shape, no real content. Given a signal object full of
placeholder values, "no trades" is a *reasonable* answer, so the 0/15 conflates
a genuine pipeline limitation with an artifact of a content-free fixture.
Rerunning with a captured real upstream context (`--mode endtoend`, or a
reference captured from a live run) is needed before any claim about step-3
model quality.

What the run does establish, and what does not depend on the fixture:

- **Steps 1–2 parsed 30/30 on both models, cleanly** — no fences, no prose
  wrappers, no truncation. On the mechanical steps, at least, a cheap model is
  not obviously the risk that motivated measuring first.
- **Nemotron is not terser than deepseek at step 1** — 1458 vs 1197 mean output
  tokens. The compounding benefit assumed for a cheap model at step 1 runs
  *backwards* here: nemotron's longer step-1 output would inflate steps 2–3's
  input. This is exactly why the projection takes measured token counts rather
  than assuming a cheap model is terse.
- **The two-gate structure is real and the second gate is where things fail.**

## What to do next

1. **Decide on the empty-orders behaviour.** It is a production correctness
   question that outranks routing and is not in this branch's scope. Options:
   treat an empty list as an explicit no-trade decision, or require models to
   emit explicit `hold` orders and validate templates against that.
2. **Re-measure step 3 against a real upstream context** before drawing any
   conclusion about which model can hold the decision step.
3. **Then, and only then**, project cost. `routing_cost.project_routing`
   refuses to produce an effective saving without these numbers, which is why
   no saving figure appears in this document.
