# The decision schema, and whether it can be trained against

**Tiers:** `MEASURED` = observed by calling the shipped code or in a recorded run. `DERIVED` = arithmetic on those. `NOT MEASURED` = not established here.

Everything below was measured against `origin/main` at `4388762f` [MEASURED]. The first run of the probe used a local checkout `533` commits behind and reported parser behaviour that has since been fixed, so the revision is recorded with the result.

## Answer: no, not yet — there is more than one schema

Three code paths turn model output into a decision and they do not agree. On the same `39` payloads the permissive pipeline parser and the strict pydantic validator agree only `21%` of the time; `31` payloads are accepted by one and rejected by the other [MEASURED].

Worse for a training target, **one parser emits `3` different action shapes** depending on which envelope the model chose [MEASURED]. A model tuned to satisfy the parser would still not know what its consumer receives.

## 1. What the schema actually is

### The three consumers

| path | parser | behaviour | Tier |
|---|---|---|---|
| multi-step pipeline | `pipeline_runner.pipeline_output_to_decision` | permissive, coercive | MEASURED |
| single prompt | `portfolio_manager.make_trading_decision_with_llm` | consumes `actions`, applies its own rules | MEASURED |
| external agents, AI hedge fund | `validator.parse_actions_payload` | pydantic; rejects rather than coerces | MEASURED |

### What the pipeline parser accepts

`34` of `40` constructed payloads were accepted [MEASURED]. Three envelopes are recognised — `actions`, `orders`, `risk_actions` — and the first match wins.

| case | accepted | what came out | Tier |
|---|---|---|---|
| actions, non-empty | yes | `1` action(s) | MEASURED |
| orders, non-empty | yes | `1` action(s) | MEASURED |
| risk_actions, non-empty | yes | `1` action(s) | MEASURED |
| actions, empty | yes | `{"actions": []}` | MEASURED |
| orders, empty | yes | `{"actions": []}` | MEASURED |
| risk_actions, empty | yes | `{"actions": []}` | MEASURED |
| unknown key only | no | — | MEASURED |
| empty object | no | — | MEASURED |
| a list, not an object | no | — | MEASURED |
| orders is not a list | no | — | MEASURED |
| orders of non-objects | no | — | MEASURED |
| template shape verbatim | yes | `1` action(s) | MEASURED |

### What it does NOT constrain

These are the places a tuned model could emit valid-but-wrong output [MEASURED]:

| input | result | Tier |
|---|---|---|
| `side=short (unknown)` | silently becomes `hold` | MEASURED |
| `side=liquidate everything` | silently becomes `hold` | MEASURED |
| `side missing` | silently becomes `hold` | MEASURED |
| `qty=-5 (negative)` | passes through negative | MEASURED |
| `qty=1e9 (huge)` | passes through unbounded | MEASURED |
| `qty='many' (unparseable)` | silently becomes `0` | MEASURED |
| `qty=2.9 (float)` | truncated toward zero | MEASURED |
| `symbol=None` | accepted, symbol stays `None` | MEASURED |
| `symbol not in DJIA` | accepted unchanged at this layer | MEASURED |
| `symbol is a number` | accepted unchanged | MEASURED |
| `confidence=2.5 (out of range)` | accepted, no range check | MEASURED |
| `confidence=-1` | accepted, no range check | MEASURED |
| `unknown extra field` | ignored silently | MEASURED |

The one input that is **not** handled gracefully is a non-numeric `confidence`: `float("high")` raises `ValueError` out of the parser [MEASURED]. It is caught by the broad handler in `portfolio_manager`, which prints "LLM decision error" and substitutes a rule-based decision — so a model writing `"confidence": "high"`, an entirely natural thing for a language model to do, silently loses the whole bar [MEASURED].

**Fields the shipped templates ask for and no parser reads:** `order_type`, `limit_price` [MEASURED]. Every model is being asked for them on every call.

### Do the templates agree?

Yes, with each other. All `11` shipped templates that define a pipeline end in the same envelope: `orders` [MEASURED]. The three-step template's intermediate steps use `facts` and `signals`, which is correct — only the last step's output is converted.

They do **not** agree with the single-prompt path. The templates ask for `orders`; `validator.py`'s prompt asks for `actions` with different field names — `action` not `side`, `position_size` not `qty`, `reasoning` not `reason` [MEASURED]. Both parse, because the pipeline parser accepts either, but they are two different output contracts for the same job.

## 2. What real traffic looks like

`325` prompt/response pairs were captured from real backtests against real Alpaca bars [MEASURED] — not from the `atl_realistic` fixture, whose own metadata records that it was built from assumed proportions.

### The single largest number in this report

Of `480852` output tokens billed, roughly `84362` appear in the response text [DERIVED]. **About `82%` of what is paid for is never shown at all** — it is the reasoning block [DERIVED].

That is the quantity a tuned model removes. It is not prose the parser ignores; it is thinking the response never contains.

| response shape | count | Tier |
|---|---:|---|
| `['thinking', 'text']` | 220 | MEASURED |
| `['thinking']` | 76 | MEASURED |
| `['text']` | 29 | MEASURED |

`76` responses carried a thinking block and no text at all [MEASURED] — the failure Phase 21 measured, reproduced here.

### The two paths are not equally reliable

| regime | path | n | parsed | hit ceiling | mean output tokens | mean prompt chars | Tier |
|---|---|---:|---:|---:|---:|---:|---|
| downtrend | `pipeline` | 63 | 100% | 0% | 903 | 1877 | MEASURED |
| downtrend | `single_prompt` | 62 | 29% | 73% | 2161 | 5370 | MEASURED |
| flat | `pipeline` | 42 | 100% | 0% | 783 | 1823 | MEASURED |
| flat | `single_prompt` | 36 | 36% | 64% | 2092 | 5315 | MEASURED |
| uptrend | `pipeline` | 63 | 100% | 0% | 798 | 1792 | MEASURED |
| uptrend | `single_prompt` | 59 | 29% | 73% | 2228 | 5274 | MEASURED |

The grid is fully crossed, which matters: read as a regime column alone the earlier partial data said "uptrend parses 29%", when what it measured was the single-prompt path [MEASURED].

**The pipeline path did not fail once.** `168` calls, `100%` parsed, `0%` hit the ceiling. The single-prompt path: `157` calls, `31%` parsed, `71%` hit the ceiling [MEASURED].

The mechanism is prompt size. A pipeline step's prompt averages `1831` characters against `5321` for the single prompt [MEASURED], and the smaller ask leaves room for both reasoning and an answer inside the same ceiling [DERIVED].

Final-step conversion: `56` of `56` pipeline bars produced a decision [MEASURED]. Intermediate steps emit `facts` and `signals` and are correctly declined by the decision parser; counting those as failures would manufacture a defect that is not there.

## 3. The empty decision

Phase 16 established that a well-formed `{"orders": []}` returned `None`. **That has been fixed upstream** — it now returns `{"actions": []}` [MEASURED]. The substitution did not disappear though; it moved.

| model output | `strict_llm=False` | `strict_llm=True` | Tier |
|---|---|---|---|
| `{"actions": []}` | **rule-based substituted**, `llm_decisions` stays `0` | accepted, `llm_decisions` `+1` | MEASURED |
| `{"orders": []}` | **rule-based substituted** | accepted | MEASURED |
| an explicit `hold` action | accepted | accepted | MEASURED |

So the strict branch already treats "do nothing" as a real decision, and its own comment says so; the non-strict branch at `portfolio_manager.py:591-593` still swaps in the reference agent and records nothing [MEASURED].

**What a "no trades" training example should look like:** explicit `hold` actions, one per symbol under consideration — not an empty array. The empty array survives only the strict path; an explicit hold survives both [MEASURED]. Teaching a model to emit `[]` would teach it an output that loses the decision on the default path.

## 4. May the collected data be used?

The pairs come from `nvidia/nemotron-3-nano-30b-a3b` served through OpenRouter, so two documents govern them.

| source | position | Tier |
|---|---|---|
| OpenRouter terms | output rights are delegated: "Your ownership rights in the Output are set forth in the Model Terms for each Model you use" | MEASURED |
| NVIDIA Open Model License | "NVIDIA claims no ownership rights in outputs"; derivative models are permitted with attribution | MEASURED |

So this particular collection is usable as a training target, subject to the licence's attribution condition and its Trustworthy AI terms [MEASURED]. That is a reading of the published terms, not legal advice, and anything shipped should be confirmed with counsel [NOT MEASURED].

The choice of provider is what makes this true. Had the pairs been collected from an OpenAI or Anthropic model, their terms generally prohibit using outputs to train competing models, and the data would have to be discarded rather than relabelled [NOT MEASURED — their terms were not fetched for this report, because no pair here came from those providers].

## 5. Is this trainable?

| question | answer | Tier |
|---|---|---|
| Is the schema stable across paths? | No — `21%` agreement between the permissive and strict parsers | MEASURED |
| Stable across templates? | Yes — all shipped pipelines end in `orders` | MEASURED |
| Does the prompt ask for what the parser accepts? | Not exactly — templates ask `orders`, the single-prompt path asks `actions` | MEASURED |
| How much output is schema-mandated? | Roughly `18%` is visible text at all; the rest is reasoning | DERIVED |
| What would "correct" mean? | Schema validity, which is checkable. Decision quality is not, and must not be substituted for it | MEASURED |
| May the data be used? | Yes for this provider, with attribution | MEASURED |

### The recommendation this leads to

A schema-compliance target **is** well defined, but not against "the ATL schema" as a whole, because there is no such single thing. It would have to name one parser and one envelope. The obvious choice is the pipeline parser with the `orders` envelope, since every shipped template already targets it.

There is an awkward finding for the motivation, though. The failure this was meant to remove — a reasoning model emitting no text — did not occur once on the pipeline path in `168` calls [MEASURED]. Decomposing the work into smaller steps already removes it, without training anything.

So the honest ordering is: the schema needs unifying before it can be a target, and the failure that motivated the target is already avoidable by configuration. Fine-tuning may still be worth doing for cost or latency — a small model emitting `orders` in few tokens is cheaper than a reasoning model emitting `~80%` invisible tokens [DERIVED] — but it should be argued on that basis, not as a fix for a defect that decomposition already fixes.

## 6. What this does not establish

- **That schema compliance predicts good trading** [NOT MEASURED]. It is deliberately not a claim about decision quality.
- **Behaviour of models other than the one sampled** [NOT MEASURED].
- **That the pipeline path is reliable in general** [NOT MEASURED]. It did not fail in these windows; that is not the same as cannot.

