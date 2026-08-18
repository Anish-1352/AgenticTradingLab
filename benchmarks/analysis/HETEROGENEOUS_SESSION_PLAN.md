# Heterogeneous serving — session plan and what it can answer

Built and dry-run only. There is no GPU and no `vllm` in this environment, so
nothing here has been executed against a card.

## The finding that came before any run

**93% of the hosted bill is on models that cannot be self-hosted at all** [DERIVED].

At 300 agents, Nof1 equity cadence, 3 calls/decision [ASSUMED — cadence and depth not confirmed with advisor]:

| Model | Monthly USD | Self-hostable? | Tier |
|---|---:|---|---|
| `openai/gpt-5.5` | 36,332 | **no public weights** | DERIVED |
| `google/gemini-3.1-pro` | 29,466 | **no public weights** | DERIVED |
| `anthropic/claude-sonnet-4-6` | 12,928 | **no public weights** | DERIVED |
| `anthropic/claude-haiku-4-5` | 4,514 | **no public weights** | DERIVED |
| `qwen/qwen3.7-plus` | 4,168 | yes | DERIVED |
| `deepseek/deepseek-v4-pro` | 1,978 | yes | DERIVED |
| `nvidia/nemotron-3-nano-30b-a3b` | 188 | yes | DERIVED |
| **Total** | **89,574** | — | DERIVED |
| **Not self-hostable** | **83,239 (93%)** | — | DERIVED |

Self-hosting can address **$6,334/month** of an **$89,574** bill [DERIVED]. The remainder is spend no card purchase touches, because OpenAI, Google and Anthropic do not publish weights.

That does not make the sweep pointless — it changes what it is for. It answers
"if we *substituted* open models, how many fit per card and what would that
cost", which is a **capability-substitution decision**, not a
lift-and-shift saving. Those are different propositions and only the first has
a self-hosted answer.

*The brief cited ~135k calls/day, ~$110k/month, GPT-5.5 plus Gemini ~$72k [NOT MEASURED] — figures as given. Computing from the seven verified per-call costs gives 140,400 calls/day, $89,574, and $65,798 for those two [DERIVED]. Same order, roughly a fifth apart — likely a different decisions/day or model split. The figures above are the ones this repository derives from measured unit costs.*

## What vLLM 0.26 actually supports

`AsyncEngineArgs` takes `model` — **singular**. There is no multi-model parameter. N models means N engines, each with its own scheduler and KV pool, which means **they do not share a batch**.

Arm C's 9.28 req/s came from 32 requests coalescing in one scheduler [MEASURED]; N schedulers cannot do that.

`gpu_memory_utilization` is a fraction of the **whole card**, claimed at init. Passing 0.9 to each of four engines OOMs the last three [DERIVED]. The runner divides it by N — that is not tuning, it is what makes N engines start.

**Not verified against an installed vLLM** (none here). It follows from the API
the existing arm C runner already uses plus documented `gpu_memory_utilization`
semantics. First action on the card is `--dry-run`, then the validation gate.

Alternative if in-process engines fail: one engine per subprocess
(`--isolation process`, declared unimplemented rather than faked — it refuses rather than pretending).
**LoRA adapters on one base model** would genuinely share a batch, but that is
one base plus N fine-tunes, not seven architectures — the right answer to a
different question.

## VRAM decides the card

Weights only, from published parameter counts [DERIVED]:

| Configuration | Weights GB | A100 80GB | L4 24GB | Tier |
|---|---:|:---:|:---:|---|
| 7 models, FP16 | 77.16 | **no** | no | DERIVED |
| 7 models, AWQ 4-bit | 22.23 | yes | **no** | DERIVED |
| 6 models, AWQ | 13.72 | yes | yes | DERIVED |
| 4 models, AWQ | 5.15 | yes | yes | DERIVED |

Usable budget is card x 0.9: 72.0 GB on A100, 21.6 GB on L4 [DERIVED].

**Quantization is the enabling condition, not an optimization.** Seven models unquantized need 77.16 GB of weights against 72.0 GB usable, so the configuration cannot load and the runner refuses it before downloading anything [DERIVED].

**Card recommendation: A100 for the full sweep.** L4 holds up to six quantized models but not seven — the 14B roster entry breaks it at 22.23 GB against 21.6 GB usable [DERIVED]. L4 is viable for an N<=4 subset at roughly a third the unit rate [NOT MEASURED] — stated rates, and is the right fallback if units run short.

Every row excludes the KV cache, which vLLM allocates on top and which is **not
shared between engines**.

## Run order and budget

~25 compute units remain; A100 ~11.8 units/hr, L4 ~4.8 [NOT MEASURED] — stated rates.

Wall time is modelled as `n_models x 2 min load + 0.5 min bench` [NOT MEASURED] — an estimate.

The bench term scales arm C's 3.4 s at C=32 and 256 tokens by ~3.4x for 860 tokens, because decode is ~98% of GPU time [MEASURED].

**Load time dominates and is the least certain term** — a cold 14B pull can exceed two minutes, and Colab bills the wait [NOT MEASURED].

| # | Run | Models | Est min | A100 units | Cumulative | Tier |
|---:|---|---:|---:|---:|---:|---|
| 1 | `n1_none_validation` (256 tok) | 1 | 2.5 | 0.49 | 0.49 | DERIVED |
| 2 | `n7_awq` (860 tok) | 7 | 14.5 | 2.85 | 3.34 | DERIVED |
| 3 | `n4_awq` | 4 | 8.5 | 1.67 | 5.02 | DERIVED |
| 4 | `n1_none_860_control` | 1 | 2.5 | 0.49 | 5.51 | DERIVED |
| 5 | `n4_none` | 4 | 8.5 | 1.67 | 7.18 | DERIVED |
| 6 | `n2_awq` | 2 | 4.5 | 0.89 | 8.06 | DERIVED |
| 7 | `n1_awq` | 1 | 2.5 | 0.49 | 8.56 | DERIVED |

All seven fit in **~8.6 units of a 25-unit budget**, roughly 3x headroom — the margin that absorbs a bad load-time estimate [DERIVED].

**Run 1 is a gate, not a warm-up.** It uses the same model, concurrency and 256-token output as `armC_shared` [MEASURED]. If it does not reproduce arm C's behaviour, the multi-model runner is wrong and the rest of the sweep is uninterpretable — stop and fix rather than continuing.

One caveat is built into the config: the gate run uses `atl_realistic` where `armC_shared` used `shared_prefix`, so **throughput is expected to differ**. What must reproduce is the mechanism. If the difference needs attributing, `n1_none_sharedprefix_control` is an exact-match control.

The highest-N runs are placed early because they carry the most information about fragmentation; a preemption partway still leaves the headline measurable.

## Why atl_realistic, and why 860 tokens

`atl_realistic` (`626e7494…`, 10.2% overlap), not `shared_prefix` (99.4%) [MEASURED].

The shared-prefix fixture is a deliberate upper bound, and **prefix caching cannot span engines anyway** — so it would flatter the multi-model result by exactly the mechanism the multi-model case cannot use.

Output length is 860 tokens, matching production Nemotron, not the 256 used previously [MEASURED].

Decode is ~98% of GPU time — 0.55 s prefill against 31.08 s decode in the Layer-3 trace [MEASURED] — so throughput scales roughly inversely with output length.

**Every existing agents-per-card figure is therefore about 3.4x optimistic** [DERIVED]. The shorter length stays available as a control so the two can be compared rather than argued about.

## What the sweep produces, and what stays unmeasurable

The deliverable is two numbers: **models-per-card** and **cards-for-the-fleet**. Everything else divides by them, which is why `self_hosted_cards()` refuses to compute when `models_per_card` is `None` rather than defaulting — inventing it would produce a confident cost from nothing.

**Still unmeasurable on this build:** KV usage and prefix-cache hit rate. Arm C found `stats_source: null` on vLLM 0.26, and running N engines does not change that [MEASURED].

The runner records absence as `available: False` with the probes it tried, never as a zero — "the engine did not report" and "nothing was reused" are different facts and only the second is a result.
