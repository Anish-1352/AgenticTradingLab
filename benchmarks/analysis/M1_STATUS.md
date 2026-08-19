# M1 — open-weight substitution and the cost of model diversity

Status against the four priorities. **`P2` did not run: there is no GPU in this
environment.** `P0` and `P1` are complete, `P3` is blocked on `P2` and says what it is
blocked on rather than estimating around it.

## P0 — push and open the PR

**The push was already done.** `feature/serving-cost-reduction` has been on the
remote since the previous session at `3559c09`; the brief's premise that it had
never been pushed is out of date. Nothing about `llm_call_usage` is blocked on
a push.

**Divergence re-verified, as asked.** `origin/main` is still at `030b177` — the
commit the branch was cut from — so the branch remains a clean fast-forward and
a reviewer's first impression will not be a conflict [MEASURED].

One self-inflicted false alarm worth recording: a naive `grep` for conflict
markers reported one hit. It was matching `ON CONFLICT (cache_key) DO UPDATE`
— Postgres upsert syntax in the branch's own added code, not a merge conflict.
`git merge-base --is-ancestor` confirms the fast-forward [MEASURED].

**Opening the PR is blocked and needs one click.** The `gh` CLI is not
installed here and no `GH_TOKEN` / `GITHUB_TOKEN` is set, so I cannot open it:

`https://github.com/Anish-1352/AgenticTradingLab/pull/new/feature/serving-cost-reduction`

The reviewer-facing description is already written and committed at
`dashboard/docs/PR_serving_cost_reduction.md` — paste it into the PR body.

## P1 — the cost-per-alpha harness (complete)

`analysis/multi_window_alpha.py`, with 21 tests [MEASURED]. It consumes N leaderboard runs
across non-overlapping windows and reports, per model: total cost, return,
Sharpe, max drawdown, cost per unit of risk-adjusted return, **and the
across-window spread** — the quantity a single window cannot supply.

**The gates are imported from `alpha_per_dollar`, not re-derived.** A test
asserts that by identity (`mw.windows_needed is ap.windows_needed`), because a
second copy of a statistical gate is a second place to get it wrong.

Behaviour on the one window that exists today:

| Property | Value | Tier |
|---|---|---|
| windows | 1 | MEASURED |
| `can_rank` | False | MEASURED |
| readiness state | `insufficient` | MEASURED |
| across-window spread | **unestimable**, not zero | MEASURED |
| `windows_needed()` | refuses | MEASURED |

Reporting a spread of `0.0` from one sample would read as "perfectly stable",
which is the opposite of what one draw supports — so the field is withheld with
a reason attached.

**`can_rank` requires separability, not just window count.** Exercised against
a synthetic three-window database, the harness reaches `state: ready` and
`windows_needed()` becomes answerable, yet `can_rank` stays **False** because
the intervals still overlap [MEASURED] (on synthetic data). That is the correct
behaviour and the reason the gate is not simply `n_windows >= 2`.

*The synthetic data validates the code path. It is not a finding and no number
from it appears in any report.*

### The single-window result this extends

Unchanged and not replaced [MEASURED]: Spearman rho `+0.071` across the `194x`
price span; no model beat `spy_index` on Sharpe (`5.95%` at `5.78`, zero LLM
cost); `deepseek_v4_pro` beat SPY on **raw return** (`7.49%`) and lost on
Sharpe. Both framings are carried through the multi-window path — the harness
reports `beat_baseline_on_sharpe` and `beat_baseline_on_return` as separate
lists rather than collapsing them.

### Why this is the largest lever in M1

Model choice spans `134x` in the sensitivity ranking, more than every other
lever combined [DERIVED]. If the premium buys no performance across several
windows, open-weight substitution is justified **on evidence rather than on
budget** — and that is a configuration change, not an infrastructure programme.

The harness is ready now so that the windows can be consumed the moment they
land, rather than the analysis being written under time pressure once they do.

## P2 — the GPU sweep (not run)

```
nvidia-smi        -> not found
torch             -> not installed
vllm              -> NOT installed
```

No card, no CUDA, no vLLM [MEASURED]. None of the six runs can execute here and no substitute was
attempted — a CPU or different-hardware run would produce exactly the
uninterpretable dataset the setup constraints exist to prevent.

### The download estimate, made auditable

`P2` flags model download as the uncertain term rather than the runs. It is, and
it dominates.

The only throughput datapoint available is `15.24 GB in 216s` over Drive FUSE,
which is `0.071 GB/s` [MEASURED] (n=1). Applying it, and accounting for the fact
that later runs reuse checkpoints earlier runs already pulled:

| Run | New GB | Download min | Run min | A100 units | Cumulative | Tier |
|---|---:|---:|---:|---:|---:|---|
| 1 · N=1 AWQ **(GATE)** | 4.39 | 1.0 | 0.75 | 0.35 | 0.35 | DERIVED |
| 2 · N=4 AWQ | 5.15 | 1.2 | 0.75 | 0.39 | 0.74 | DERIVED |
| 3 · N=2 AWQ | 0.00 | 0.0 | 0.75 | 0.15 | 0.89 | DERIVED |
| 4 · N=7 AWQ | 12.69 | 3.0 | 0.75 | 0.74 | 1.62 | DERIVED |
| 5 · N=1 FP16 | 15.24 | 3.6 | 0.75 | 0.86 | 2.48 | DERIVED |
| 6 · N=4 FP16 | 17.88 | 4.2 | 0.75 | 0.98 | 3.46 | DERIVED |

**The whole sweep is ~3.5 units of the 25 available, about a third of the ~10 allotted** [DERIVED].

The minimum viable result — the gate plus the two smallest multi-model runs — is `0.89` units [DERIVED].

Two things that estimate makes visible:

- **Download is roughly 4x compute** — all six runs' inference is ~0.9 units [DERIVED], and the rest is pulling weights. Optimising run configuration would be optimising the small term.
- **The unquantized runs cost the most for the least** — they are `1.84` of the `3.46` units [DERIVED], and they are refinement rather than headline. If the session is cut, they are the right thing to lose.

The local HF cache holds only `11 MB` for the anchor model — tokenizer and
config from the fixture build, **no weights** [MEASURED] — and it is this Mac's
cache, not Colab's. Check `HF_HOME` inside the session before trusting any of
the above; a warm cache removes most of the cost.

### One thing to check before running

`--n-models 1` selects the **smallest** roster entry (`Qwen2.5-0.5B`), not the
`Qwen2.5-7B` anchor the gate needs. `heterogeneous_sweep.yaml` handles this correctly —
every `n_models: 1` entry pins `models: [Qwen/Qwen2.5-7B-Instruct]` explicitly
[MEASURED]. Drive the sweep from the config, not from `--n-models`, or the gate
will validate against the wrong model and reproduce nothing.

## P3 — synthesis (blocked, and on what)

Every deliverable in this section is a function of measurements that do not exist yet:

| Deliverable | Blocked on | Tier |
|---|---|---|
| models per card, measured | run 4 (N=7) | NOT MEASURED |
| the fragmentation curve | runs 1-4 | NOT MEASURED |
| the quantization effect | runs 5-6 against 1-2 | NOT MEASURED |
| cost per agent-decision by configuration | the above | NOT MEASURED |
| heterogeneous self-hosting threshold | models-per-card | NOT MEASURED |

`self_hosted_cards()` already refuses to compute without a measured
`models_per_card` rather than defaulting, so the blockage is structural rather
than a note — nothing downstream can quietly produce a number from nothing.

### The framing that survives without the GPU data

**Self-hosting addresses only the minority of the projected bill that is not on
closed-weight models — unless models are substituted.** Four of the seven
models ATL serves — GPT-5.5, Gemini, Claude Sonnet and Claude Haiku — have no public weights, and at an even split they carry `93%` of the projected spend [DERIVED]. No card purchase touches that.

So the heterogeneous sweep does not answer "what would it cost to self-host
what we run today". It answers "if we substituted open models, how many fit per
card and at what throughput". **The substitution is a capability decision, and
it rests on `P1`'s result, not on `P2`'s.** That ordering matters: if several
windows show the price premium buying no performance, the substitution is
justified and the sweep tells you what it costs to host. If they show the
opposite, the sweep answers a question nobody needs asked.

## What remains unobservable

`stats_source` was null on this vLLM build, so KV usage and prefix-cache hit
rate will very likely be unobtainable again. **The throughput delta across N is
the observable that survives that gap** — it is measured directly and does not
depend on the engine reporting anything about its cache. The runner records
absence as `available: False` with the probes it tried, never as a zero.
