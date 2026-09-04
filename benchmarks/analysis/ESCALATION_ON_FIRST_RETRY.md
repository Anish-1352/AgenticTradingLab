# Escalating the output ceiling on the first retry

**Tiers:** `MEASURED` = observed in the paired runs recorded here. `DERIVED` = arithmetic on those. `NOT MEASURED` = not established.

## The change

An empty reply is what a reply looks like when reasoning ran out the output ceiling before any text was emitted — every such failure came back at exactly `max_tokens`, and none below it [MEASURED]. Asking again at the ceiling that just failed asks for the same failure. The single-prompt loop does that four times before raising it, on the `5`th attempt.

`pipeline_runner` already disagrees: an empty pipeline response is retried **once**, immediately at the recovery ceiling [MEASURED]. So the same failure costs one extra call on one path and five on the other. This makes them agree.

The first request is untouched — in call shape, not merely in resolved value — so a decision that succeeds first time issues exactly the request it does today [MEASURED]. That is what keeps this a cost change rather than a behaviour change.

## What it does

Arms are **paired and interleaved** over `3` runs each. That matters more than it sounds: two runs of the same configuration on the same window came out at `2.93` and `2.21` attempts per decision [MEASURED], so a single-run comparison of this flag would be measuring the day, not the change.

| | off | on | delta | Tier |
|---|---:|---:|---:|---|
| decisions | 42 | 42 | — | MEASURED |
| attempts | 112 | 79 | — | MEASURED |
| attempts/decision | 2.67 | 1.88 | -29.5% | MEASURED |
| retries | 70 | 37 | -47.1% | MEASURED |
| decisions needing >= 4 attempts | 14 | 0 | — | MEASURED |
| input tokens | 224255 | 150130 | -33.1% | MEASURED |
| output tokens | 241848 | 190003 | -21.4% | MEASURED |
| wall seconds | 1531.21 | 1426.74 | -6.8% | MEASURED |
| cost USD | 0.0596 | 0.0455 | -23.6% | MEASURED |

| attempts | decisions (off) | decisions (on) | Tier |
|---:|---:|---:|---|
| 1 | 13 | 10 | MEASURED |
| 2 | 11 | 27 | MEASURED |
| 3 | 4 | 5 | MEASURED |
| 4 | 5 | 0 | MEASURED |
| 5 | 9 | 0 | MEASURED |

### The tail is what moved

Decisions that burned `4` or more attempts: `14` of `42` with the flag off, `0` of `42` with it on [MEASURED]. One-sided Fisher exact `p = 0.0000` [DERIVED].

The worst decision seen went from `5` attempts to `3` [MEASURED]. That is the mechanism doing exactly what it was built to do: it cannot stop the first failure, only stop the pipeline re-asking for it.

### Where the saving comes from

**Input tokens fall furthest** (`-33.1%`) [MEASURED]: a retry drags a fresh copy of the whole prompt with it, so removing retries removes prompts, not just completions.

**Output tokens fall less** (`-21.4%`) [MEASURED], because each escalated retry returns a longer reply — the change trades several truncated completions for one complete one.

**Wall time is roughly unchanged** (`-6.8%`) [MEASURED]. Fewer calls, each generating more tokens, very nearly cancel. This is not a latency fix and should not be adopted as one.

## Honest summary

| claim | supported? | Tier |
|---|---|---|
| Removes the `4`–`5` attempt tail | yes, `p = 0.0000` | MEASURED |
| Fewer calls per decision | yes, `-29.5%` | MEASURED |
| Cheaper | yes, -23.6% | MEASURED |
| Faster | no — flat at -6.8% | MEASURED |
| Changes decisions that succeed first time | no — identical request | MEASURED |

So it is cheaper, and cheaper for the reason the ceiling work predicted: a truncated attempt is billed in full and thrown away [DERIVED]. The larger prize is the tail — the decisions that exhaust the loop are the ones that fall back to the rule-based agent, and `14` of them across `42` decisions became `0` [MEASURED].

It also makes behaviour far more consistent. Across three runs of the same configuration the flag-off arm ranged `2.21`–`3.36` attempts per decision; with the flag on the three runs came out `1.93`, `1.93`, `1.79` [MEASURED]. Removing the tail removes most of the variance with it.

### Did it change the trading?

| arm | returns | mean | sd | Tier |
|---|---|---:|---:|---|
| off | `-0.0092`, `-0.0053`, `-0.0010` | -0.0052 | 0.0034 | MEASURED |
| on | `-0.0037`, `-0.0032`, `0.0017` | -0.0017 | 0.0025 | MEASURED |

Mean return moved `-0.0052` → `-0.0017` [MEASURED] — about `1.0` standard deviation on `3` runs a side [DERIVED], which settles nothing.

What it does show is the thing worth showing: **it did not get worse.** Raising the default ceiling changed every request and made the same window's return worse; this changes only the requests that had already failed, and returns moved slightly the other way [MEASURED]. That asymmetry is the argument for this lever over that one.

## What this does not establish

- **Any effect on returns** [NOT MEASURED]. The first request is unchanged, so first-time successes are identical; decisions that needed a retry may differ, and this work did not compare them.
- **Behaviour on other models** [NOT MEASURED]. Only nemotron, which is the model that exhibits the failure.
- **Whether one window generalises** [NOT MEASURED]. Runs are replicated on one window, which controls run-to-run noise but not the choice of window.

