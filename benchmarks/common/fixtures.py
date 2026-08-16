"""Deterministic prompt fixtures at an exact token length.

Two sets, differing only in how much prefix they share:

``shared_prefix``
    One identical system prompt + one identical market-context block, with a
    short per-ticker tail. Models a fleet of agents reasoning over the same
    task context — the case a prefix/prompt cache is supposed to win.

``low_overlap``
    Distinct synthetic filing text per request, same exact token length. The
    case a prefix cache cannot help.

**The gap between the two sets is the prefix-caching result.** Running only
one of them measures a cache hit rate that is an artifact of the fixture.

WHY THIS IS NOT ``"...text..." * N``
------------------------------------
v1 built its context as a single string repeated 200 times, and every request
used the identical prompt. Under any prefix-caching runtime that is a ~100%
hit rate by construction — it measures the cache, not the workload, and the
number does not generalize to an agent fleet whose prompts differ. Here the
shared portion is explicit and measured (``common_prefix_tokens``), so the hit
rate is a property of the fixture that gets reported rather than assumed.

TOKEN IDS ARE CANONICAL, TEXT IS REFERENCE
------------------------------------------
Fixtures carry ``prompt_token_ids`` and the runner feeds those to the model.
Two reasons:

1. **Exactness.** Truncating an id list to N gives exactly N tokens. Decoding
   to text and re-encoding does not reliably round-trip, so a text-based
   fixture cannot promise an exact length — and context length is a controlled
   variable in this study.
2. **A real shared prefix.** Concatenating *text* and re-encoding lets the
   tokenizer merge across the prefix/tail boundary, so the "shared" prefix can
   differ in its final token between requests and defeat the cache under test.
   Concatenating *ids* makes the shared region byte-identical by construction.

``prompt_text`` is a decode of the ids, stored for human inspection only. It
may not re-encode to the same ids; do not feed it to a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

__all__ = [
    "Tokenizer",
    "StubTokenizer",
    "load_hf_tokenizer",
    "Fixture",
    "build_shared_prefix",
    "build_low_overlap",
    "build_fixture",
    "save_fixture",
    "load_fixture",
    "resolve_fixture",
    "FIXTURE_BUILDERS",
]

DEFAULT_CONTEXT_TOKENS = 2620
DEFAULT_TAIL_TOKENS = 24
DEFAULT_SEED = 20260820


class Tokenizer(Protocol):
    """Minimal tokenizer surface this module needs."""

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class StubTokenizer:
    """Offline stand-in: one token per whitespace-delimited word.

    Exists so ``--dry-run`` and the unit tests work on a machine with no
    transformers install and no model download. It is **not** valid for real
    fixtures: token counts under it do not correspond to Qwen's tokenizer, so a
    fixture built with it would silently change the study's controlled context
    length.

    Fixtures record the tokenizer that built them, and the runner refuses to
    execute a real benchmark against a stub-built fixture.
    """

    name = "stub-whitespace"
    is_stub = True

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._inv: Dict[int, str] = {}

    def encode(self, text: str) -> List[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                idx = len(self._vocab) + 1
                self._vocab[word] = idx
                self._inv[idx] = word
            ids.append(self._vocab[word])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._inv.get(int(i), "<unk>") for i in ids)


def load_hf_tokenizer(model_id: str):
    """Load the real tokenizer. Imported lazily — transformers is GPU-host only."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_id)

    class _HF:
        name = model_id
        is_stub = False

        def encode(self, text: str) -> List[int]:
            return tok(text, add_special_tokens=False).input_ids

        def decode(self, ids: Sequence[int]) -> str:
            return tok.decode(list(ids), skip_special_tokens=True)

    return _HF()


# --------------------------------------------------------------------------
# Synthetic corpus
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a disciplined equity trading analyst operating under a strict risk "
    "mandate. You read the supplied market context and filing excerpts, weigh "
    "the evidence, and issue exactly one decision with a short justification. "
    "You never speculate beyond the provided material, never invent figures, "
    "and never exceed the stated position limits. Respond with a decision of "
    "BUY, SELL, or HOLD, a confidence between 0 and 1, and two sentences of "
    "reasoning grounded in the context above."
)

_SECTORS = (
    "semiconductors", "cloud infrastructure", "consumer discretionary",
    "industrial automation", "biotechnology", "regional banking",
    "energy transition", "logistics", "advertising technology", "medical devices",
)
_METRICS = (
    "gross margin", "operating margin", "free cash flow", "days sales outstanding",
    "inventory turnover", "customer acquisition cost", "net revenue retention",
    "backlog coverage", "capital expenditure intensity", "deferred revenue",
)
_DIRECTIONS = ("expanded", "contracted", "held flat", "recovered", "deteriorated")
_QUALIFIERS = (
    "against a difficult prior-year comparison", "on a constant-currency basis",
    "excluding one-time restructuring charges", "before stock-based compensation",
    "net of the divested segment", "adjusted for the extra fiscal week",
)
_RISKS = (
    "supply concentration among two vendors", "elevated channel inventory",
    "pending regulatory review", "covenant headroom narrowing",
    "customer concentration above thirty percent", "foreign exchange translation",
)
_TICKERS = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD", "CRM",
    "ORCL", "ADBE", "NFLX", "INTC", "QCOM",
)


def _sentence(rng: random.Random) -> str:
    return (
        f"In the {rng.choice(_SECTORS)} segment, {rng.choice(_METRICS)} "
        f"{rng.choice(_DIRECTIONS)} by {rng.randint(10, 940)} basis points "
        f"{rng.choice(_QUALIFIERS)}, while management flagged "
        f"{rng.choice(_RISKS)} as the principal near-term risk."
    )


def _paragraphs(rng: random.Random, min_words: int) -> str:
    """Deterministic synthetic filing text of at least ``min_words`` words."""
    parts: List[str] = []
    words = 0
    while words < min_words:
        para = " ".join(_sentence(rng) for _ in range(rng.randint(4, 7)))
        parts.append(para)
        words += len(para.split())
    return "\n\n".join(parts)


def _exact_ids(tokenizer: Tokenizer, text_factory, n_tokens: int) -> List[int]:
    """Encode text from ``text_factory(min_words)`` and truncate to exactly n.

    Grows the source until it encodes to at least ``n_tokens``, then slices.
    Slicing the id list is what makes the length exact; no decode/re-encode
    round trip is involved.
    """
    if n_tokens <= 0:
        return []
    min_words = max(int(n_tokens * 0.9), 16)
    for _ in range(12):
        ids = tokenizer.encode(text_factory(min_words))
        if len(ids) >= n_tokens:
            return list(ids[:n_tokens])
        min_words = int(min_words * 1.6) + 32
    raise RuntimeError(
        f"could not generate {n_tokens} tokens after 12 growth attempts "
        f"(reached {len(ids)}) — check the tokenizer"
    )


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


@dataclass
class Fixture:
    name: str
    tokenizer_name: str
    tokenizer_is_stub: bool
    seed: int
    context_tokens: int
    n_requests: int
    requests: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def canonical_payload(self) -> Dict[str, Any]:
        """The exact structure that gets hashed. Ids only — never the decoded
        text, which is derived and would make the hash tokenizer-version
        sensitive in a second, redundant way."""
        return {
            "name": self.name,
            "tokenizer_name": self.tokenizer_name,
            "seed": self.seed,
            "context_tokens": self.context_tokens,
            "n_requests": self.n_requests,
            "requests": [
                {"request_id": r["request_id"], "prompt_token_ids": r["prompt_token_ids"]}
                for r in self.requests
            ],
        }

    def sha256(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def token_counts(self) -> List[int]:
        return [len(r["prompt_token_ids"]) for r in self.requests]

    def common_prefix_tokens(self) -> int:
        """Length of the token prefix shared by *every* request.

        This is the quantity that determines a prefix cache's hit rate, so it
        is measured and recorded rather than assumed from the fixture's name.
        """
        seqs = [r["prompt_token_ids"] for r in self.requests]
        if not seqs:
            return 0
        if len(seqs) == 1:
            return len(seqs[0])
        shortest = min(len(s) for s in seqs)
        for i in range(shortest):
            first = seqs[0][i]
            if any(s[i] != first for s in seqs[1:]):
                return i
        return shortest

    def to_meta(self) -> Dict[str, Any]:
        counts = self.token_counts()
        common = self.common_prefix_tokens()
        return {
            "name": self.name,
            "sha256": self.sha256(),
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_is_stub": self.tokenizer_is_stub,
            "seed": self.seed,
            "context_tokens": self.context_tokens,
            "n_requests": self.n_requests,
            "token_count_min": min(counts) if counts else 0,
            "token_count_max": max(counts) if counts else 0,
            "token_counts_all_equal": len(set(counts)) <= 1,
            "common_prefix_tokens": common,
            "common_prefix_fraction": (common / self.context_tokens) if self.context_tokens else 0.0,
            **self.meta,
        }


def build_shared_prefix(
    tokenizer: Tokenizer,
    n_requests: int,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    seed: int = DEFAULT_SEED,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> Fixture:
    """Identical system prompt + market context; short per-ticker tail."""
    if tail_tokens >= context_tokens:
        raise ValueError("tail_tokens must be smaller than context_tokens")

    prefix_len = context_tokens - tail_tokens
    shared_rng = random.Random(seed)

    def _shared(min_words: int) -> str:
        rng = random.Random(seed)  # re-seeded: the prefix must be stable
        return (
            _SYSTEM_PROMPT
            + "\n\n--- SHARED MARKET CONTEXT ---\n\n"
            + _paragraphs(rng, min_words)
        )

    shared_ids = _exact_ids(tokenizer, _shared, prefix_len)
    del shared_rng

    requests: List[Dict[str, Any]] = []
    for i in range(n_requests):
        ticker = _TICKERS[i % len(_TICKERS)]

        def _tail(min_words: int, _t=ticker, _i=i) -> str:
            rng = random.Random(seed + 1_000_000 + _i)
            return (
                f"\n\n--- TICKER UNDER REVIEW: {_t} ---\n"
                f"Position id {_i}. " + _paragraphs(rng, min_words)
            )

        tail_ids = _exact_ids(tokenizer, _tail, tail_tokens)
        ids = list(shared_ids) + tail_ids
        requests.append(
            {
                "request_id": f"shared_prefix-{i:04d}",
                "ticker": ticker,
                "prompt_token_ids": ids,
                "prompt_text": tokenizer.decode(ids),
            }
        )

    return Fixture(
        name="shared_prefix",
        tokenizer_name=getattr(tokenizer, "name", "unknown"),
        tokenizer_is_stub=bool(getattr(tokenizer, "is_stub", False)),
        seed=seed,
        context_tokens=context_tokens,
        n_requests=n_requests,
        requests=requests,
        meta={
            "design": "identical system prompt + identical market context, "
            "per-ticker tail only",
            "intended_prefix_tokens": prefix_len,
            "tail_tokens": tail_tokens,
        },
    )


def build_low_overlap(
    tokenizer: Tokenizer,
    n_requests: int,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    seed: int = DEFAULT_SEED,
) -> Fixture:
    """Distinct synthetic filing text per request, identical exact length."""
    requests: List[Dict[str, Any]] = []
    for i in range(n_requests):
        ticker = _TICKERS[i % len(_TICKERS)]

        def _body(min_words: int, _t=ticker, _i=i) -> str:
            rng = random.Random(seed + 2_000_000 + _i)
            return (
                f"--- FILING EXCERPT {_i} :: {_t} ---\n\n"
                + _paragraphs(rng, min_words)
            )

        ids = _exact_ids(tokenizer, _body, context_tokens)
        requests.append(
            {
                "request_id": f"low_overlap-{i:04d}",
                "ticker": ticker,
                "prompt_token_ids": ids,
                "prompt_text": tokenizer.decode(ids),
            }
        )

    return Fixture(
        name="low_overlap",
        tokenizer_name=getattr(tokenizer, "name", "unknown"),
        tokenizer_is_stub=bool(getattr(tokenizer, "is_stub", False)),
        seed=seed,
        context_tokens=context_tokens,
        n_requests=n_requests,
        requests=requests,
        meta={
            "design": "distinct synthetic filing text per request, "
            "same exact token length",
        },
    )


def build_atl_realistic(
    tokenizer: Tokenizer,
    n_requests: int,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    seed: int = DEFAULT_SEED,
    *,
    static_tokens: Optional[int] = None,
    per_agent_tokens: Optional[int] = None,
    derived_from: Optional[Dict[str, Any]] = None,
) -> Fixture:
    """A third point between ``shared_prefix`` and ``low_overlap``, from measurement.

    The other two fixtures are deliberate BOUNDS: 99.4% and 0.3% common prefix,
    both chosen. This one's composition comes from
    ``analysis/atl_pipeline_audit.py``, which found that a pipeline step prompt
    is assembled as:

      1. a static frame  ("=== SUB-AGENT: …", the output-format header, the
         execution rules) — identical for every agent and every decision;
      2. the step's own configured ``prompt`` / ``outputFormat`` / ``label`` —
         **per agent**, and rewritten daily by post-trade prompt patching;
      3. ``json.dumps(market_snapshot)`` on step 0, or
         ``json.dumps(prior_outputs)`` on later steps — **per decision**.

    The per-decision JSON is what caps prefix sharing: it appears partway into
    the prompt, so nothing after it is shareable no matter how much static text
    follows. Two agents share only segment 1; the same agent across two
    decisions shares segments 1 and 2.

    ``derived_from`` records which audit/extract outputs produced these numbers,
    so the fixture is traceable to a measurement rather than to a guess. Building
    it without that provenance is allowed but flagged in the meta.
    """
    # Defaults are deliberately conservative and OVERRIDDEN by measurement:
    # roughly a tenth static frame, a fifth per-agent configuration, the rest
    # per-decision payload.
    static_n = static_tokens if static_tokens is not None else max(1, context_tokens // 10)
    agent_n = per_agent_tokens if per_agent_tokens is not None else max(1, context_tokens // 5)
    decision_n = context_tokens - static_n - agent_n
    if decision_n <= 0:
        raise ValueError(
            f"static_tokens + per_agent_tokens ({static_n + agent_n}) must be "
            f"less than context_tokens ({context_tokens})"
        )

    def _static(min_words: int) -> str:
        return (
            _SYSTEM_PROMPT
            + "\n\n=== REQUIRED OUTPUT FORMAT ===\n"
            + _paragraphs(random.Random(seed), min_words)
        )

    static_ids = _exact_ids(tokenizer, _static, static_n)

    requests: List[Dict[str, Any]] = []
    for i in range(n_requests):
        ticker = _TICKERS[i % len(_TICKERS)]
        # Per-agent: the configured step prompt. Constant for one agent across
        # decisions, different between agents.
        agent_seed = seed + 3_000_000 + (i % len(_TICKERS))

        def _agent(min_words: int, _s=agent_seed, _t=ticker) -> str:
            return (f"\n\n=== SUB-AGENT: {_t} desk ===\n"
                    + _paragraphs(random.Random(_s), min_words))

        def _decision(min_words: int, _i=i) -> str:
            return ("\n\n=== MARKET SNAPSHOT ===\n"
                    + _paragraphs(random.Random(seed + 4_000_000 + _i), min_words))

        ids = (list(static_ids)
               + _exact_ids(tokenizer, _agent, agent_n)
               + _exact_ids(tokenizer, _decision, decision_n))
        requests.append({
            "request_id": f"atl_realistic-{i:04d}",
            "ticker": ticker,
            "prompt_token_ids": ids,
            "prompt_text": tokenizer.decode(ids),
        })

    return Fixture(
        name="atl_realistic",
        tokenizer_name=getattr(tokenizer, "name", "unknown"),
        tokenizer_is_stub=bool(getattr(tokenizer, "is_stub", False)),
        seed=seed,
        context_tokens=context_tokens,
        n_requests=n_requests,
        requests=requests,
        meta={
            "design": (
                "static frame + per-agent configured prompt + per-decision "
                "market payload, in the order pipeline_runner assembles them"
            ),
            "static_tokens": static_n,
            "per_agent_tokens": agent_n,
            "per_decision_tokens": decision_n,
            "expected_cross_agent_prefix_tokens": static_n,
            "expected_same_agent_prefix_tokens": static_n + agent_n,
            "derived_from": derived_from or {
                "warning": (
                    "NOT derived from a measurement — default proportions were "
                    "used. Pass derived_from with the audit/extract outputs, or "
                    "treat this fixture as another guess."
                )
            },
        },
    )


FIXTURE_BUILDERS = {
    "shared_prefix": build_shared_prefix,
    "low_overlap": build_low_overlap,
    "atl_realistic": build_atl_realistic,
}


def build_fixture(
    name: str,
    tokenizer: Tokenizer,
    n_requests: int,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    seed: int = DEFAULT_SEED,
) -> Fixture:
    if name not in FIXTURE_BUILDERS:
        raise KeyError(f"unknown fixture {name!r}; have {sorted(FIXTURE_BUILDERS)}")
    return FIXTURE_BUILDERS[name](
        tokenizer, n_requests, context_tokens=context_tokens, seed=seed
    )


def save_fixture(fixture: Fixture, out_dir: str) -> Dict[str, str]:
    """Write ``<name>.json`` (full, gitignored) and ``<name>.meta.json`` (committed).

    The full fixture is large — 105 requests x 2620 ids — and regenerable
    exactly from (tokenizer, seed, context_tokens). Only the small meta, which
    carries the sha256, is committed: a regeneration whose hash differs from
    the committed meta is a real signal (tokenizer version drift), and one that
    matches proves the heavy artifact did not need to be stored.
    """
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, f"{fixture.name}.json")
    meta_path = os.path.join(out_dir, f"{fixture.name}.meta.json")

    with open(full_path, "w") as fh:
        json.dump(
            {
                **fixture.canonical_payload(),
                "sha256": fixture.sha256(),
                "requests_text": {
                    r["request_id"]: r.get("prompt_text", "") for r in fixture.requests
                },
            },
            fh,
        )
    with open(meta_path, "w") as fh:
        json.dump(fixture.to_meta(), fh, indent=2)

    return {"full": full_path, "meta": meta_path}


def load_fixture(path: str) -> Fixture:
    with open(path) as fh:
        data = json.load(fh)
    fx = Fixture(
        name=data["name"],
        tokenizer_name=data.get("tokenizer_name", "unknown"),
        tokenizer_is_stub=bool(data.get("tokenizer_is_stub", False)),
        seed=data["seed"],
        context_tokens=data["context_tokens"],
        n_requests=data["n_requests"],
        requests=data["requests"],
    )
    stored = data.get("sha256")
    if stored and stored != fx.sha256():
        raise ValueError(
            f"fixture {path} failed integrity check: stored sha256 {stored} != "
            f"recomputed {fx.sha256()}"
        )
    return fx


def resolve_fixture(
    name: str,
    n_requests: int,
    context_tokens: int,
    seed: int,
    model_id: str,
    fixtures_dir: str,
    use_stub: bool = False,
    rebuild: bool = False,
):
    """Load the fixture from disk, or build it. Shared by every arm.

    Prefers a prebuilt ``fixtures/<name>.json`` so a sweep does not re-tokenize
    on every invocation and — more importantly — so every level of the sweep,
    and every arm of the study, drives a byte-identical prompt set.

    The ``context_tokens`` mismatch check is deliberately fatal: context length
    is a controlled variable, and silently benchmarking 2048-token prompts
    against a config that claims 2620 would corrupt every cross-arm comparison
    downstream while looking perfectly healthy.
    """
    path = os.path.join(fixtures_dir, f"{name}.json")
    if os.path.exists(path) and not rebuild:
        fixture = load_fixture(path)
        if fixture.n_requests < n_requests:
            raise SystemExit(
                f"fixture {path} has {fixture.n_requests} requests, need {n_requests}. "
                f"Rebuild with --rebuild-fixture."
            )
        if fixture.context_tokens != context_tokens:
            raise SystemExit(
                f"fixture {path} has context_tokens={fixture.context_tokens}, config "
                f"says {context_tokens}. Context length is a controlled variable — "
                f"rebuild the fixture or fix the config."
            )
        return fixture, "loaded"

    tokenizer = StubTokenizer() if use_stub else load_hf_tokenizer(model_id)
    fixture = build_fixture(
        name, tokenizer, n_requests, context_tokens=context_tokens, seed=seed
    )
    save_fixture(fixture, fixtures_dir)
    return fixture, "built"


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build benchmark prompt fixtures.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-requests", type=int, default=105)
    ap.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "fixtures"))
    ap.add_argument(
        "--stub-tokenizer",
        action="store_true",
        help="use the offline whitespace tokenizer (testing only; produces "
        "fixtures the runner will refuse for real benchmarks)",
    )
    ap.add_argument("--only", choices=sorted(FIXTURE_BUILDERS), default=None)
    args = ap.parse_args(argv)

    tokenizer = StubTokenizer() if args.stub_tokenizer else load_hf_tokenizer(args.model)
    if args.stub_tokenizer:
        print("!! STUB TOKENIZER — token counts are NOT Qwen counts. Testing only.\n")

    names = [args.only] if args.only else sorted(FIXTURE_BUILDERS)
    for name in names:
        fx = build_fixture(
            name, tokenizer, args.n_requests,
            context_tokens=args.context_tokens, seed=args.seed,
        )
        paths = save_fixture(fx, args.out_dir)
        meta = fx.to_meta()
        if meta["token_counts_all_equal"]:
            counts_desc = f"{meta['token_count_min']} (all equal)"
        else:
            counts_desc = (
                f"{meta['token_count_min']}..{meta['token_count_max']} "
                f"** NOT EQUAL — context length is a controlled variable **"
            )
        print(f"[{name}]")
        print(f"  sha256                {meta['sha256']}")
        print(f"  requests              {meta['n_requests']}")
        print(f"  tokens/request        {counts_desc}")
        print(f"  common prefix tokens  {meta['common_prefix_tokens']}"
              f"  ({meta['common_prefix_fraction'] * 100:.1f}% of context)")
        print(f"  full                  {paths['full']}")
        print(f"  meta                  {paths['meta']}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
