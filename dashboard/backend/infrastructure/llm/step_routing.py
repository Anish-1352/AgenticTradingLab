"""Per-step model routing for the decision pipeline.

Cost per call spans 193x across the seven measured models ($0.000447 for
Nemotron to $0.086262 for GPT-5.5), so sending mechanical steps to a cheap
model is the largest single cost lever available. This module decides WHICH
model each step uses. It does not call anything.

ROUTING IS OPT-IN AND THE DEFAULT IS EXACTLY CURRENT BEHAVIOUR
---------------------------------------------------------------
``RoutingConfig.disabled()`` — and an absent, empty, or ``enabled: false``
config — resolves every step to the agent's configured model, which is what
``run_pipeline_decision`` already did. There is no code path where loading a
default config changes what a run does.

THE FAILURE MODE THAT DECIDES WHETHER THIS IS USABLE
-----------------------------------------------------
``pipeline_runner`` aborts the ENTIRE decision when any step returns
unparseable JSON (``pipeline_runner.py:479``). There is no retry on that path.
Downstream, ``portfolio_manager`` either raises ``LLMDecisionError`` under
``strict_llm`` or silently falls back to rule-based trading — so on the
leaderboard a parse failure does not degrade the decision, it replaces it with
a non-LLM one.

The final step carries a SECOND gate: ``pipeline_output_to_decision`` returns
None unless the parsed dict has a non-empty ``actions``, ``orders`` or
``risk_actions`` list, and that also aborts. A model can therefore parse
perfectly and still fail the last step.

So the failure mode of routing a step to a weak model is not "slightly worse
extraction" — it is a decision that never completes. Measure parse reliability
per model per step before trusting any cost projection; see
``dashboard/scripts/measure_step_reliability.py``.

HOW A STEP IS IDENTIFIED — TAKEN FROM THE PIPELINE, NOT INVENTED
------------------------------------------------------------------
Pipeline steps are dicts. The real ones carry ``id``, ``label``, ``prompt`` and
``outputFormat``; UI presets additionally carry ``presetKey``, which
``pipeline_runner`` already treats as identity when applying post-trade
patches. All four keys below already exist in the data:

===========  ==============================================================
``id``       exact step in one specific pipeline (``"bench-3-1"``). Most
             specific; does not generalise across pipelines.
``presetKey``the step's preset role. Absent on hand-written pipelines.
``label``    the step's ROLE (``"Risk Assessment"``). Shared across
             pipelines — the 3-step and 5-step configs both use
             ``"Technical Read"`` — which makes it the useful routing key
             and the one a human writes by hand.
``index``    1-based position. Weakest: "step 3" is the decision step in a
             3-step pipeline and the middle of a 5-step one.
===========  ==============================================================

Resolution tries them in that order and stops at the first match, so a config
can say "every Risk Assessment step goes to Nemotron, except this one".
Matching is recorded on the result (``matched_by``/``matched_key``) so a run
can be audited without re-deriving the decision.

Labels are user-editable free text, which is why they are matched
case-insensitively on stripped text, and why ``id`` outranks them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "StepRoute",
    "RoutingConfig",
    "RouteResolution",
    "MATCH_PRECEDENCE",
    "load_routing_config",
]

# Most specific first. Every name here is a key the pipeline already carries.
MATCH_PRECEDENCE: Tuple[str, ...] = ("id", "presetKey", "label", "index")


def _norm_label(value: Any) -> Optional[str]:
    """Labels are free text a user typed; compare on stripped lowercase."""
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() if text else None


@dataclass(frozen=True)
class StepRoute:
    """One routing target: which model, and optionally on which integration.

    ``integration`` is carried because the cheap model may live behind a
    different provider than the expensive one — the whole point is to mix an
    OpenRouter Nemotron with a CommonStack Claude. ``None`` means "the client
    the pipeline was already given", which is what makes a single-provider
    config work without mentioning providers at all.

    A self-hosted engine is expressed as an OpenAI/Anthropic-compatible
    endpoint: set ``integration="openrouter"`` and point ``OPENROUTER_BASE_URL``
    at it, which is the existing override in ``providers/openrouter.py``. No
    new provider type is introduced here.
    """

    model: str
    integration: Optional[str] = None
    note: Optional[str] = None

    @staticmethod
    def coerce(value: Any) -> "StepRoute":
        """Accept ``"model-id"`` or ``{"model": ..., "integration": ...}``.

        The bare-string form is the one a human writes; the dict form is for
        cross-provider routing. Both are supported because forcing the dict
        form on the common case makes the config harder to read for no gain.
        """
        if isinstance(value, StepRoute):
            return value
        if isinstance(value, str):
            model = value.strip()
            if not model:
                raise ValueError("route model id is empty")
            return StepRoute(model=model)
        if isinstance(value, Mapping):
            model = str(value.get("model") or value.get("model_id") or "").strip()
            if not model:
                raise ValueError(f"route is missing a model id: {value!r}")
            integration = value.get("integration")
            return StepRoute(
                model=model,
                integration=str(integration).strip() or None if integration else None,
                note=value.get("note"),
            )
        raise TypeError(f"route must be a string or mapping, got {type(value).__name__}")


@dataclass(frozen=True)
class RouteResolution:
    """What a step resolved to, and why.

    ``matched_by`` is ``None`` when nothing in the config matched and the
    agent's own model was used. That is the case worth being able to see: a
    config that silently matches nothing looks identical to no config at all
    in the cost numbers, and identical to a working one in the logs.
    """

    model: str
    integration: Optional[str]
    matched_by: Optional[str]
    matched_key: Optional[str]
    is_default: bool

    @property
    def routed(self) -> bool:
        """True when the config chose this model rather than the agent's."""
        return self.matched_by is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "integration": self.integration,
            "matched_by": self.matched_by,
            "matched_key": self.matched_key,
            "is_default": self.is_default,
            "routed": self.routed,
        }


@dataclass(frozen=True)
class RoutingConfig:
    """A resolved routing table. Immutable; build it with ``from_dict``.

    ``enabled=False`` is preserved as a distinct state rather than collapsed to
    an empty table, so a config can be written, committed and reviewed while
    switched off — and so turning it on is a one-word diff.
    """

    enabled: bool = False
    by_id: Mapping[str, StepRoute] = field(default_factory=dict)
    by_preset: Mapping[str, StepRoute] = field(default_factory=dict)
    by_label: Mapping[str, StepRoute] = field(default_factory=dict)
    by_index: Mapping[int, StepRoute] = field(default_factory=dict)
    fallback: Optional[StepRoute] = None

    # ---------------------------------------------------------------- build --

    @staticmethod
    def disabled() -> "RoutingConfig":
        """Current behaviour, explicitly. Every step uses the agent's model."""
        return RoutingConfig(enabled=False)

    @staticmethod
    def from_dict(raw: Optional[Mapping[str, Any]]) -> "RoutingConfig":
        """Parse a routing config, or return the disabled one.

        Raises on a malformed config rather than silently ignoring it: a typo
        in a routing key would otherwise show up as an unexplained cost figure
        weeks later, and the whole point of this module is that the model a
        step ran on is never a mystery.
        """
        if not raw:
            return RoutingConfig.disabled()
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"step routing config must be a mapping, got {type(raw).__name__}")

        # JSON has no comment syntax, so an underscore-prefixed key is the
        # conventional stand-in. Ignoring those keeps the strict check on
        # everything that could actually be a typo'd routing directive.
        unknown = {k for k in raw if not str(k).startswith("_")} - {
            "enabled", "steps", "fallback", "default", "note", "notes"}
        if unknown:
            raise ValueError(
                f"unknown step-routing keys: {sorted(unknown)}. "
                f"Expected: enabled, steps, fallback, note.")

        by_id: Dict[str, StepRoute] = {}
        by_preset: Dict[str, StepRoute] = {}
        by_label: Dict[str, StepRoute] = {}
        by_index: Dict[int, StepRoute] = {}

        steps = raw.get("steps") or {}
        if steps and not isinstance(steps, Mapping):
            raise TypeError("step routing 'steps' must be a mapping")

        for key, value in steps.items():
            route = StepRoute.coerce(value)
            kind, parsed = _classify_key(key)
            if kind == "index":
                by_index[parsed] = route
            elif kind == "presetKey":
                by_preset[parsed] = route
            elif kind == "id":
                by_id[parsed] = route
            else:
                by_label[parsed] = route

        fallback_raw = raw.get("fallback", raw.get("default"))
        fallback = StepRoute.coerce(fallback_raw) if fallback_raw else None

        return RoutingConfig(
            enabled=bool(raw.get("enabled", False)),
            by_id=by_id,
            by_preset=by_preset,
            by_label=by_label,
            by_index=by_index,
            fallback=fallback,
        )

    # -------------------------------------------------------------- resolve --

    def resolve(
        self,
        step: Optional[Mapping[str, Any]],
        index: int,
        default_model: str,
        default_integration: Optional[str] = None,
    ) -> RouteResolution:
        """Pick the model for one step. ``index`` is 0-based, as the runner has it.

        When routing is disabled, or nothing matches and there is no fallback,
        this returns ``default_model`` with ``matched_by=None`` — byte-identical
        behaviour to the unrouted runner.
        """
        if not self.enabled:
            return RouteResolution(
                model=default_model, integration=default_integration,
                matched_by=None, matched_key=None, is_default=True)

        step = step if isinstance(step, Mapping) else {}

        for kind in MATCH_PRECEDENCE:
            if kind == "id":
                raw_key = step.get("id")
                table, lookup = self.by_id, (
                    str(raw_key).strip() if raw_key is not None else None)
            elif kind == "presetKey":
                raw_key = step.get("presetKey")
                table, lookup = self.by_preset, (
                    str(raw_key).strip() if raw_key is not None else None)
            elif kind == "label":
                raw_key = step.get("label")
                table, lookup = self.by_label, _norm_label(raw_key)
            else:
                raw_key = index + 1          # config is 1-based, like the UI
                table, lookup = self.by_index, index + 1

            if lookup is not None and lookup in table:
                route = table[lookup]
                return RouteResolution(
                    model=route.model,
                    integration=route.integration or default_integration,
                    matched_by=kind,
                    matched_key=str(raw_key),
                    is_default=False)

        if self.fallback is not None:
            return RouteResolution(
                model=self.fallback.model,
                integration=self.fallback.integration or default_integration,
                matched_by="fallback",
                matched_key=None,
                is_default=False)

        return RouteResolution(
            model=default_model, integration=default_integration,
            matched_by=None, matched_key=None, is_default=True)

    def plan(
        self,
        steps: List[Mapping[str, Any]],
        default_model: str,
        default_integration: Optional[str] = None,
    ) -> List[RouteResolution]:
        """Resolve a whole pipeline without running it — the dry-run surface."""
        return [
            self.resolve(step, i, default_model, default_integration)
            for i, step in enumerate(steps or [])
        ]

    def describe(self) -> Dict[str, Any]:
        """Serialisable summary, for manifests and run records."""
        return {
            "enabled": self.enabled,
            "n_rules": (len(self.by_id) + len(self.by_preset)
                        + len(self.by_label) + len(self.by_index)),
            "by_id": {k: v.model for k, v in self.by_id.items()},
            "by_presetKey": {k: v.model for k, v in self.by_preset.items()},
            "by_label": {k: v.model for k, v in self.by_label.items()},
            "by_index": {k: v.model for k, v in self.by_index.items()},
            "fallback": self.fallback.model if self.fallback else None,
        }


def _classify_key(key: Any) -> Tuple[str, Any]:
    """Work out which identifier a config key is addressing.

    Explicit prefixes (``id:``, ``preset:``, ``label:``, ``step:``) always win,
    because a pipeline is free to label a step ``"3"`` and a config must be able
    to say which of the two it meant. Without a prefix, a bare integer is a
    step index and anything else is a label — matching how someone writes this
    by hand.
    """
    if isinstance(key, int):
        return "index", key
    text = str(key).strip()
    if not text:
        raise ValueError("step routing key is empty")

    for prefix, kind in (("id:", "id"), ("preset:", "presetKey"),
                         ("presetkey:", "presetKey"), ("label:", "label"),
                         ("step:", "index"), ("index:", "index")):
        if text.lower().startswith(prefix):
            rest = text[len(prefix):].strip()
            if not rest:
                raise ValueError(f"step routing key {key!r} has an empty value")
            if kind == "index":
                try:
                    return "index", int(rest)
                except ValueError:
                    raise ValueError(
                        f"step routing key {key!r} is not a step number") from None
            return kind, (_norm_label(rest) if kind == "label" else rest)

    if text.isdigit():
        return "index", int(text)
    return "label", _norm_label(text)


def load_routing_config(
    source: Any = None,
    *,
    env_var: str = "ATL_STEP_ROUTING",
) -> RoutingConfig:
    """Build a config from a dict, a JSON file path, or the environment.

    Precedence: an explicit ``source`` argument, then ``$ATL_STEP_ROUTING``
    (a path or inline JSON), then disabled. The env var exists so the
    reliability harness and a backtest can be pointed at the same file without
    editing agent config, NOT as a way to switch production behaviour by
    accident — it is read only where a caller asks for it.
    """
    if source is None:
        source = os.getenv(env_var) or None
    if source is None:
        return RoutingConfig.disabled()
    if isinstance(source, RoutingConfig):
        return source
    if isinstance(source, Mapping):
        return RoutingConfig.from_dict(source)

    text = str(source).strip()
    if not text:
        return RoutingConfig.disabled()
    if text.startswith("{"):
        return RoutingConfig.from_dict(json.loads(text))
    with open(text, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    # Accept both a bare routing object and one nested under a conventional key.
    if isinstance(payload, Mapping) and "step_routing" in payload:
        payload = payload["step_routing"]
    return RoutingConfig.from_dict(payload)
