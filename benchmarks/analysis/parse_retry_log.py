"""Reconstruct the killed run's attempt structure from its unfiltered log.

The log is the production retry print, so this is MEASURED evidence of the
mechanism even though the run itself was terminated.
"""
import re, sys, json
from collections import Counter
path = sys.argv[1]
txt = open(path, encoding="utf-8", errors="ignore").read()
decisions = txt.count("Calling LLM for trading decision")
retries = re.findall(r"No text content block in LLM response \(content types: (\[[^\]]*\])\); retry (\d+)/4", txt)
rescues = txt.count("Final rescue call")
by_stage = Counter(int(n) for _c, n in retries)
content_types = Counter(c for c, _n in retries)
print(json.dumps({
  "log": path,
  "decisions_started": decisions,
  "retry_events": len(retries),
  "rescue_events": rescues,
  "extra_attempts": len(retries) + rescues,
  "retries_by_stage": {str(k): v for k, v in sorted(by_stage.items())},
  "observed_content_types_on_retry": dict(content_types),
  "implied_attempts": decisions + len(retries) + rescues,
  "implied_attempts_per_decision": round((decisions + len(retries) + rescues) / decisions, 2) if decisions else None,
}, indent=1))
