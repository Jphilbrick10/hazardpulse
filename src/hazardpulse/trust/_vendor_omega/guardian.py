"""
guardian.py - Module 12: the Coherence Immune System.

A measurable SAFETY WRAPPER around the engine. It does NOT add intelligence - it turns the
coherence signals the engine already produces (abstention, drift phase, contradiction, the
fixed-point operating domain, evidence) into ENFORCED policy + an audit trail, and refuses /
degrades / quarantines before an internal collapse mode becomes an output failure.

What it actually catches (honest - these are the failure modes the signals genuinely expose,
not a universal hallucination detector):
  - out-of-domain / overflow inputs  -> PREFLIGHT rejects them before the engine wraps (composes
    with basin_fp.cl's source-level overflow abstain).
  - overconfidence vs evidence       -> a CONFIDENT answer with low external evidence is the
    blueprint's hallucination signature (high fluency, low evidence) -> refuse the high-confidence
    answer (cap by the evidence ceiling).
  - memory poisoning / contradiction -> a high contradiction rate (or a CONFLICTED protected
    core) drives the gateway toward REPAIR -> quarantine + refuse.
  - identity drift / model collapse  -> the drift "weather" (Unstable / Fragmenting) drives
    CAUTIOUS / DEGRADED -> tighten or refuse.

The gateway mode (NORMAL / CAUTIOUS / DEGRADED / REPAIR) is the coherence_lang NPE-gateway
pattern, adapted as our own and made standalone (numpy only). Every guard action is loggable
to the audit ledger.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._guards import finite_or


class GatewayMode(str, Enum):
    NORMAL = "NORMAL"       # healthy: answer normally
    CAUTIOUS = "CAUTIOUS"   # mild drift/uncertainty: tighten, prefer abstaining on the margin
    DEGRADED = "DEGRADED"   # unhealthy: answer only the clearly-confident, refuse the rest
    REPAIR = "REPAIR"       # collapse risk: refuse all answers, quarantine, await consolidation


_DRIFT_HEALTH = {"Stable": 1.0, "Resonating": 0.7, "Unstable": 0.4, "Fragmenting": 0.1}


@dataclass
class GuardVerdict:
    label: object                # the FINAL decision (None == refused / abstained)
    action: str                  # what the guardian did (pass / refuse:* / reject:* / cap:*)
    mode: GatewayMode            # the gateway health mode at decision time
    coherence_score: float       # the composite system-health score in [0,1]
    original: object             # the engine's pre-guard decision (for the audit trail)


class CoherenceGuardian:
    """The coherence immune system. `preflight(x)` validates an input; `health()` maps the
    rolling coherence signals to a GatewayMode; `guard(label, state, ...)` applies the repair
    policy and returns a GuardVerdict. Stateful: it tracks a rolling window of recent decisions
    to estimate the abstention rate, and a quarantine list."""

    def __init__(self, *, ledger=None, max_input=1.0e7,
                 overconfidence_evidence_min=0.25, window=64, min_health_samples=8,
                 normal_above=0.85, cautious_above=0.6, degraded_above=0.3):
        self.ledger = ledger
        self.max_input = float(max_input)
        self.overconfidence_evidence_min = float(overconfidence_evidence_min)
        self.window = int(window)
        self.min_health_samples = int(min_health_samples)
        self._normal, self._cautious, self._degraded = normal_above, cautious_above, degraded_above
        self._recent_abstain: deque = deque(maxlen=self.window)
        self.quarantined: list = []
        self.mode = GatewayMode.NORMAL
        self.coherence_score = 1.0

    def _log(self, op, payload):
        if self.ledger is not None:
            self.ledger.append({"op": op, **payload})

    # ----------------------------------------------------------------- preflight
    def preflight(self, x):
        """Validate an input BEFORE the engine sees it. Returns (ok, reason). Rejects non-finite
        values and magnitudes outside the fixed-point operating domain (which would overflow /
        wrap) - failing fast is safer than relying only on the downstream abstain."""
        try:
            xa = np.asarray(x, float)
        except Exception:
            return False, "uncoercible-input"
        if not np.all(np.isfinite(xa)):
            return False, "non-finite-input"
        if xa.size and float(np.abs(xa).max()) > self.max_input:
            return False, "out-of-domain-magnitude"
        return True, "ok"

    # -------------------------------------------------------------------- health
    def health(self, *, drift_phase="Stable", contradiction_rate=0.0):
        """Map the rolling coherence signals to a composite health score + GatewayMode:
        answering health (1 - abstain rate), drift 'weather', and contradiction load."""
        # The rolling abstain-rate only informs health once we have enough samples - a single
        # rejection must NOT panic the gateway into CAUTIOUS (small-sample noise).
        if len(self._recent_abstain) >= self.min_health_samples:
            abstain_rate = sum(self._recent_abstain) / len(self._recent_abstain)
            answering = 1.0 - min(max(abstain_rate, 0.0), 1.0)
        else:
            answering = 1.0
        drift = _DRIFT_HEALTH.get(str(drift_phase), 0.5)
        # a NON-FINITE contradiction signal cannot be trusted -> treat as MAXIMAL contradiction
        # (1.0), which drives REPAIR. Fail-closed: a NaN must never dilute to a healthy score.
        cr = min(max(finite_or(contradiction_rate, 1.0), 0.0), 1.0)
        score = 0.4 * answering + 0.35 * drift + 0.25 * (1.0 - cr)
        if score >= self._normal:
            smooth = GatewayMode.NORMAL
        elif score >= self._cautious:
            smooth = GatewayMode.CAUTIOUS
        elif score >= self._degraded:
            smooth = GatewayMode.DEGRADED
        else:
            smooth = GatewayMode.REPAIR
        # HARD safety FLOORS: a single severe signal forces a worse mode than the smooth
        # average would - the average must not be able to dilute a real collapse signal.
        dp = str(drift_phase)
        floor = GatewayMode.NORMAL
        if dp == "Unstable" or cr >= 0.3:
            floor = GatewayMode.CAUTIOUS
        if dp == "Fragmenting" or cr >= 0.5:
            floor = GatewayMode.DEGRADED
        if (dp == "Fragmenting" and cr >= 0.3) or cr >= 0.8:
            floor = GatewayMode.REPAIR
        order = [GatewayMode.NORMAL, GatewayMode.CAUTIOUS, GatewayMode.DEGRADED, GatewayMode.REPAIR]
        mode = max(smooth, floor, key=order.index)        # the WORSE (more severe) of the two
        self.coherence_score, self.mode = float(score), mode
        return mode

    # --------------------------------------------------------------------- guard
    def guard(self, label, state, *, x=None, evidence_score=None,
              drift_phase="Stable", contradiction_rate=0.0) -> GuardVerdict:
        """Apply the immune policy to one engine decision. Order: preflight (reject bad inputs)
        -> health gate -> repair. Returns a GuardVerdict; label=None means the guardian refused."""
        mode = self.health(drift_phase=drift_phase, contradiction_rate=contradiction_rate)
        original = label

        # 1. preflight: a malformed / out-of-domain input is rejected outright
        if x is not None:
            ok, reason = self.preflight(x)
            if not ok:
                self._record(abstained=True)
                v = GuardVerdict(None, f"reject:{reason}", mode, self.coherence_score, original)
                self._log("guardian_reject", {"reason": reason, "mode": mode.value})
                return v

        st_mode = getattr(state, "mode", None)

        # 2. REPAIR: collapse risk - refuse every answer, await consolidation
        if mode == GatewayMode.REPAIR:
            self._record(abstained=True)
            self._log("guardian_refuse", {"action": "repair", "mode": mode.value,
                                          "score": round(self.coherence_score, 4)})
            return GuardVerdict(None, "refuse:repair", mode, self.coherence_score, original)

        # 3. overconfidence cap (the hallucination signature: confident answer, low evidence).
        # A PROVIDED-but-non-finite evidence_score is treated as 0.0 (the least-trusted value), so
        # a NaN can never slip past `< min` and let an unverified confident answer through.
        if st_mode == "confident" and evidence_score is not None:
            ev = finite_or(evidence_score, 0.0)
            if ev < self.overconfidence_evidence_min:
                self._record(abstained=True)
                self._log("guardian_cap", {"action": "overconfidence", "evidence": round(ev, 4)})
                return GuardVerdict(None, "cap:overconfidence", mode, self.coherence_score, original)

        # 4. DEGRADED: answer only the clearly-confident; refuse uncertain
        if mode == GatewayMode.DEGRADED and st_mode != "confident":
            self._record(abstained=True)
            self._log("guardian_refuse", {"action": "degraded", "mode": mode.value})
            return GuardVerdict(None, "refuse:degraded", mode, self.coherence_score, original)

        # 5. pass through (the engine's own abstention is honored: label may already be None)
        self._record(abstained=(label is None))
        return GuardVerdict(label, "pass", mode, self.coherence_score, original)

    def quarantine(self, item, reason):
        """Hold a suspicious input / claim out of the decision path (memory-poisoning defense)."""
        self.quarantined.append({"item": item, "reason": reason})
        self._log("guardian_quarantine", {"reason": reason, "n_quarantined": len(self.quarantined)})

    def _record(self, *, abstained):
        self._recent_abstain.append(1 if abstained else 0)

    def stats(self):
        ar = (sum(self._recent_abstain) / len(self._recent_abstain)) if self._recent_abstain else 0.0
        return {"mode": self.mode.value, "coherence_score": round(self.coherence_score, 4),
                "abstain_rate": round(ar, 4), "quarantined": len(self.quarantined)}
