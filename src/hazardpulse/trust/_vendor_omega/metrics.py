"""
metrics.py - Section 7: how to MEASURE coherence in ML (the measurement layer).

The blueprint defines coherence operationally; these are those formulas, implemented exactly,
as standalone measurable functions (no magic - each is a named quantity you can audit):

  7.1 representation_tau_c  : the coherence TIME of a hidden-state stream, from its normalized
                             autocorrelation R(Delta); tau_c = the lag where R drops below 1/e.
  7.2 candidate_coherence   : 1/(1+spread) of N answer embeddings - high when candidates agree.
  7.3 evidence_coherence    : E_support * (1 - H_contradiction) * Q_source.
  7.5 entropy_load          : the weighted entropy load (token / candidate / retrieval / tool / drift).
  7.6 useful_info_gain      : U_after - U_before (calibrated-uncertainty reduction, etc).
  7.7 coherence_score       : the composite sigma(...) over the above.

Plus the Sec 2.4-2.6 derived rates, which were DECLARED as CoherenceState fields but never
computed (an honesty gap the audit caught - those dead fields are removed, and the real
operational forms live here):
  2.4 snr_coh               : dI / sigma_phi(tau_c)        (coherence signal-to-noise)
  2.6 identity_renewal_rate : info_efficiency / tau_c

These need a STREAM / a candidate set / before-after info, which the per-sample control loop does
not have - so they live here, to be measured where that context exists (a sequence of hidden
states, a set of candidate answers, a before/after task). Numpy-only.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "representation_tau_c", "candidate_coherence", "evidence_coherence", "entropy_load",
    "useful_info_gain", "snr_coh", "info_efficiency", "identity_renewal_rate", "coherence_score",
]

_E_INV = float(np.exp(-1.0))     # the 1/e threshold for the coherence-time crossing


def representation_tau_c(hidden_states, max_lag=None) -> float:
    """Sec 7.1: coherence time of a hidden-state stream. R(Delta) = mean_t cos(h_t, h_{t+Delta});
    tau_c = the smallest lag where R(Delta) < 1/e (linearly interpolated between integer lags). If
    R never crosses 1/e within max_lag, fit R ~ exp(-Delta/tau_c) over the measured lags. Returns
    +inf if the stream is perfectly coherent over the whole window."""
    H = np.asarray(hidden_states, float)
    if H.ndim != 2 or len(H) < 2:
        raise ValueError("hidden_states must be (T, d) with T >= 2")
    T = len(H)
    max_lag = (T - 1) if max_lag is None else min(int(max_lag), T - 1)
    norms = np.linalg.norm(H, axis=1) + 1e-12
    Hn = H / norms[:, None]
    R = np.empty(max_lag + 1)
    for dlag in range(max_lag + 1):
        dots = np.sum(Hn[: T - dlag] * Hn[dlag:], axis=1)   # cos similarity at each t
        R[dlag] = float(dots.mean())
    # first crossing of 1/e, linearly interpolated
    for dlag in range(1, max_lag + 1):
        if R[dlag] < _E_INV:
            r0, r1 = R[dlag - 1], R[dlag]
            frac = (r0 - _E_INV) / (r0 - r1 + 1e-12) if r0 != r1 else 0.0
            return float((dlag - 1) + np.clip(frac, 0.0, 1.0))
    # no crossing: fit R ~ exp(-Delta/tau_c) over the positive part of the curve
    lags = np.arange(max_lag + 1)
    pos = R > 1e-6
    if pos.sum() >= 2:
        slope = np.polyfit(lags[pos], np.log(R[pos]), 1)[0]
        if slope < -1e-9:
            return float(-1.0 / slope)
    return float("inf")


def candidate_coherence(embeddings) -> float:
    """Sec 7.2: C_A = 1/(1 + sigma_A), sigma_A = mean ||e_i - e_bar||^2 of N candidate-answer
    embeddings. ->1 when candidates agree (low spread), ->0 when they diverge."""
    E = np.asarray(embeddings, float)
    if E.ndim != 2 or len(E) == 0:
        raise ValueError("embeddings must be (N, d) with N >= 1")
    sigma_a = float(np.mean(np.sum((E - E.mean(0)) ** 2, axis=1)))
    return 1.0 / (1.0 + sigma_a)


def evidence_coherence(support, contradiction, source_quality=1.0) -> float:
    """Sec 7.3: C_evidence = E_support * (1 - H_contradiction) * Q_source, each clamped to [0,1]."""
    s = float(np.clip(support, 0.0, 1.0))
    h = float(np.clip(contradiction, 0.0, 1.0))
    q = float(np.clip(source_quality, 0.0, 1.0))
    return s * (1.0 - h) * q


def entropy_load(token_h=0.0, candidate_spread=0.0, retrieval_h=0.0, tool_h=0.0,
                 memory_drift=0.0, weights=(0.3, 0.2, 0.2, 0.15, 0.15)) -> float:
    """Sec 7.5: weighted entropy load = w.[token H, candidate spread, retrieval H, tool H, memory
    drift]. Higher = more disorder the engine is carrying (drives the control plane to slow down)."""
    w = np.asarray(weights, float)
    x = np.array([token_h, candidate_spread, retrieval_h, tool_h, memory_drift], float)
    return float(np.dot(w, x))


def useful_info_gain(u_before, u_after) -> float:
    """Sec 7.6: Delta I_useful = U_after - U_before (e.g. calibrated-uncertainty reduction,
    benchmark improvement, verified claims added). Positive = the step was informative."""
    return float(u_after) - float(u_before)


def info_efficiency(delta_info, delta_entropy) -> float:
    """Sec 2.5: eta = dI / dS - useful information gained per unit entropy spent. Guarded so a
    near-zero entropy change does not explode; sign of dS preserved (spending entropy for info)."""
    ds = float(delta_entropy)
    denom = abs(ds) + 1e-9
    return float(delta_info) / denom * (1.0 if ds >= 0 else -1.0)


def snr_coh(delta_info, phase_noise) -> float:
    """Sec 2.4: SNR_coh = dI / sigma_phi(tau_c) - coherence signal-to-noise (information signal
    over the phase noise at the coherence time). Higher = a cleaner, more coherent signal."""
    return float(delta_info) / (abs(float(phase_noise)) + 1e-9)


def identity_renewal_rate(eta, tau_c) -> float:
    """Sec 2.6: I_r = eta / tau_c - identity renewal rate (efficiency per unit coherence time)."""
    return float(eta) / (abs(float(tau_c)) + 1e-9)


def coherence_score(*, candidate=0.0, evidence=0.0, memory_residual=0.0, tau_c=1.0,
                    snr=0.0, entropy=0.0, risk=0.0,
                    weights=(1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0)) -> float:
    """Sec 7.7: the composite coherence score, sigmoid of
        w1 C_A + w2 C_evidence + w3 (1 - R_I) + w4 log(1+tau_c) + w5 SNR - w6 S_load - w7 R_risk.
    A single [0,1] health number; "should be calibrated empirically" - this is the un-tuned form."""
    w = weights
    tau_term = np.log1p(max(float(tau_c), 0.0)) if np.isfinite(tau_c) else np.log1p(1e6)
    z = (w[0] * candidate + w[1] * evidence + w[2] * (1.0 - memory_residual)
         + w[3] * tau_term + w[4] * snr - w[5] * entropy - w[6] * risk)
    return float(1.0 / (1.0 + np.exp(-z)))
