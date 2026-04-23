"""CME → tropical cyclone intensification correlation.

  - **run_cme_hurricane_intensification** — Thakur+ / Prager+ disputed
    hypothesis that interplanetary CME passage correlates with
    rapid intensification in tropical cyclones.

Implementation: for each RI event time, count the number of solar
wind ``sw_speed_max_72h > 600 km/s`` flags (proxy for CME shock
arrivals at L1) in the 72h window before RI. Compare against
matched no-RI control cases.

Falsified if the difference is statistically zero. The control
result is itself publishable (the literature is contested at small
N — a clean null result on a population would constrain the
hypothesis).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass

import numpy as np

from hazardpulse.data.space_weather import space_weather_features_for_window


CME_SPEED_THRESHOLD_KMS = 600.0
SUSTAINED_NEG_BZ_THRESHOLD_NT = -8.0


@dataclass
class CmeHurricaneResult:
    name: str = "cme_hurricane_intensification"
    n_ri_events: int = 0
    n_no_ri_events: int = 0
    cme_marker_pre_ri_mean: float = float("nan")
    cme_marker_no_ri_mean: float = float("nan")
    delta_mean: float = float("nan")
    welch_t: float = float("nan")
    welch_p: float = float("nan")
    bootstrap_ci_lo: float = float("nan")
    bootstrap_ci_hi: float = float("nan")
    notes: str = (
        "Thakur+/Prager+ CME-RI hypothesis. Counts (sustained sw_speed > "
        "600 km/s) AND (sw_bz < -8 nT) in 72h pre-RI vs matched controls."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def _cme_marker(event_time: dt.datetime) -> float:
    """Return 1.0 if a CME-like signature occurred in 72h pre-event, else 0.0.

    Definition: sw_speed_max > threshold AND sw_bz_min < neg threshold
    in the 72h window. Both conditions must hold.
    """
    feats = space_weather_features_for_window(event_time, window_h=72)
    speed_ok = (
        feats.get("sw_speed_max_72h") is not None
        and not np.isnan(feats.get("sw_speed_max_72h", float("nan")))
        and feats["sw_speed_max_72h"] > CME_SPEED_THRESHOLD_KMS
    )
    bz_ok = (
        feats.get("sw_bz_min_72h") is not None
        and not np.isnan(feats.get("sw_bz_min_72h", float("nan")))
        and feats["sw_bz_min_72h"] < SUSTAINED_NEG_BZ_THRESHOLD_NT
    )
    return 1.0 if (speed_ok and bz_ok) else 0.0


def _welch(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    se = math.sqrt(va / len(a) + vb / len(b))
    if se <= 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    try:
        from scipy.stats import t as _t
        df_num = (va / len(a) + vb / len(b)) ** 2
        df_den = (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
        df = df_num / df_den if df_den > 0 else 1.0
        p = 2.0 * (1.0 - _t.cdf(abs(t), df))
    except ImportError:
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p)


def _boot_ci(a, b, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        deltas[i] = ai.mean() - bi.mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_cme_hurricane_intensification(
    ri_event_times: list[dt.datetime],
    no_ri_event_times: list[dt.datetime],
) -> CmeHurricaneResult:
    """Test CME-RI coupling hypothesis."""
    a = np.array([_cme_marker(t) for t in ri_event_times])
    b = np.array([_cme_marker(t) for t in no_ri_event_times])
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    t, p = _welch(a, b)
    lo, hi = _boot_ci(a, b)
    return CmeHurricaneResult(
        n_ri_events=int(len(a_clean)),
        n_no_ri_events=int(len(b_clean)),
        cme_marker_pre_ri_mean=float(np.mean(a_clean)) if a_clean.size else float("nan"),
        cme_marker_no_ri_mean=float(np.mean(b_clean)) if b_clean.size else float("nan"),
        delta_mean=float(np.mean(a_clean) - np.mean(b_clean))
            if a_clean.size and b_clean.size else float("nan"),
        welch_t=t, welch_p=p,
        bootstrap_ci_lo=lo, bootstrap_ci_hi=hi,
    )
