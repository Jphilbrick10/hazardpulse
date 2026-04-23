"""GLM lightning ↔ severe-weather correlation analyses.

  - **run_hurricane_lightning_correlation** — Fierro+ 2014: GLM
    flash rate inside the storm core correlates with rapid
    intensification (RI). Tests pre-RI flash rate elevation.

  - **run_tornado_lightning_leadup** — Steiger+ 2007 / Schultz+ 2011:
    flash-rate jumps precede EF2+ tornado formation by 15-45 minutes.
    Tests jump_ratio (15min/60min flash rate) pre-tornado vs control.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import numpy as np

from hazardpulse.data.glm_lightning import compute_block_l


@dataclass
class HurricaneLightningResult:
    name: str = "hurricane_lightning_correlation"
    n_ri_events: int = 0
    n_no_ri_events: int = 0
    flash_rate_pre_ri_mean: float = float("nan")
    flash_rate_no_ri_mean: float = float("nan")
    delta_mean: float = float("nan")
    welch_t: float = float("nan")
    welch_p: float = float("nan")
    bootstrap_ci_lo: float = float("nan")
    bootstrap_ci_hi: float = float("nan")
    notes: str = "Fierro+2014 GLM-RI correlation."

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TornadoLightningResult:
    name: str = "tornado_lightning_leadup"
    n_tornado_events: int = 0
    n_null_events: int = 0
    jump_ratio_pre_tornado_mean: float = float("nan")
    jump_ratio_null_mean: float = float("nan")
    delta_mean: float = float("nan")
    welch_t: float = float("nan")
    welch_p: float = float("nan")
    bootstrap_ci_lo: float = float("nan")
    bootstrap_ci_hi: float = float("nan")
    notes: str = "Steiger+2007 / Schultz+2011 flash-jump pre-tornado."

    def to_dict(self) -> dict:
        return asdict(self)


def _welch(a, b):
    import math
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


def run_hurricane_lightning_correlation(
    ri_events: list[tuple[dt.datetime, float, float]],
    no_ri_events: list[tuple[dt.datetime, float, float]],
    *,
    window_h: float = 6.0,
    bbox_halfwidth_deg: float = 2.0,
) -> HurricaneLightningResult:
    """Compare GLM flash rate inside storm core for RI vs no-RI cases.

    ``ri_events`` and ``no_ri_events`` are lists of
    (event_time, lat, lon) for the storm at start of intensification
    window. ``event_time`` should be the moment RI was first observed
    (or the matched no-RI reference).
    """
    rates_ri = []
    rates_no = []
    for (t, lat, lon) in ri_events:
        feats = compute_block_l(t, lat, lon, bbox_halfwidth_deg=bbox_halfwidth_deg)
        rates_ri.append(feats.get("ltg_flash_rate_per_min", float("nan")))
    for (t, lat, lon) in no_ri_events:
        feats = compute_block_l(t, lat, lon, bbox_halfwidth_deg=bbox_halfwidth_deg)
        rates_no.append(feats.get("ltg_flash_rate_per_min", float("nan")))

    a = np.array(rates_ri)
    b = np.array(rates_no)
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    t, p = _welch(a, b)
    lo, hi = _boot_ci(a, b)
    return HurricaneLightningResult(
        n_ri_events=int(len(a_clean)),
        n_no_ri_events=int(len(b_clean)),
        flash_rate_pre_ri_mean=float(np.mean(a_clean)) if a_clean.size else float("nan"),
        flash_rate_no_ri_mean=float(np.mean(b_clean)) if b_clean.size else float("nan"),
        delta_mean=float(np.mean(a_clean) - np.mean(b_clean))
            if a_clean.size and b_clean.size else float("nan"),
        welch_t=t, welch_p=p,
        bootstrap_ci_lo=lo, bootstrap_ci_hi=hi,
    )


def run_tornado_lightning_leadup(
    tornado_events: list[tuple[dt.datetime, float, float]],
    null_events: list[tuple[dt.datetime, float, float]],
    *,
    bbox_halfwidth_deg: float = 1.0,
) -> TornadoLightningResult:
    """Compare flash-jump ratio in 15-45 min pre-tornado vs null cases.

    ``tornado_events`` are (formation_time, lat, lon).
    ``null_events`` are matched (no-tornado-occurred) cases at
    similar storm intensity.
    """
    jumps_t = []
    jumps_n = []
    for (t, lat, lon) in tornado_events:
        # Use 15min before tornado
        end = t - dt.timedelta(minutes=15)
        feats = compute_block_l(end, lat, lon, bbox_halfwidth_deg=bbox_halfwidth_deg)
        jumps_t.append(feats.get("ltg_jump_ratio", float("nan")))
    for (t, lat, lon) in null_events:
        end = t - dt.timedelta(minutes=15)
        feats = compute_block_l(end, lat, lon, bbox_halfwidth_deg=bbox_halfwidth_deg)
        jumps_n.append(feats.get("ltg_jump_ratio", float("nan")))

    a = np.array(jumps_t)
    b = np.array(jumps_n)
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    t, p = _welch(a, b)
    lo, hi = _boot_ci(a, b)
    return TornadoLightningResult(
        n_tornado_events=int(len(a_clean)),
        n_null_events=int(len(b_clean)),
        jump_ratio_pre_tornado_mean=float(np.mean(a_clean)) if a_clean.size else float("nan"),
        jump_ratio_null_mean=float(np.mean(b_clean)) if b_clean.size else float("nan"),
        delta_mean=float(np.mean(a_clean) - np.mean(b_clean))
            if a_clean.size and b_clean.size else float("nan"),
        welch_t=t, welch_p=p,
        bootstrap_ci_lo=lo, bootstrap_ci_hi=hi,
    )
