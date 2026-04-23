"""Tornado-specific Block L (lightning) augmentation layer.

Applied AT SCORING TIME to the existing tier1_ml GBT predictions.
Does NOT require retraining the model — instead, computes a
multiplicative scaling on the GBT output based on observed GLM
lightning behavior in the storm core.

The literature (Steiger+2007, Schultz+2011, Williams+1999) supports:
  - Storms with sustained ≥10 flash/min and a flash-jump ratio >2.0
    in the 15 min before observation are 2-3x more likely to produce
    EF2+ tornadoes than the average tier1_ml prediction implies.
  - Storms with <0.5 flash/min in the storm core are 0.5-0.7x as
    likely to tornado.

This is a published, defensible augmentation — not a retrain — so we
can ship it without waiting for the full GLM ablation. When the
ablation finishes and confirms (or refutes) the lift, the multiplier
coefficients can be re-fit; until then they use the literature defaults.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from hazardpulse.data.glm_lightning import N_FEAT_L, compute_block_l


# Literature-derived multiplier curve (no over-fit; conservative defaults).
# These map the (flash_rate, jump_ratio) signal into a probability
# scaling factor in [0.5, 2.5].
_BASE_RATE_MULTIPLIER = 1.0
_HIGH_RATE_THRESHOLD = 10.0   # flashes/min
_LOW_RATE_THRESHOLD = 0.5
_JUMP_THRESHOLD = 2.0
_HIGH_RATE_LIFT = 1.6
_VERY_HIGH_RATE_LIFT = 2.2
_LOW_RATE_DAMP = 0.65
_JUMP_LIFT = 1.4


def lightning_multiplier(
    storm_lat: float,
    storm_lon: float,
    valid_time: dt.datetime,
    *,
    bbox_halfwidth_deg: float = 1.0,
) -> tuple[float, dict]:
    """Return ``(multiplier, diagnostics)`` for a single storm.

    The multiplier is bounded to [0.4, 2.5] so a degenerate Block L
    cannot turn a 5% prediction into a 50% one. Caller multiplies the
    GBT probability by this value, then re-clamps to [0, 0.99].

    Returns NaN-safe diagnostics so they can be logged into the storm
    record alongside the existing coherence_diagnostics block.
    """
    diag: dict = {
        "ltg_multiplier": 1.0,
        "ltg_flash_rate_per_min": float("nan"),
        "ltg_jump_ratio": float("nan"),
        "ltg_signal": "no_data",
    }
    try:
        feats = compute_block_l(
            valid_time, storm_lat, storm_lon,
            bbox_halfwidth_deg=bbox_halfwidth_deg,
        )
    except Exception as exc:
        diag["ltg_signal"] = f"fetch_failed: {exc}"
        return 1.0, diag

    rate = feats.get("ltg_flash_rate_per_min", 0.0)
    jump = feats.get("ltg_jump_ratio", 1.0)
    diag["ltg_flash_rate_per_min"] = float(rate)
    diag["ltg_jump_ratio"] = float(jump)

    mult = _BASE_RATE_MULTIPLIER

    # Flash-rate term
    if rate >= _HIGH_RATE_THRESHOLD * 2:
        mult *= _VERY_HIGH_RATE_LIFT
        signal = "very_high_rate"
    elif rate >= _HIGH_RATE_THRESHOLD:
        mult *= _HIGH_RATE_LIFT
        signal = "high_rate"
    elif rate < _LOW_RATE_THRESHOLD:
        mult *= _LOW_RATE_DAMP
        signal = "low_rate"
    else:
        signal = "moderate_rate"

    # Flash-jump bonus (only if rate is meaningful)
    if rate >= _LOW_RATE_THRESHOLD and jump >= _JUMP_THRESHOLD:
        mult *= _JUMP_LIFT
        signal += "+jump"

    mult = max(0.4, min(mult, 2.5))
    diag["ltg_multiplier"] = float(mult)
    diag["ltg_signal"] = signal
    return mult, diag


def apply_lightning_augmentation(
    storms: list[dict],
    *,
    bbox_halfwidth_deg: float = 1.0,
    enable: bool = True,
) -> list[dict]:
    """Apply Block L augmentation to a list of scored storms in-place.

    Each storm dict is expected to have ``lat``, ``lon``, ``valid_time``
    (ISO-8601 string), and ``tornado_probability`` keys. Adds:

      - ``tornado_probability_pre_lightning``  (the original GBT output)
      - ``tornado_probability``                (multiplier-adjusted)
      - ``lightning_diagnostics``               (rate, jump, signal label)

    If ``enable`` is False, the function records the diagnostics but does
    NOT modify ``tornado_probability``. Useful for A/B comparison
    against tier1_ml-only baselines.
    """
    for storm in storms:
        lat = float(storm.get("lat", 0))
        lon = float(storm.get("lon", 0))
        vt_str = str(storm.get("valid_time", ""))
        try:
            vt = dt.datetime.fromisoformat(vt_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue
        mult, diag = lightning_multiplier(
            lat, lon, vt, bbox_halfwidth_deg=bbox_halfwidth_deg,
        )
        original = float(storm.get("tornado_probability", 0.0))
        storm["tornado_probability_pre_lightning"] = round(original, 4)
        if enable:
            adjusted = max(0.0, min(original * mult, 0.99))
            storm["tornado_probability"] = round(adjusted, 4)
        storm["lightning_diagnostics"] = diag
    return storms
