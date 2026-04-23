"""Block W (space weather) feature extraction for earthquake samples.

Bridges the generic ``hazardpulse.data.space_weather`` module to the
earthquake training/scoring pipelines:

  - ``BLOCK_W_NAMES`` re-exported with the EQ-specific ordering
  - ``compute_block_w_for_event(event_time)`` returns a fixed-length
    np.float32 ndarray suitable for concatenation with Block S + C.

The feature definitions are intentionally identical to what the hurricane
and tornado pipelines see, so cross-hazard ablation experiments produce
directly comparable numbers.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import numpy as np

from hazardpulse.data.space_weather import (
    BLOCK_W_NAMES,
    N_FEAT_W,
    space_weather_features_for_window,
)

__all__ = ["BLOCK_W_NAMES", "N_FEAT_W", "compute_block_w_for_event",
           "compute_block_w_batch"]


def compute_block_w_for_event(
    event_time: dt.datetime,
    *,
    window_h: int = 72,
) -> np.ndarray:
    """Return Block W as a (N_FEAT_W,) float32 ndarray (NaN-where-missing)."""
    feats = space_weather_features_for_window(event_time, window_h=window_h)
    return np.array([feats[n] for n in BLOCK_W_NAMES], dtype=np.float32)


def compute_block_w_batch(
    event_times: Iterable[dt.datetime],
    *,
    window_h: int = 72,
    progress_every: int = 200,
) -> np.ndarray:
    """Return (n_events, N_FEAT_W) float32 ndarray for a batch of events."""
    times = list(event_times)
    out = np.full((len(times), N_FEAT_W), np.nan, dtype=np.float32)
    for i, t in enumerate(times):
        if progress_every and i and i % progress_every == 0:
            print(f"    Block W: {i}/{len(times)}...")
        out[i] = compute_block_w_for_event(t, window_h=window_h)
    return out
