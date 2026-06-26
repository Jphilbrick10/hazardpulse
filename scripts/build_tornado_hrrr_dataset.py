#!/usr/bin/env python3
"""Self-contained HRRR-environment tornado dataset (NO ProbSevere dependency).

The live tornado tier is gated at ~0.58-0.64 because it lacks real-time atmospheric
data. This builds a real one straight from the public HRRR archive + SPC reports:

  positives : HRRR 80 km cells that contained >=1 SPC tornado that day
  negatives : CONVECTIVE cells the same day with NO tornado (cape >= --neg-min-cape)
              -- NOT random benign cells, which would make the task trivially easy
              and the AUC dishonest. The real skill is tornadic vs non-tornadic STORM
              environments.
  features  : the HRRR analysis variables at the cell, peak-pooled (instability /
              helicity / reflectivity use the cell MAX, the rest the mean).

Fetches are parallel + retrying; already-cached dates are reused. Output is an .npz
the trainer consumes for an honest temporal-holdout AUC.

    python scripts/build_tornado_hrrr_dataset.py --start 20210401 --end 20240831 \
        --hour 20 --neg-per-day 40 --neg-min-cape 250 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Peak-pool the 80 km cells (capture the convective maximum) BEFORE importing hrrr.
os.environ.setdefault("HAZARDPULSE_HRRR_POOL", "max")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data.hrrr import (  # noqa: E402
    HRRR_VARS,
    HRRR_N_LAT,
    HRRR_N_LON,
    fetch_hrrr_grid,
    load_cached_hrrr,
    latlon_to_hrrr_cell,
)
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr  # noqa: E402
from hazardpulse.tornado.definitive_model import load_spc_tornado_reports  # noqa: E402

# Raw HRRR vars + the canonical derived tornado discriminators (bulk shear, LCL
# height, 0-500 m SRH, Significant Tornado Parameter, streamwise vorticity). STP/SRH/
# shear are what actually separate tornadic from non-tornadic storm environments.
_VAR_NAMES = list(HRRR_VARS)
_DERIVED_NAMES = [
    "shear_01", "shear_06", "storm_speed", "td_depression", "lcl_est",
    "srh_05_est", "stp_eff", "rfd_warmth", "streamwise_vort",
]
_FEATURE_NAMES = _VAR_NAMES + _DERIVED_NAMES
_SPC_CSV = REPO / ".cache" / "spc" / "1950-2024_actual_tornadoes.csv"
_OUT = REPO / ".cache" / "tornado" / "hrrr_env_dataset.npz"


def _date_range(start: str, end: str) -> list[str]:
    import datetime as dt
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


def _feature_vector(grids: dict, derived: dict, i: int, j: int) -> np.ndarray:
    raw = [float(grids[v][i, j]) for v in _VAR_NAMES]
    der = [float(derived[v][i, j]) for v in _DERIVED_NAMES]
    return np.array(raw + der, dtype=np.float32)


def _cells_for(grids: dict, reports: list[dict], neg_per_day: int,
               neg_min_cape: float, rng: np.random.RandomState):
    """Positive (tornado) + hard-negative (convective, no-tornado) cells for one day."""
    try:
        derived = compute_derived_hrrr(grids)
    except Exception:
        # a missing raw var would break derived; skip the day rather than emit garbage
        return [], []
    pos_cells = set()
    for r in reports:
        try:
            i, j = latlon_to_hrrr_cell(float(r["slat"]), float(r["slon"]))
            pos_cells.add((i, j))
        except Exception:
            continue
    cape = grids.get("mlcape")
    if cape is None:
        cape = grids.get("cape")
    rows, labels = [], []
    for (i, j) in pos_cells:
        rows.append(_feature_vector(grids, derived, i, j)); labels.append(1)
    # hard negatives: convective cells (cape >= threshold), not tornadic
    cand = [
        (i, j)
        for i in range(HRRR_N_LAT) for j in range(HRRR_N_LON)
        if (i, j) not in pos_cells
        and np.isfinite(cape[i, j]) and cape[i, j] >= neg_min_cape
    ]
    if cand:
        k = min(neg_per_day, len(cand))
        for idx in rng.choice(len(cand), k, replace=False):
            i, j = cand[idx]
            rows.append(_feature_vector(grids, derived, i, j)); labels.append(0)
    return rows, labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--hour", type=int, default=20, help="HRRR analysis hour (Z)")
    ap.add_argument("--neg-per-day", type=int, default=40)
    ap.add_argument("--neg-min-cape", type=float, default=250.0,
                    help="negatives must be at least this convective (J/kg)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-dates", type=int, default=0, help="0 = no cap")
    ap.add_argument("--tornado-days-only", action="store_true",
                    help="only fetch days with >=1 tornado (cheaper, balanced)")
    ap.add_argument("--cached-only", action="store_true",
                    help="use only already-cached HRRR dates (no network) -- instant rebuild")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args(argv)

    reports = load_spc_tornado_reports(_SPC_CSV)
    dates = _date_range(args.start, args.end)
    if args.tornado_days_only:
        dates = [d for d in dates if reports.get(d)]
    if args.max_dates:
        dates = dates[: args.max_dates]
    print(f"{len(dates)} candidate dates ({args.start}..{args.end}), "
          f"{sum(1 for d in dates if reports.get(d))} with tornadoes")

    # Gather HRRR grids: cached-only (instant, no network) or parallel fetch.
    fetched: dict[str, dict] = {}
    if args.cached_only:
        for d in dates:
            g = load_cached_hrrr(d, args.hour)
            if g is not None:
                fetched[d] = g
        print(f"HRRR cached for {len(fetched)}/{len(dates)} dates (no network)")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_hrrr_grid, d, args.hour): d for d in dates}
            for fut in as_completed(futs):
                d = futs[fut]
                done += 1
                try:
                    g = fut.result()
                except Exception as exc:
                    print(f"  {d}: fetch error {exc}"); g = None
                if g is not None:
                    fetched[d] = g
                if done % 25 == 0:
                    print(f"  fetched {done}/{len(dates)} ({len(fetched)} ok)")
        print(f"HRRR available for {len(fetched)}/{len(dates)} dates")

    X_rows, y_rows, date_rows = [], [], []
    for d in sorted(fetched):
        rng = np.random.RandomState(int(d))   # deterministic per-day negatives
        rows, labels = _cells_for(fetched[d], reports.get(d, []),
                                  args.neg_per_day, args.neg_min_cape, rng)
        X_rows.extend(rows); y_rows.extend(labels); date_rows.extend([int(d)] * len(rows))

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int8)
    dates_arr = np.asarray(date_rows, dtype=np.int64)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, dates=dates_arr,
                        feature_names=np.array(_FEATURE_NAMES))
    n_pos = int(y.sum())
    print(f"dataset: {len(y)} cells  ({n_pos} tornado, {len(y) - n_pos} convective-null)  "
          f"features={X.shape[1]}  dates={len(set(date_rows))}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
