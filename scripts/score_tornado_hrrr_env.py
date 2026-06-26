#!/usr/bin/env python3
"""Live tornado forecast from the HRRR atmospheric environment + the signed forest.

Fetches the day's HRRR analysis, builds the 26-feature environment vector for every
CONUS cell (the SAME features the model trained on, via tornado.hrrr_env), and scores
each convective cell with the deployed, 0-ULP-signed VerifiableForest -> a calibrated
P(tornado-in-cell). This is the data-rich path that measured 0.81-0.85 AUC, versus the
~0.64 data-starved live tier. Standalone -- does not touch the ProbSevere scorer.

    python scripts/score_tornado_hrrr_env.py --date 20240526 --hour 20 --top 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HAZARDPULSE_HRRR_POOL", "max")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data.hrrr import (  # noqa: E402
    HRRR_N_LON, GRID_LATS, GRID_LONS, fetch_hrrr_grid, load_cached_hrrr,
)
from hazardpulse.tornado.hrrr_env import grid_feature_matrix, FEATURE_NAMES  # noqa: E402
from hazardpulse.trust.forest_serve import load_forest_scorer  # noqa: E402

_FP_DIR = REPO / "results" / "calibration"


def _risk_band(p: float) -> str:
    if p >= 0.50:
        return "high"
    if p >= 0.20:
        return "elevated"
    if p >= 0.05:
        return "marginal"
    return "low"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--hour", type=int, default=20)
    ap.add_argument("--min-cape", type=float, default=250.0,
                    help="only score convective cells (mlcape >= this)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    scorer = load_forest_scorer("tornado", _FP_DIR)
    if scorer is None:
        print("No tornado_forest_fp.json deployed yet (run train_tornado_hrrr.py --deploy).")
        return 1
    # the served feature space must match what the forest references
    feat_idx = [int(f) for f in scorer.constants.get("feat", []) if int(f) >= 0]
    if feat_idx and max(feat_idx) >= len(FEATURE_NAMES):
        print(f"Refusing: forest references feature {max(feat_idx)} >= {len(FEATURE_NAMES)}.")
        return 1

    grids = (load_cached_hrrr(args.date, args.hour) if args.cached_only
             else fetch_hrrr_grid(args.date, args.hour))
    if grids is None:
        print(f"No HRRR for {args.date} {args.hour}z.")
        return 1

    X, cape = grid_feature_matrix(grids)
    proba = scorer.raw_proba(X)                     # P(tornado) per cell (raw forest)
    cape_flat = np.asarray(cape, float).ravel()
    convective = np.isfinite(cape_flat) & (cape_flat >= args.min_cape)

    cells = []
    for flat in np.where(convective)[0]:
        i, j = divmod(int(flat), HRRR_N_LON)
        p = float(proba[flat])
        cells.append({
            "lat": round(float(GRID_LATS[i]), 2), "lon": round(float(GRID_LONS[j]), 2),
            "row": i, "col": j, "probability": round(p, 4),
            "risk_band": _risk_band(p), "mlcape": round(float(cape_flat[flat]), 0),
        })
    cells.sort(key=lambda c: c["probability"], reverse=True)

    forecast = {
        "hazard": "tornado", "method": "hrrr_environment_forest",
        "date": args.date, "hour": args.hour,
        "model_sha256": scorer.constants.get("model_sha256") or scorer.model_sha256,
        "n_convective_cells": int(convective.sum()),
        "max_probability": (cells[0]["probability"] if cells else 0.0),
        "top_cells": cells[: args.top],
    }
    print(f"HRRR-env tornado forecast {args.date} {args.hour}z: "
          f"{forecast['n_convective_cells']} convective cells, "
          f"max P(tor)={forecast['max_probability']:.3f}")
    for c in cells[: args.top]:
        print(f"  ({c['lat']:.1f},{c['lon']:.1f})  P={c['probability']:.3f}  "
              f"{c['risk_band']:9s} cape={c['mlcape']:.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps(forecast, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
