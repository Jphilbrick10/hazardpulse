#!/usr/bin/env python3
"""Fetch active tropical cyclones from NHC ATCF, score with v8.1, output JSON.

Designed to run once daily via GitHub Actions cron. Outputs:
  - dist/data/live-storms.json    (scored active storms)
  - dist/data/live-pulse.json     (updated hurricane entry)

If no active storms exist, outputs empty arrays with honest timestamps.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

# Add src to path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.data.http import fetch_bytes, fetch_text  # noqa: E402
from hazardpulse.hurricane.atcf import (  # noqa: E402
    ATCF_ROOT,
    ATCFRecord,
    DEFAULT_AID_MODELS,
    DEFAULT_ANALYSIS_PRIORITY,
    DEFAULT_LEAD_TIMES,
    parse_atcf_deck,
)
from hazardpulse.hurricane.operational_ri import (  # noqa: E402
    best_known_operational_ri_config,
    brier_skill_score,
    build_feature_matrix,
    compute_roc_auc,
    platt_calibrate,
    predict_bagged,
    predict_histogram_gbt,
    predict_logistic,
    precompute_bins,
    rank_ensemble,
    select_features_by_logistic,
    sigmoid,
    standardize,
    train_bagged_logistic,
    train_histogram_gbt,
    train_logistic,
    train_median_impute,
)

DIST = Path(__file__).resolve().parents[1] / "dist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
HISTORICAL_DATA = RESULTS / "hurricane_operational_ri_2000_2024_al_sst.jsonl"
PRIMARY_DOMAIN = "https://hazardpulse.com"

# NHC real-time ATCF data URLs
REALTIME_ADECK_INDEX = f"{ATCF_ROOT}/aid_public/"
REALTIME_ADECK_FILE_RE = re.compile(
    r'href="(a[a-z]{2}\d{2}\d{4}\.dat\.gz)"', re.IGNORECASE
)

# Basin prefixes for all global basins
ALL_BASINS = ("AL", "EP", "CP", "WP", "IO", "SH")

# Category thresholds (Saffir-Simpson)
CATEGORY_THRESHOLDS = [
    (137, "Category 5"),
    (113, "Category 4"),
    (96, "Category 3"),
    (83, "Category 2"),
    (64, "Category 1"),
    (34, "Tropical Storm"),
    (0, "Tropical Depression"),
]


def classify_storm(vmax_kt: float | None) -> str:
    if vmax_kt is None:
        return "Unknown"
    for threshold, label in CATEGORY_THRESHOLDS:
        if vmax_kt >= threshold:
            return label
    return "Tropical Disturbance"


def list_realtime_storms(
    basin_prefixes: tuple[str, ...] = ALL_BASINS,
) -> list[str]:
    """List active storm IDs from NHC real-time a-deck index."""

    try:
        html = fetch_text(
            REALTIME_ADECK_INDEX,
            namespace="atcf_realtime",
            use_cache=False,
        )
    except Exception as e:
        print(f"  Warning: Could not fetch NHC real-time index: {e}")
        return []

    wanted = {p.upper() for p in basin_prefixes}
    storm_ids: set[str] = set()
    for filename in REALTIME_ADECK_FILE_RE.findall(html):
        storm_id = filename[1:9].upper()
        if storm_id[:2] in wanted:
            storm_ids.add(storm_id)
    return sorted(storm_ids)


def fetch_realtime_adeck(storm_id: str) -> list[ATCFRecord]:
    """Fetch real-time a-deck data for an active storm."""

    filename = f"a{storm_id.lower()}.dat.gz"
    url = f"{REALTIME_ADECK_INDEX}{filename}"
    try:
        data = fetch_bytes(url, namespace="atcf_realtime", use_cache=False)
        text = gzip.decompress(data).decode("utf-8", errors="replace")
        return parse_atcf_deck(text)
    except Exception as e:
        print(f"  Warning: Could not fetch a-deck for {storm_id}: {e}")
        return []


def _select_analysis_record(
    by_model_tau: dict[tuple[str, int], ATCFRecord],
    analysis_priority: tuple[str, ...] = DEFAULT_ANALYSIS_PRIORITY,
) -> ATCFRecord | None:
    for model in analysis_priority:
        record = by_model_tau.get((model, 0))
        if record is not None and record.vmax_kt is not None:
            return record
    return None


def build_live_case(
    storm_id: str,
    adeck_records: list[ATCFRecord],
    *,
    aid_models: tuple[str, ...] = DEFAULT_AID_MODELS,
    lead_times: tuple[int, ...] = DEFAULT_LEAD_TIMES,
) -> dict[str, object] | None:
    """Build a single feature case from the LATEST advisory cycle of an active storm.

    Unlike build_operational_ri_cases, this doesn't need b-deck/best-track
    because we're scoring a live storm, not evaluating historical accuracy.
    """

    if not adeck_records:
        return None

    # Find the latest cycle with analysis data
    cycles = sorted(
        {r.cycle for r in adeck_records if r.tau_hours == 0},
        reverse=True,
    )

    for cycle in cycles:
        by_model_tau = {
            (r.model, r.tau_hours): r
            for r in adeck_records
            if r.cycle == cycle
        }
        analysis = _select_analysis_record(by_model_tau)
        if analysis is None or analysis.vmax_kt is None:
            continue

        current_vmax = analysis.vmax_kt
        current_mslp = analysis.mslp_hpa

        # Build the case dict — same structure as historical cases
        row: dict[str, object] = {
            "storm_id": storm_id.upper(),
            "season_year": int(storm_id[-4:]),
            "issue_time": cycle.isoformat(),
            "basin": analysis.basin,
            "storm_number": analysis.storm_number,
            "storm_name": analysis.storm_name or storm_id,
            "analysis_model": analysis.model,
            "analysis_lat": analysis.lat,
            "analysis_lon": analysis.lon,
            "analysis_vmax_kt": current_vmax,
            "analysis_mslp_hpa": current_mslp,
            # No ri_label — this is a live case, truth not yet known
            "ri_label_30kt": 0,  # placeholder, not used for scoring
        }

        # Previous cycles for delta features
        for lag_hours, field_suffix in ((6, "6h"), (12, "12h"), (24, "24h")):
            lag_cycle = cycle - dt.timedelta(hours=lag_hours)
            lag_map = {
                (r.model, r.tau_hours): r
                for r in adeck_records
                if r.cycle == lag_cycle
            }
            prev = _select_analysis_record(lag_map)
            if prev is not None and prev.vmax_kt is not None:
                row[f"analysis_dv_{field_suffix}"] = current_vmax - prev.vmax_kt
            else:
                row[f"analysis_dv_{field_suffix}"] = None
            if (
                prev is not None
                and current_mslp is not None
                and prev.mslp_hpa is not None
            ):
                row[f"analysis_dp_{field_suffix}"] = current_mslp - prev.mslp_hpa
            else:
                row[f"analysis_dp_{field_suffix}"] = None

        # Aid model features
        available_aids = 0
        dv_by_lead: dict[int, list[float]] = {lt: [] for lt in lead_times}

        for aid in aid_models:
            for lead in lead_times:
                key_root = f"aid_{aid.lower()}_{lead}h"
                aid_record = by_model_tau.get((aid, lead))
                if aid_record is None or aid_record.vmax_kt is None:
                    row[f"{key_root}_available"] = 0
                    row[f"{key_root}_vmax_kt"] = None
                    row[f"{key_root}_dv_kt"] = None
                    row[f"{key_root}_mslp_hpa"] = None
                    continue
                available_aids += 1
                dv = float(aid_record.vmax_kt - current_vmax)
                row[f"{key_root}_available"] = 1
                row[f"{key_root}_vmax_kt"] = aid_record.vmax_kt
                row[f"{key_root}_dv_kt"] = dv
                row[f"{key_root}_mslp_hpa"] = aid_record.mslp_hpa
                dv_by_lead[lead].append(dv)

        row["available_aid_forecasts"] = available_aids

        # Consensus features per lead time
        for lead in lead_times:
            prefix = f"aid_consensus_{lead}h"
            dvs = dv_by_lead[lead]
            if dvs:
                row[f"{prefix}_dv_mean"] = float(np.mean(dvs))
                row[f"{prefix}_dv_min"] = float(np.min(dvs))
                row[f"{prefix}_dv_max"] = float(np.max(dvs))
                row[f"{prefix}_dv_range"] = float(np.max(dvs) - np.min(dvs))
            else:
                row[f"{prefix}_dv_mean"] = None
                row[f"{prefix}_dv_min"] = None
                row[f"{prefix}_dv_max"] = None
                row[f"{prefix}_dv_range"] = None

        # OFCL-specific features
        ofcl_24 = by_model_tau.get(("OFCL", 24))
        if ofcl_24 is not None and ofcl_24.vmax_kt is not None:
            row["aid_ofcl_24h_dv_kt"] = float(ofcl_24.vmax_kt - current_vmax)
        consensus_24_mean = row.get("aid_consensus_24h_dv_mean")
        ofcl_24_dv = row.get("aid_ofcl_24h_dv_kt")
        if consensus_24_mean is not None and ofcl_24_dv is not None:
            row["aid_ofcl_minus_consensus_24h"] = float(ofcl_24_dv) - float(
                consensus_24_mean
            )
        else:
            row["aid_ofcl_minus_consensus_24h"] = None

        # Month features
        month = cycle.month
        row["issue_month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        row["issue_month_cos"] = float(np.cos(2 * np.pi * month / 12.0))

        return row

    return None


def load_historical_cases() -> list[dict[str, object]]:
    """Load the historical JSONL training dataset."""

    if not HISTORICAL_DATA.exists():
        print(f"  ERROR: Historical data not found at {HISTORICAL_DATA}")
        return []

    cases: list[dict[str, object]] = []
    with HISTORICAL_DATA.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"  Loaded {len(cases)} historical cases")
    return cases


def train_and_score(
    historical_cases: list[dict[str, object]],
    live_cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Train v8.1 on all historical data, score the live cases."""

    if not historical_cases or not live_cases:
        return []

    config = best_known_operational_ri_config()

    # Build feature matrix from historical data
    X_hist, y_hist, years_hist, feature_names = build_feature_matrix(historical_cases)
    print(f"  Training on {len(X_hist)} cases, {len(feature_names)} raw features")

    # Build feature matrix for live cases using same feature names
    X_live, _, _, _ = build_feature_matrix(live_cases, feature_names=feature_names)

    # Impute and standardize
    X_hist_imp, X_live_imp = train_median_impute(X_hist, X_live)
    X_hist_std, X_live_std = standardize(X_hist_imp, X_live_imp)

    pos = int(np.sum(y_hist == 1))
    neg = int(np.sum(y_hist == 0))
    cw_ratio = min(neg / max(pos, 1), 20.0)
    scale_pw = cw_ratio

    # L2 feature selection
    sel_idx, sel_names, _ = select_features_by_logistic(
        X_hist_std, y_hist, feature_names,
        n_select=config.n_features_select,
        lam=config.feature_select_lam,
        n_epochs=config.feature_select_epochs,
        cw_ratio=cw_ratio,
    )
    print(f"  Selected {len(sel_idx)} features")

    X_hist_sel = X_hist_imp[:, sel_idx]
    X_live_sel = X_live_imp[:, sel_idx]
    X_hist_s, X_live_s = standardize(X_hist_sel, X_live_sel)

    # Train all 4 models
    print("  Training histogram GBT depth-3...")
    gbt3_trees, gbt3_init, gbt3_lr, _, _ = train_histogram_gbt(
        X_hist_sel, y_hist,
        n_trees=config.hgbt_d3_n_trees, max_depth=config.hgbt_d3_max_depth,
        learning_rate=config.hgbt_d3_lr, lambda_reg=config.hgbt_d3_lambda_reg,
        scale_pos_weight=scale_pw, n_bins=config.n_bins,
    )

    print("  Training histogram GBT depth-4...")
    gbt4_trees, gbt4_init, gbt4_lr, _, _ = train_histogram_gbt(
        X_hist_sel, y_hist,
        n_trees=config.hgbt_d4_n_trees, max_depth=config.hgbt_d4_max_depth,
        learning_rate=config.hgbt_d4_lr, lambda_reg=config.hgbt_d4_lambda_reg,
        min_samples_leaf=config.hgbt_d4_min_samples_leaf,
        min_child_weight=config.hgbt_d4_min_child_weight,
        colsample=config.hgbt_d4_colsample, subsample=config.hgbt_d4_subsample,
        gamma=config.hgbt_d4_gamma, scale_pos_weight=scale_pw, n_bins=config.n_bins,
    )

    print("  Training logistic regression...")
    w_lr, b_lr = train_logistic(
        X_hist_s, y_hist,
        lam=config.logistic_lam, lr=config.logistic_lr,
        n_epochs=config.logistic_epochs_final, cw_ratio=cw_ratio,
    )

    print("  Training bagged logistic (50 bags)...")
    bags = train_bagged_logistic(
        X_hist_s, y_hist,
        n_bags=config.bag_n_bags_final, feat_fraction=config.bag_feat_fraction,
        lam=config.bag_lam, lr=config.bag_lr,
        n_epochs=config.bag_epochs_final, cw_ratio=cw_ratio,
    )

    # Score live cases with all 4 models
    p_gbt3 = predict_histogram_gbt(X_live_sel, gbt3_trees, gbt3_init, gbt3_lr)
    p_gbt4 = predict_histogram_gbt(X_live_sel, gbt4_trees, gbt4_init, gbt4_lr)
    p_lr = predict_logistic(X_live_s, w_lr, b_lr)
    p_bag = predict_bagged(X_live_s, bags)

    # Bag-weighted ensemble (weight by historical AUC approximation)
    # Use equal weights since we don't have holdout AUCs for full-data training
    p_ensemble = (p_gbt3 + p_gbt4 + p_lr + p_bag) / 4.0

    # Platt calibration using training predictions
    p_hist_ens = (
        predict_histogram_gbt(X_hist_sel, gbt3_trees, gbt3_init, gbt3_lr)
        + predict_histogram_gbt(X_hist_sel, gbt4_trees, gbt4_init, gbt4_lr)
        + predict_logistic(X_hist_s, w_lr, b_lr)
        + predict_bagged(X_hist_s, bags)
    ) / 4.0

    p_calibrated, platt_a, platt_b = platt_calibrate(y_hist, p_hist_ens, p_ensemble)

    # Build scored results
    scored: list[dict[str, object]] = []
    for i, case in enumerate(live_cases):
        scored.append({
            "storm_id": case["storm_id"],
            "storm_name": case.get("storm_name", case["storm_id"]),
            "basin": case.get("basin", ""),
            "lat": case.get("analysis_lat"),
            "lon": case.get("analysis_lon"),
            "vmax_kt": case.get("analysis_vmax_kt"),
            "mslp_hpa": case.get("analysis_mslp_hpa"),
            "category": classify_storm(case.get("analysis_vmax_kt")),
            "issue_time": case.get("issue_time"),
            "ri_probability": round(float(p_calibrated[i]), 4),
            "ri_probability_raw": round(float(p_ensemble[i]), 4),
            "model_scores": {
                "gbt_d3": round(float(p_gbt3[i]), 4),
                "gbt_d4": round(float(p_gbt4[i]), 4),
                "logistic": round(float(p_lr[i]), 4),
                "bagged": round(float(p_bag[i]), 4),
            },
            "n_features": len(sel_idx),
            "model_version": "hurricane_ri_v8_1",
            "calibration": "platt",
        })

    return scored


def write_outputs(
    scored_storms: list[dict[str, object]],
    now: dt.datetime,
) -> None:
    """Write scored results to dist/data/."""

    # Write live-storms.json
    output = {
        "updated_at": now.isoformat() + "Z",
        "model_version": "hurricane_ri_v8_1",
        "n_active_storms": len(scored_storms),
        "storms": scored_storms,
    }
    storms_path = DIST / "data" / "live-storms.json"
    storms_path.parent.mkdir(parents=True, exist_ok=True)
    storms_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {storms_path} ({len(scored_storms)} storms)")

    # Update live-pulse.json hurricane entry
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "hu":
                if scored_storms:
                    # Use highest RI probability storm
                    top = max(scored_storms, key=lambda s: s["ri_probability"])
                    hazard["probability"] = top["ri_probability"]
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = _risk_band(top["ri_probability"])
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = "hurricane_ri_v8_1"
                    hazard["n_active_storms"] = len(scored_storms)
                else:
                    hazard["probability"] = 0.0
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = "none"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = "hurricane_ri_v8_1"
                    hazard["n_active_storms"] = 0
                break
        pulse["updated_at"] = now.isoformat() + "Z"
        pulse_path.write_text(json.dumps(pulse, indent=2) + "\n", encoding="utf-8")
        print(f"  Updated {pulse_path}")


def _risk_band(prob: float) -> str:
    if prob >= 0.50:
        return "critical"
    if prob >= 0.30:
        return "elevated"
    if prob >= 0.15:
        return "watch"
    if prob >= 0.05:
        return "guarded"
    return "none"


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_time(ts: dt.datetime) -> str:
    return ts.strftime("%a, %d %b %Y %H:%M:%S UTC")


def render_hurricane_page(
    scored_storms: list[dict[str, object]],
    now: dt.datetime,
) -> None:
    """Render an honest hurricane live page from the current saved feed."""
    page_path = DIST / "live" / "hurricane" / "index.html"
    page_path.parent.mkdir(parents=True, exist_ok=True)

    if scored_storms:
        rows = []
        for storm in sorted(
            scored_storms,
            key=lambda item: float(item.get("ri_probability", 0) or 0),
            reverse=True,
        )[:8]:
            rows.append(
                f"<tr>"
                f"<td>{_esc(storm.get('storm_name', storm.get('storm_id', 'Storm')))}</td>"
                f"<td>{_esc(storm.get('category', '--'))}</td>"
                f"<td>{storm.get('lat', '--')}, {storm.get('lon', '--')}</td>"
                f"<td>{float(storm.get('ri_probability', 0) or 0) * 100:.1f}%</td>"
                f"<td>{storm.get('vmax_kt', '--')} kt</td>"
                f"</tr>"
            )
        storms_html = (
            "<table><thead><tr><th>Storm</th><th>Status</th><th>Location</th><th>RI 24h</th><th>Wind</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        summary_html = (
            f"<div class=\"card hazard-hu\">"
            f"<h2 style=\"margin-top:0;\">Top storm</h2>"
            f"<div class=\"metric\">{float(scored_storms[0].get('ri_probability', 0) or 0) * 100:.1f}%</div>"
            f"<div class=\"metric-label\">Rapid intensification in 24h</div>"
            f"<div class=\"kv\"><span>Name</span><strong>{_esc(scored_storms[0].get('storm_name', scored_storms[0].get('storm_id', 'Storm')))}</strong></div>"
            f"<div class=\"kv\"><span>Status</span><strong>{_esc(scored_storms[0].get('category', '--'))} · {scored_storms[0].get('vmax_kt', '--')} kt</strong></div>"
            f"</div>"
        )
    else:
        storms_html = (
            '<div class="card"><p class="muted" style="margin:0;">'
            "No active tropical cyclones are present in the current feed."
            "</p></div>"
        )
        summary_html = (
            '<div class="card hazard-hu"><h2 style="margin-top:0;">Current state</h2>'
            '<div class="metric">0.0%</div>'
            '<div class="metric-label">Rapid intensification in 24h</div>'
            '<p class="muted">No active tropical cyclones are present in the current feed.</p>'
            "</div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Hurricane Forecasts - HazardPulse</title>
  <meta name="description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/hurricane/">
  <link rel="stylesheet" href="/assets/styles.css?v=7">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta property="og:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/hurricane/">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta name="twitter:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
        <small>Classic</small>
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/" aria-current="page">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to light mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Hurricane live page</div>
      <h1>Current tropical cyclone view</h1>
      <p class="subtitle">
        This page is rendered from the current tropical cyclone feed. When there are no active storms,
        it says so plainly instead of showing synthetic examples.
      </p>
      <p class="muted">Updated {_esc(_format_time(now))} &middot; Model: hurricane_ri_v8_1 &middot; Experimental research output only</p>
    </section>

    <section class="section">
      <div class="grid">
        <div class="col-4">
          {summary_html}
        </div>
        <div class="card col-8">
          <h2 style="margin-top:0;">Active tropical systems</h2>
          {storms_html}
          <p class="muted" style="margin-top:12px;">Source feed: <a href="/data/live-storms.json">/data/live-storms.json</a>. Always follow official advisories from the National Hurricane Center and local authorities.</p>
        </div>
      </div>
    </section>
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>Legal</h4>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        HazardPulse provides experimental research outputs only. These are not official forecasts or warnings.
        Always follow guidance from the NHC, JTWC, WMO RSMCs, and local emergency authorities.
      </p>
      <p class="footer-build">Static-first HTML &middot; Live data under <code>/data</code> &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
</body>
</html>
"""
    page_path.write_text(html, encoding="utf-8")
    print(f"  Wrote {page_path} ({len(scored_storms)} storms, honest state)")


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    print(f"HazardPulse v8.1 scoring pipeline — {now.isoformat()}Z")
    print()

    # Step 1: Check for active storms
    print("Step 1: Checking NHC for active tropical cyclones...")
    active_ids = list_realtime_storms()
    if not active_ids:
        print("  No active tropical cyclones in any basin.")
        print("  Writing empty state...")
        write_outputs([], now)
        render_hurricane_page([], now)
        print()
        print("Done. No storms to score.")
        return

    print(f"  Found {len(active_ids)} active storms: {', '.join(active_ids)}")

    # Step 2: Fetch a-deck data and build live cases
    print()
    print("Step 2: Fetching ATCF a-deck data...")
    live_cases: list[dict[str, object]] = []
    for sid in active_ids:
        print(f"  Fetching {sid}...")
        records = fetch_realtime_adeck(sid)
        if not records:
            print(f"    No records for {sid}, skipping")
            continue
        case = build_live_case(sid, records)
        if case is not None:
            name = case.get("storm_name", sid)
            vmax = case.get("analysis_vmax_kt", "?")
            cat = classify_storm(vmax if isinstance(vmax, (int, float)) else None)
            print(f"    {name}: {cat} ({vmax} kt) at {case.get('analysis_lat')}N, {case.get('analysis_lon')}W")
            live_cases.append(case)
        else:
            print(f"    Could not build case for {sid}")

    if not live_cases:
        print("  No scoreable cases found.")
        write_outputs([], now)
        render_hurricane_page([], now)
        print()
        print("Done. No storms to score.")
        return

    # Step 3: Load historical data and train model
    print()
    print("Step 3: Loading historical training data...")
    historical = load_historical_cases()
    if not historical:
        print("  ERROR: Cannot score without historical data. Exiting.")
        return

    # Step 4: Train and score
    print()
    print("Step 4: Training v8.1 model and scoring active storms...")
    scored = train_and_score(historical, live_cases)

    for s in scored:
        print(f"  {s['storm_name']}: P(RI) = {s['ri_probability']:.1%} "
              f"({s['category']}, {s['vmax_kt']} kt)")

    # Step 5: Write outputs
    print()
    print("Step 5: Writing outputs...")
    write_outputs(scored, now)
    render_hurricane_page(scored, now)

    print()
    print(f"Done. Scored {len(scored)} storms.")


if __name__ == "__main__":
    main()
