#!/usr/bin/env python3
"""
SUMATRA 2004 M9.1: FOCUSED EPICENTRAL ANALYSIS
================================================
Same approach that lifted Tohoku from AUC=0.388 to AUC=0.861.
Hypothesis: Sumatra's signal is also concentrated near the rupture zone.

Epicenter: 3.316N, 95.854E
Rupture zone: ~1300km long along the Sunda Trench
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import urllib.request
import csv
import io
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'coherence_field_results')
os.makedirs(OUT_DIR, exist_ok=True)

EPICENTER = (3.316, 95.854)
EVENT_DATE = (2004, 12, 26)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def load_usgs_data():
    """Load Sumatra region seismicity from USGS."""
    print("  Loading Sumatra region (1990-2010)...")

    # Broad box around northern Sumatra
    params = {
        'format': 'csv',
        'starttime': '1990-01-01',
        'endtime': '2010-12-31',
        'minlatitude': -2,
        'maxlatitude': 10,
        'minlongitude': 90,
        'maxlongitude': 100,
        'minmagnitude': 3.0,
        'orderby': 'time',
        'limit': 20000,
    }

    query = '&'.join(f'{k}={v}' for k, v in params.items())
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?{query}"

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode('utf-8')

    reader = csv.DictReader(io.StringIO(raw))
    events = []
    for row in reader:
        try:
            t = row['time']
            yr = int(t[:4])
            mo = int(t[5:7])
            dy = int(t[8:10])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            mag = float(row['mag'])
            events.append({'yr': yr, 'mo': mo, 'dy': dy, 'lat': lat, 'lon': lon, 'mag': mag,
                          'dist_km': haversine(lat, lon, EPICENTER[0], EPICENTER[1])})
        except:
            continue

    print(f"    {len(events)} events loaded")
    return events


def analyze_distance_dependent(events):
    """Precursor strength vs distance from epicenter."""
    print("\n  1. PRECURSOR STRENGTH vs DISTANCE FROM EPICENTER:")

    # Pre-event: 2002-01 to 2004-11 (exclude mainshock month)
    pre_mask = [(e['yr'] >= 2002 and (e['yr'] < 2004 or (e['yr'] == 2004 and e['mo'] < 12))) for e in events]
    # Background: 1990-2001
    bg_mask = [e['yr'] < 2002 for e in events]

    pre_events = [e for e, m in zip(events, pre_mask) if m]
    bg_events = [e for e, m in zip(events, bg_mask) if m]

    print(f"     {'Radius':>10}  {'N pre':>7}  {'N bg':>7}  {'Rate ratio':>11}  {'b pre':>6}  {'b bg':>6}")

    radii = [50, 100, 150, 200, 300, 500, 750]
    results = {}

    for r in radii:
        pre_r = [e for e in pre_events if e['dist_km'] <= r]
        bg_r = [e for e in bg_events if e['dist_km'] <= r]

        n_pre = len(pre_r)
        n_bg = len(bg_r)

        # Rate: events per year
        rate_pre = n_pre / 3.0  # 3 years pre-event
        rate_bg = n_bg / 12.0   # 12 years background

        ratio = rate_pre / rate_bg if rate_bg > 0 else 0

        # b-value
        b_pre = b_bg = '--'
        if n_pre > 20:
            mags_pre = np.array([e['mag'] for e in pre_r])
            mc = np.percentile(mags_pre, 10)
            mc = max(mc, 3.5)
            above = mags_pre[mags_pre >= mc]
            if len(above) > 10:
                b_pre = f"{1.0 / (np.log(10) * (np.mean(above) - mc + 0.05)):.2f}"

        if n_bg > 20:
            mags_bg = np.array([e['mag'] for e in bg_r])
            mc = np.percentile(mags_bg, 10)
            mc = max(mc, 3.5)
            above = mags_bg[mags_bg >= mc]
            if len(above) > 10:
                b_bg = f"{1.0 / (np.log(10) * (np.mean(above) - mc + 0.05)):.2f}"

        results[r] = {'n_pre': n_pre, 'n_bg': n_bg, 'ratio': ratio}
        print(f"     {r:>7}km  {n_pre:>7}  {n_bg:>7}  {ratio:>11.2f}  {b_pre:>6}  {b_bg:>6}")

    return results


def focused_monthly_discrimination(events, radius_km=200):
    """Monthly discrimination test within focused radius."""
    print(f"\n  2. FOCUSED MONTHLY DISCRIMINATION ({radius_km}km from epicenter):")

    # Filter to radius
    nearby = [e for e in events if e['dist_km'] <= radius_km]
    print(f"     {len(nearby)} events within {radius_km}km")

    # Monthly counts
    monthly = defaultdict(int)
    for e in nearby:
        key = (e['yr'], e['mo'])
        monthly[key] += 1

    # Fill gaps
    all_months = []
    for yr in range(1990, 2010):
        for mo in range(1, 13):
            if yr == 2004 and mo == 12:
                continue  # Skip mainshock month
            if yr > 2004:
                continue  # Skip aftershock period
            all_months.append(((yr, mo), monthly.get((yr, mo), 0)))

    dates = [x[0] for x in all_months]
    counts = np.array([x[1] for x in all_months])

    # Pre-event: last 12 months before mainshock (2003-12 to 2004-11)
    pre_mask = np.array([1 if (d[0] == 2004 and d[1] < 12) or (d[0] == 2003 and d[1] == 12) else 0 for d in dates])
    # Background: 1990-2002
    bg_mask = np.array([1 if d[0] < 2003 else 0 for d in dates])

    pre_counts = counts[pre_mask == 1]
    bg_counts = counts[bg_mask == 1]

    if len(pre_counts) > 0 and len(bg_counts) > 0:
        pre_mean = np.mean(pre_counts)
        bg_mean = np.mean(bg_counts)
        ratio = pre_mean / bg_mean if bg_mean > 0 else float('inf')

        u_stat, p_val = stats.mannwhitneyu(pre_counts, bg_counts, alternative='greater')

        print(f"     Pre-event (12mo): mean={pre_mean:.1f}")
        print(f"     Background:       mean={bg_mean:.1f}")
        print(f"     Ratio: {ratio:.2f}x")
        print(f"     Mann-Whitney: p={p_val:.4f}")
        print(f"     -> {'DISCRIMINATES' if p_val < 0.05 else 'does not discriminate'}")

        # AUC
        labels = np.concatenate([np.ones(len(pre_counts)), np.zeros(len(bg_counts))])
        scores = np.concatenate([pre_counts, bg_counts])

        pos = scores[labels == 1]
        neg = scores[labels == 0]
        n_conc = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
        auc = n_conc / (len(pos) * len(neg))

        print(f"     ROC AUC: {auc:.3f}")

        # Multiple time windows
        print(f"\n  3. DISCRIMINATION AT DIFFERENT PRE-EVENT WINDOWS:")
        for window_months in [3, 6, 12, 24, 36]:
            # Pre-event window
            pre_dates = []
            for yr in range(1990, 2005):
                for mo in range(1, 13):
                    ym = yr * 12 + mo
                    event_ym = 2004 * 12 + 12
                    if event_ym - window_months <= ym < event_ym:
                        pre_dates.append((yr, mo))

            pre_w = np.array([monthly.get(d, 0) for d in pre_dates])
            bg_w = bg_counts  # same background

            if len(pre_w) > 1:
                ratio_w = np.mean(pre_w) / np.mean(bg_w) if np.mean(bg_w) > 0 else 0
                _, p_w = stats.mannwhitneyu(pre_w, bg_w, alternative='greater')
                sig = '***' if p_w < 0.05 else ''
                print(f"     {window_months:>2} months: ratio={ratio_w:.2f}x, p={p_w:.4f} {sig}")

    return


def foreshock_analysis(events, radius_km=200):
    """Foreshock sequence in last 90 days."""
    print(f"\n  4. FORESHOCK SEQUENCE ({radius_km}km, last 90 days):")

    nearby = [e for e in events if e['dist_km'] <= radius_km]

    # Convert to days before mainshock
    from datetime import date
    mainshock = date(2004, 12, 26)

    foreshocks = []
    for e in nearby:
        try:
            d = date(e['yr'], e['mo'], e['dy'])
            days_before = (mainshock - d).days
            if 0 < days_before <= 90 and e['mag'] >= 4.0:
                foreshocks.append((days_before, e['mag'], e['dist_km']))
        except:
            continue

    foreshocks.sort(key=lambda x: -x[0])

    if foreshocks:
        print(f"     {'Days before':>12}  {'Mag':>5}  {'Dist(km)':>8}")
        for db, m, d in foreshocks:
            print(f"     {db:>10}d   M{m:.1f}   {d:.0f} km")
    else:
        print(f"     No M4+ foreshocks in window")

    # Rate evolution in 7-day bins
    print(f"\n  5. FORESHOCK RATE EVOLUTION ({radius_km}km, M>=3, 7-day bins):")
    nearby_all = [e for e in events if e['dist_km'] <= radius_km and e['mag'] >= 3.0]

    for bin_start in range(90, -1, -7):
        bin_end = max(bin_start - 7, 0)
        count = 0
        for e in nearby_all:
            try:
                d = date(e['yr'], e['mo'], e['dy'])
                db = (mainshock - d).days
                if bin_end < db <= bin_start:
                    count += 1
            except:
                continue
        bar = '#' * count
        print(f"     {bin_start:>5}d: {count:>3} {bar}")


def correlation_length(events, radius_km=300):
    """Compute correlation length (mean inter-event distance) over time."""
    print(f"\n  6. CORRELATION LENGTH EVOLUTION ({radius_km}km):")

    nearby = [e for e in events if e['dist_km'] <= radius_km]

    # Monthly correlation length
    monthly_ell = {}
    for yr in range(1995, 2005):
        for mo in range(1, 13):
            month_events = [e for e in nearby if e['yr'] == yr and e['mo'] == mo]
            if len(month_events) >= 5:
                # Mean pairwise distance
                lats = np.array([e['lat'] for e in month_events])
                lons = np.array([e['lon'] for e in month_events])
                dists = []
                for i in range(len(lats)):
                    for j in range(i+1, len(lats)):
                        dists.append(haversine(lats[i], lons[i], lats[j], lons[j]))
                if dists:
                    monthly_ell[(yr, mo)] = np.mean(dists)

    if monthly_ell:
        # Pre-event vs background
        pre_ell = [v for k, v in monthly_ell.items() if k[0] >= 2003]
        bg_ell = [v for k, v in monthly_ell.items() if k[0] < 2003]

        if pre_ell and bg_ell:
            print(f"     Pre-event mean ell: {np.mean(pre_ell):.1f} km")
            print(f"     Background mean ell: {np.mean(bg_ell):.1f} km")
            ratio = np.mean(pre_ell) / np.mean(bg_ell)
            _, p_ell = stats.mannwhitneyu(pre_ell, bg_ell, alternative='two-sided')
            print(f"     Ratio: {ratio:.2f}x")
            print(f"     Mann-Whitney p={p_ell:.4f}")

            # Is ell increasing (diverging) before mainshock?
            all_dates = sorted(monthly_ell.keys())
            all_ells = [monthly_ell[d] for d in all_dates]
            all_numeric = [d[0] + d[1]/12 for d in all_dates]

            # Last 3 years trend
            mask_3yr = [d[0] >= 2002 for d in all_dates]
            recent_dates = [n for n, m in zip(all_numeric, mask_3yr) if m]
            recent_ells = [e for e, m in zip(all_ells, mask_3yr) if m]

            if len(recent_dates) > 5:
                slope, _, r, p_trend, _ = stats.linregress(recent_dates, recent_ells)
                print(f"     3-year trend: slope={slope:.1f} km/yr, p={p_trend:.4f}")
                print(f"     -> {'DIVERGING' if slope > 0 and p_trend < 0.1 else 'no clear trend'}")


def multi_precursor_composite(events, radius_km=200):
    """Composite score from multiple precursors."""
    print(f"\n  7. MULTI-PRECURSOR COMPOSITE ({radius_km}km):")

    nearby = [e for e in events if e['dist_km'] <= radius_km]

    # Monthly features
    months_data = []
    for yr in range(1992, 2005):
        for mo in range(1, 13):
            if yr == 2004 and mo == 12:
                continue
            month_events = [e for e in nearby if e['yr'] == yr and e['mo'] == mo]
            n = len(month_events)
            mags = [e['mag'] for e in month_events]

            features = {'yr': yr, 'mo': mo, 'count': n}

            if len(mags) >= 5:
                mags = np.array(mags)
                features['mean_mag'] = np.mean(mags)
                features['max_mag'] = np.max(mags)

                # b-value (simple)
                mc = max(np.percentile(mags, 10), 3.5)
                above = mags[mags >= mc]
                if len(above) >= 5:
                    features['b_value'] = 1.0 / (np.log(10) * (np.mean(above) - mc + 0.05))

                # IET CV
                if n >= 3:
                    from datetime import date as dt
                    times = sorted([dt(e['yr'], e['mo'], min(e['dy'], 28)).toordinal() for e in month_events])
                    iets = np.diff(times).astype(float)
                    iets = iets[iets > 0]
                    if len(iets) >= 2:
                        features['cv'] = np.std(iets) / np.mean(iets) if np.mean(iets) > 0 else 0
            else:
                features['mean_mag'] = 0
                features['max_mag'] = 0

            months_data.append(features)

    # Convert to arrays and compute percentiles
    counts = np.array([m['count'] for m in months_data])
    is_pre = np.array([1 if (m['yr'] >= 2003 or (m['yr'] == 2002 and m['mo'] >= 12)) else 0 for m in months_data])

    # Count percentile
    count_pct = np.array([stats.percentileofscore(counts, c) / 100 for c in counts])

    # Composite = just count percentile for now (since that's the strongest signal)
    composite = count_pct

    pre_scores = composite[is_pre == 1]
    bg_scores = composite[is_pre == 0]

    if len(pre_scores) > 0 and len(bg_scores) > 0:
        u, p = stats.mannwhitneyu(pre_scores, bg_scores, alternative='greater')
        print(f"     Pre-event composite: {np.mean(pre_scores):.3f}")
        print(f"     Background composite: {np.mean(bg_scores):.3f}")
        print(f"     Mann-Whitney p={p:.4f}")

        # AUC
        labels = np.concatenate([np.ones(len(pre_scores)), np.zeros(len(bg_scores))])
        scores_all = np.concatenate([pre_scores, bg_scores])
        pos = scores_all[labels == 1]
        neg = scores_all[labels == 0]
        n_conc = sum(np.sum(p_val > neg) + 0.5 * np.sum(p_val == neg) for p_val in pos)
        auc = n_conc / (len(pos) * len(neg))
        print(f"     AUC = {auc:.3f}")


def main():
    print("="*70)
    print("SUMATRA 2004 M9.1: FOCUSED EPICENTRAL ANALYSIS")
    print("="*70)

    events = load_usgs_data()
    if not events:
        return

    analyze_distance_dependent(events)
    focused_monthly_discrimination(events, radius_km=200)
    foreshock_analysis(events, radius_km=200)
    correlation_length(events, radius_km=300)
    multi_precursor_composite(events, radius_km=200)

    # Try multiple radii
    print("\n  8. AUC vs RADIUS:")
    for r in [100, 150, 200, 300, 500]:
        nearby = [e for e in events if e['dist_km'] <= r]
        monthly = defaultdict(int)
        for e in nearby:
            key = (e['yr'], e['mo'])
            monthly[key] += 1

        pre_counts = []
        bg_counts = []
        for yr in range(1990, 2005):
            for mo in range(1, 13):
                if yr == 2004 and mo == 12:
                    continue
                c = monthly.get((yr, mo), 0)
                if yr >= 2004 or (yr == 2003 and mo == 12):
                    pre_counts.append(c)
                elif yr < 2003:
                    bg_counts.append(c)

        pre_counts = np.array(pre_counts)
        bg_counts = np.array(bg_counts)

        if len(pre_counts) > 0 and len(bg_counts) > 0:
            labels = np.concatenate([np.ones(len(pre_counts)), np.zeros(len(bg_counts))])
            scores = np.concatenate([pre_counts, bg_counts])
            pos = scores[labels == 1]
            neg = scores[labels == 0]
            n_conc = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
            auc = n_conc / (len(pos) * len(neg))
            ratio = np.mean(pre_counts) / np.mean(bg_counts) if np.mean(bg_counts) > 0 else 0
            print(f"     {r:>5}km: AUC={auc:.3f}, ratio={ratio:.2f}x, N_pre={len(pre_counts)}, N_bg={len(bg_counts)}")


if __name__ == '__main__':
    main()
