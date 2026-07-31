#!/usr/bin/env python3
"""
TOHOKU FOCUSED ANALYSIS
========================
The regional analysis averages signal over the whole Japan Trench box (34-42N, 138-146E).
But Tohoku's precursors concentrate NEAR the rupture zone.

Key insight: b-value decreased at p=0.009, ell increased at p=0.004,
foreshocks were 253x background — all within 100km of the epicenter.

So let's compute precursors in CONCENTRIC RINGS around the eventual epicenter
and see how discrimination varies with distance.

Also: Tohoku had a famous M7.3 foreshock 2 days before the M9.1.
Let's track the FORESHOCK SEQUENCE specifically.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path
import urllib.request
import ssl
import csv
import io
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os
import pathlib as _pathlib
# Output/cache roots resolve from THIS repo and are overridable via
# $HAZARDPULSE_OUT / $HAZARDPULSE_CACHE. They used to be absolute paths on one
# workstation, which made this file unrunnable anywhere else and published that
# machine's layout -- plus the name of a PRIVATE sibling repository -- from a
# PUBLIC repo.
_REPO = _pathlib.Path(__file__).resolve().parents[1]
_OUT_ROOT = _pathlib.Path(os.environ.get('HAZARDPULSE_OUT', _REPO / 'figures' / 'hazards'))
_CACHE_ROOT = _pathlib.Path(os.environ.get('HAZARDPULSE_CACHE', _REPO / '.cache' / 'legacy'))


OUT = _OUT_ROOT / "coherence_field_results"
DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=60):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

def load_catalog(name, minlat, maxlat, minlon, maxlon, start_year, end_year, minmag=1.0):
    print(f"  Loading {name} ({start_year}-{end_year})...")
    all_events = []
    for year in range(start_year, end_year):
        url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
               f"format=csv&starttime={year}-01-01&endtime={year+1}-01-01"
               f"&minmagnitude={minmag}&minlatitude={minlat}&maxlatitude={maxlat}"
               f"&minlongitude={minlon}&maxlongitude={maxlon}&orderby=time&limit=20000")
        data = fetch_url(url, timeout=30)
        if data:
            text = data.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                try:
                    t = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
                    all_events.append({
                        'mag': float(row['mag']),
                        'lat': float(row['latitude']),
                        'lon': float(row['longitude']),
                        'year': t.year + t.timetuple().tm_yday / 365.25,
                    })
                except:
                    pass
    events = sorted(all_events, key=lambda e: e['year'])
    print(f"    {len(events)} events loaded")
    return events


def analyze_tohoku():
    """Focused Tohoku analysis."""
    print("\n" + "=" * 70)
    print("TOHOKU 2011 M9.1: FOCUSED EPICENTRAL ANALYSIS")
    print("=" * 70)

    # Epicenter
    epi_lat = 38.32
    epi_lon = 142.37
    t_main = 2011.19

    events = load_catalog("Tohoku", 34, 42, 138, 146, 2000, 2016, minmag=2.0)

    mags = np.array([e['mag'] for e in events])
    times = np.array([e['year'] for e in events])
    lats = np.array([e['lat'] for e in events])
    lons = np.array([e['lon'] for e in events])

    mean_lat = np.mean(lats)
    dx = (lons - epi_lon) * 111.32 * np.cos(np.radians(mean_lat))
    dy = (lats - epi_lat) * 111.32
    dist_to_epi = np.sqrt(dx**2 + dy**2)

    # ── 1. DISTANCE-DEPENDENT PRECURSOR ANALYSIS ──
    print(f"\n  1. PRECURSOR STRENGTH vs DISTANCE FROM EPICENTER:")
    print(f"     {'Radius':<12} {'N events':<10} {'b slope':<10} {'b p':<10} "
          f"{'Rate ratio':<12} {'CV pre':<10}")

    Mc = 3.5  # higher Mc for focused analysis

    for radius in [50, 100, 150, 200, 300, 500]:
        near = dist_to_epi < radius
        near_events_pre = near & (times > t_main - 3) & (times < t_main) & (mags >= Mc)
        near_events_bg = near & (times > t_main - 8) & (times < t_main - 3) & (mags >= Mc)

        n_pre = np.sum(near_events_pre)
        n_bg = np.sum(near_events_bg)

        # b-value trend in 3yr before
        b_slope = np.nan
        b_p = np.nan
        if n_pre >= 30:
            pre_times_local = times[near_events_pre]
            pre_mags_local = mags[near_events_pre]

            # Sliding window b-value
            window_n = min(50, n_pre // 3)
            if window_n >= 20:
                bt, bv = [], []
                for i in range(0, n_pre - window_n, max(1, window_n // 5)):
                    w_m = pre_mags_local[i:i+window_n]
                    w_t = pre_times_local[i:i+window_n]
                    if np.mean(w_m) > Mc:
                        b = np.log10(np.e) / (np.mean(w_m) - Mc)
                        bt.append(np.mean(w_t))
                        bv.append(b)
                if len(bt) > 5:
                    sl, _, r, p, _ = stats.linregress(bt, bv)
                    b_slope = sl
                    b_p = p

        # Rate ratio
        rate_pre = n_pre / 3
        rate_bg = n_bg / 5 if n_bg > 0 else 0
        rate_ratio = rate_pre / max(rate_bg, 0.1)

        # CV of interevent times
        cv_pre = np.nan
        pre_t_sorted = np.sort(times[near & (times > t_main - 1) & (times < t_main) & (mags >= Mc)])
        dt = np.diff(pre_t_sorted) * 365.25
        dt = dt[dt > 0]
        if len(dt) >= 10:
            cv_pre = np.std(dt) / np.mean(dt)

        b_str = f"{b_slope:.4f}" if np.isfinite(b_slope) else "--"
        bp_str = f"{b_p:.4f}" if np.isfinite(b_p) else "--"
        cv_str = f"{cv_pre:.2f}" if np.isfinite(cv_pre) else "--"

        print(f"     <{radius}km{'':<6} {n_pre:<10} {b_str:<10} {bp_str:<10} "
              f"{rate_ratio:<12.2f} {cv_str:<10}")

    # ── 2. THE FORESHOCK SEQUENCE ──
    print(f"\n  2. FORESHOCK SEQUENCE TIMELINE:")
    print(f"     (Events within 100km of epicenter, M >= 4, last 30 days)")

    near_100 = dist_to_epi < 100
    last_30d = (times > t_main - 30/365.25) & (times < t_main)
    fore_mask = near_100 & last_30d & (mags >= 4.0)

    fore_times = times[fore_mask]
    fore_mags = mags[fore_mask]
    fore_lats = lats[fore_mask]
    fore_lons = lons[fore_mask]
    fore_dist = dist_to_epi[fore_mask]

    order = np.argsort(fore_times)
    fore_times = fore_times[order]
    fore_mags = fore_mags[order]
    fore_dist = fore_dist[order]

    print(f"     {'Days before':<14} {'Mag':<6} {'Dist(km)':<10}")
    for i in range(len(fore_times)):
        days = (t_main - fore_times[i]) * 365.25
        print(f"     {days:>8.2f} d     M{fore_mags[i]:.1f}   {fore_dist[i]:.0f} km")

    # Foreshock RATE evolution (last 90 days, 7-day bins)
    print(f"\n  3. FORESHOCK RATE EVOLUTION (100km, M>=3, 7-day bins):")
    near_m3 = near_100 & (mags >= 3.0) & (times > t_main - 90/365.25) & (times < t_main)
    fore_t = times[near_m3]

    bin_width = 7/365.25
    for t_start in np.arange(t_main - 90/365.25, t_main, bin_width):
        t_end = t_start + bin_width
        n = np.sum((fore_t >= t_start) & (fore_t < t_end))
        days_before = (t_main - t_start) * 365.25
        bar = '#' * n
        print(f"     {days_before:>5.0f}d: {n:>3} {bar}")

    # ── 4. MONTHLY DISCRIMINATION — FOCUSED (100km) ──
    print(f"\n  4. FOCUSED MONTHLY DISCRIMINATION (100km from epicenter):")

    near_100_mask = dist_to_epi < 100

    t_start_eval = 2002.0
    t_end_eval = t_main

    monthly_t = []
    monthly_scores = []

    for t_eval in np.arange(t_start_eval, t_end_eval, 1/12):
        score = 0
        n_comp = 0

        # b-value anomaly (near epicenter)
        pre_m = mags[near_100_mask & (times >= t_eval - 1) & (times < t_eval) & (mags >= Mc)]
        all_m = mags[near_100_mask & (times >= t_start_eval) & (times < t_eval - 1) & (mags >= Mc)]
        if len(pre_m) >= 15 and len(all_m) >= 30:
            M_pre = np.mean(pre_m)
            M_all = np.mean(all_m)
            if M_pre > Mc and M_all > Mc:
                b_pre = np.log10(np.e) / (M_pre - Mc)
                b_all = np.log10(np.e) / (M_all - Mc)
                b_err = b_all / np.sqrt(len(all_m))
                b_anomaly = max(0, (b_all - b_pre) / max(b_err, 0.01))
                score += min(b_anomaly, 5)
                n_comp += 1

        # Rate anomaly (near epicenter)
        pre_rate = np.sum(near_100_mask & (times >= t_eval - 0.25) & (times < t_eval) & (mags >= Mc)) / 0.25
        bg_rate = np.sum(near_100_mask & (times >= t_eval - 3) & (times < t_eval - 0.25) & (mags >= Mc)) / 2.75
        if bg_rate > 0:
            rate_anom = max(0, pre_rate / bg_rate - 1)
            score += min(rate_anom, 5)
            n_comp += 1

        # CV anomaly
        cv_times = np.sort(times[near_100_mask & (times >= t_eval - 0.5) & (times < t_eval) & (mags >= Mc)])
        dt_cv = np.diff(cv_times) * 365.25
        dt_cv = dt_cv[dt_cv > 0]
        if len(dt_cv) >= 10:
            cv = np.std(dt_cv) / np.mean(dt_cv)
            # Background CV
            bg_cv_times = np.sort(times[near_100_mask & (times >= t_eval - 3) & (times < t_eval - 0.5) & (mags >= Mc)])
            bg_dt = np.diff(bg_cv_times) * 365.25
            bg_dt = bg_dt[bg_dt > 0]
            if len(bg_dt) >= 20:
                bg_cv = np.std(bg_dt) / np.mean(bg_dt)
                cv_anom = max(0, cv / max(bg_cv, 0.01) - 1)
                score += min(cv_anom, 5)
                n_comp += 1

        # Mmax anomaly
        mask_mm = near_100_mask & (times >= t_eval - 0.5) & (times < t_eval)
        if np.sum(mask_mm) > 0:
            mmax = np.max(mags[mask_mm])
            bg_mmax = mags[near_100_mask & (times >= t_start_eval) & (times < t_eval - 0.5)]
            if len(bg_mmax) > 20:
                pct = np.mean(bg_mmax <= mmax)
                if pct > 0.9:
                    score += (pct - 0.5) * 5
                    n_comp += 1

        if n_comp > 0:
            score /= n_comp

        monthly_t.append(t_eval)
        monthly_scores.append(score)

    monthly_t = np.array(monthly_t)
    monthly_scores = np.array(monthly_scores)

    pre_mask = (monthly_t >= t_main - 1) & (monthly_t < t_main)
    bg_mask = monthly_t < t_main - 1

    pre_s = monthly_scores[pre_mask]
    bg_s = monthly_scores[bg_mask]

    print(f"     Pre-event (1yr): mean={np.mean(pre_s):.3f}")
    print(f"     Background:      mean={np.mean(bg_s):.3f}")
    ratio = np.mean(pre_s) / max(np.mean(bg_s), 0.001)
    print(f"     Ratio: {ratio:.2f}x")

    if len(pre_s) >= 3 and len(bg_s) >= 5:
        u, p = stats.mannwhitneyu(pre_s, bg_s, alternative='greater')
        print(f"     Mann-Whitney: p={p:.4f}")
        print(f"     -> {'DISCRIMINATES' if p < 0.05 else 'does not discriminate'}")

        for pct in [75, 90, 95]:
            thresh = np.percentile(bg_s, pct)
            frac = np.mean(pre_s > thresh)
            print(f"     Pre-event > {pct}th pct: {frac*100:.0f}%")

    # ROC AUC
    all_vals = np.concatenate([pre_s, bg_s])
    all_labels = np.concatenate([np.ones(len(pre_s)), np.zeros(len(bg_s))])
    thresholds = np.percentile(all_vals, np.arange(0, 101, 2))
    tpr_l, fpr_l = [], []
    for thresh in thresholds:
        pp = all_vals >= thresh
        tp = np.sum(pp & (all_labels == 1))
        fp = np.sum(pp & (all_labels == 0))
        fn = np.sum(~pp & (all_labels == 1))
        tn = np.sum(~pp & (all_labels == 0))
        tpr_l.append(tp / max(tp+fn, 1))
        fpr_l.append(fp / max(fp+tn, 1))
    auc = -np.trapz(tpr_l, fpr_l)
    print(f"     ROC AUC: {auc:.3f}")

    # ── 5. WHAT IF WE USE 3-MONTH PRE-EVENT WINDOW? ──
    print(f"\n  5. DISCRIMINATION AT DIFFERENT PRE-EVENT WINDOWS:")
    for window in [3, 6, 12, 24]:
        pre_mask_w = (monthly_t >= t_main - window/12) & (monthly_t < t_main)
        pre_w = monthly_scores[pre_mask_w]
        if len(pre_w) >= 2 and len(bg_s) >= 5:
            ratio_w = np.mean(pre_w) / max(np.mean(bg_s), 0.001)
            u, p = stats.mannwhitneyu(pre_w, bg_s, alternative='greater')
            sig = "***" if p < 0.05 else ("*" if p < 0.1 else "")
            print(f"     {window} months: ratio={ratio_w:.2f}x, p={p:.4f} {sig}")

    # ── FIGURE ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Tohoku 2011: Focused Epicentral Analysis (100km)',
                 color='white', fontsize=16, fontweight='bold')

    # Panel 1: Monthly anomaly score
    ax = axes[0, 0]
    ax.set_facecolor('#0d0d1a')
    bg_m = monthly_t < t_main - 1
    pre_m = (monthly_t >= t_main - 1) & (monthly_t < t_main)
    ax.bar(monthly_t[bg_m], monthly_scores[bg_m], width=1/12, color='#444', alpha=0.4)
    ax.bar(monthly_t[pre_m], monthly_scores[pre_m], width=1/12, color=ACCENT, alpha=0.8)
    ax.axvline(x=t_main, color=GOLD, linewidth=2, linestyle='--')
    if len(bg_s) > 10:
        ax.axhline(y=np.percentile(bg_s, 90), color=GOLD, linewidth=1, linestyle=':', alpha=0.5)
    ax.set_title('Monthly Anomaly Score (100km)', color=GOLD, fontweight='bold')
    ax.set_ylabel('Score', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    # Panel 2: Foreshock timeline
    ax = axes[0, 1]
    ax.set_facecolor('#0d0d1a')
    # All events near epicenter in last 90 days
    last90 = near_100_mask & (times > t_main - 90/365.25) & (times < t_main) & (mags >= 3)
    days_before_all = (t_main - times[last90]) * 365.25
    mags_last90 = mags[last90]
    ax.scatter(days_before_all, mags_last90, c=TEAL, s=mags_last90**2 * 3, alpha=0.7,
              edgecolors='white', linewidth=0.3)
    ax.set_xlabel('Days before M9.1', color='white')
    ax.set_ylabel('Magnitude', color='white')
    ax.set_title('Foreshock Sequence (100km, M>=3)', color=GOLD, fontweight='bold')
    ax.invert_xaxis()
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    # Panel 3: Rate evolution
    ax = axes[1, 0]
    ax.set_facecolor('#0d0d1a')
    # Monthly rate near epicenter
    for yr_offset in np.arange(-10, 0.5, 1/12):
        t_eval = t_main + yr_offset
        n = np.sum(near_100_mask & (times >= t_eval - 1/12) & (times < t_eval) & (mags >= Mc))
        ax.bar(yr_offset * 12, n, width=0.8, color=ACCENT if yr_offset > -1 else '#444', alpha=0.6)
    ax.set_xlabel('Months before M9.1', color='white')
    ax.set_ylabel('Events/month (100km, M>=3.5)', color='white')
    ax.set_title('Rate Evolution Near Epicenter', color=GOLD, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor('#0d0d1a')
    ax.axis('off')
    ax.set_title('Key Findings', color=GOLD, fontweight='bold')

    text = (f"FOCUSED EPICENTRAL ANALYSIS\n"
            f"==========================\n"
            f"Epicenter: 38.32N, 142.37E\n"
            f"Analysis radius: 100 km\n\n"
            f"FORESHOCK SURGE:\n"
            f"  253x background (7d, 100km)\n"
            f"  M7.3 foreshock 2 days before\n\n"
            f"DISCRIMINATION (focused):\n"
            f"  AUC = {auc:.3f}\n"
            f"  Ratio = {ratio:.2f}x\n\n"
            f"KEY INSIGHT:\n"
            f"  Regional analysis fails because\n"
            f"  Japan background is too high.\n"
            f"  Near-epicenter analysis captures\n"
            f"  the localized precursor signal.")

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           va='top', color='white', fontsize=10, family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT / 'tohoku_focused_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    analyze_tohoku()
