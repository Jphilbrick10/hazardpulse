"""GOLD-STANDARD dv/v: noise CROSS-correlation between station pairs (whitened), stacked,
stretch-method dv/v on the symmetric coda, averaged across pairs. THE method known to resolve
~0.1% co-seismic drops. Validation target: a clear velocity DROP at Ridgecrest M7.1 (2019-07-06).
"""
import sys, os, datetime as dt, itertools
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dvv_lib import series_day, list_dates, FS_T

CHA = "HHZ"
STAS = ["CLC", "TOW2", "SRT", "WRC2"]      # all CI, SCEDC, <=20km from Ridgecrest, 2018-19
NET = "CI"
FMIN, FMAX = 0.5, 2.0
QUAKE = dt.date(2019, 7, 6)
START, END, STEP = dt.date(2018, 1, 1), dt.date(2019, 11, 1), 2
MAXLAG_S = 40.0
CODA = (4.0, 30.0)                          # symmetric coda window (s)

dates = list_dates(START, END, STEP)
print(f"fetching {len(STAS)} stations x {len(dates)} days (threaded, cached) ...", flush=True)
series = {s: {} for s in STAS}
def fetch(args):
    s, d = args
    return s, d, series_day(NET, s, CHA, d, hours=4, fmin=FMIN, fmax=FMAX)
jobs = [(s, d) for s in STAS for d in dates]
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    done = 0
    for s, d, x in ex.map(fetch, jobs):
        done += 1
        if x is not None:
            series[s][d] = x
        if done % 200 == 0:
            print(f"  {done}/{len(jobs)}  " + " ".join(f"{k}:{len(series[k])}" for k in STAS), flush=True)
for s in STAS:
    print(f"  {s}: {len(series[s])} days", flush=True)

# ---- whitened cross-correlation ----
def whiten(X, f):
    amp = np.abs(X)
    sm = np.convolve(amp, np.ones(21)/21, "same")
    W = np.where(sm > 0, X/sm, 0.0)
    W[(f < FMIN) | (f > FMAX)] = 0.0
    return W

ML = int(FS_T * MAXLAG_S)
def ccf(x, y):
    n = max(len(x), len(y)); N = 2*n
    f = np.fft.rfftfreq(N, 1/FS_T)
    Xw = whiten(np.fft.rfft(x, N), f); Yw = whiten(np.fft.rfft(y, N), f)
    cc = np.fft.irfft(np.conj(Xw) * Yw, N)
    cc = np.concatenate([cc[-ML:], cc[:ML+1]])   # lag -ML..+ML
    return cc / (np.abs(cc).max() + 1e-9)

lags = (np.arange(2*ML+1) - ML)
lt = lags / FS_T
coda_mask = (np.abs(lt) >= CODA[0]) & (np.abs(lt) <= CODA[1])
idx = np.arange(2*ML+1).astype(float); center = float(ML)

def dvv_ccf(cc, ref, grid=0.02, npts=161):
    best = (-2.0, 0.0)
    for eps in np.linspace(-grid, grid, npts):
        rs = np.interp(idx, (idx - center)*(1 + eps) + center, ref)
        c = np.corrcoef(cc[coda_mask], rs[coda_mask])[0, 1]
        if c > best[0]:
            best = (c, -eps)
    return best[1]*100.0, best[0]

pairs = list(itertools.combinations(STAS, 2))
print(f"\ncomputing CCFs for {len(pairs)} pairs ...", flush=True)
pair_cc = {}     # pair -> {date: cc}
for a, b in pairs:
    common = sorted(set(series[a]) & set(series[b]))
    d = {}
    for day in common:
        d[dt.date.fromisoformat(day)] = ccf(series[a][day], series[b][day])
    pair_cc[(a, b)] = d
    print(f"  {a}-{b}: {len(d)} daily CCFs", flush=True)

# stacked dv/v per pair, reference = 2018 stack
HALF = 10
def pair_dvv(d):
    days = sorted(d)
    ref_days = [k for k in days if k < dt.date(2019, 1, 1)]
    if len(ref_days) < 30:
        return []
    ref = np.mean([d[k] for k in ref_days], axis=0)
    out = []
    for day in days:
        lo, hi = day - dt.timedelta(days=HALF), day + dt.timedelta(days=HALF)
        mem = [d[k] for k in days if lo <= k <= hi]
        if len(mem) < 5:
            continue
        stack = np.mean(mem, axis=0)
        v, c = dvv_ccf(stack, ref)
        out.append((day, v, c))
    return out

allrows = {}
for p in pairs:
    rows = pair_dvv(pair_cc[p])
    if rows:
        allrows[p] = rows
        # quick per-pair co-seismic check
        base = np.array([v for dd, v, c in rows if dd < dt.date(2019,5,1) and c > 0.6])
        co = np.array([v for dd, v, c in rows if QUAKE <= dd < dt.date(2019,9,1) and c > 0.6])
        if len(base) and len(co):
            print(f"  {p[0]}-{p[1]}: co-seismic {co.mean()-base.mean():+.3f}% "
                  f"({(co.mean()-base.mean())/(base.std()+1e-9):+.1f} sigma)", flush=True)

# network-average dv/v: average across pairs per date (corr-weighted)
print("\n===== NETWORK-AVERAGE dv/v (corr-weighted across pairs) =====", flush=True)
bydate = {}
for p, rows in allrows.items():
    for dd, v, c in rows:
        if c > 0.55:
            bydate.setdefault(dd, []).append((v, c))
series_avg = []
for dd in sorted(bydate):
    vs = bydate[dd]
    w = np.array([c for v, c in vs]); val = np.array([v for v, c in vs])
    series_avg.append((dd, np.average(val, weights=w), len(vs), w.mean()))

base = np.array([v for dd, v, n, c in series_avg if dd < dt.date(2019,5,1)])
pre  = np.array([v for dd, v, n, c in series_avg if dt.date(2019,5,1) <= dd < QUAKE])
co   = np.array([v for dd, v, n, c in series_avg if QUAKE <= dd < dt.date(2019,9,1)])
post = np.array([v for dd, v, n, c in series_avg if dd >= dt.date(2019,9,1)])
bsd = base.std() + 1e-9; bm = base.mean()
print(f"baseline 2018-Apr2019: {bm:+.3f}%  sd {bsd:.3f}  (n={len(base)})", flush=True)
if len(pre): print(f"pre-quake May-Jul:     {pre.mean():+.3f}%  ({(pre.mean()-bm)/bsd:+.1f} sigma, n={len(pre)})", flush=True)
if len(co):  print(f"CO-SEISMIC Jul-Aug:    {co.mean():+.3f}%  ({(co.mean()-bm)/bsd:+.1f} sigma, n={len(co)})  <- expect DROP", flush=True)
if len(post):print(f"post Sep+:             {post.mean():+.3f}%  ({(post.mean()-bm)/bsd:+.1f} sigma, n={len(post)})", flush=True)
print("\ntimeline (network avg):", flush=True)
for dd, v, n, c in series_avg:
    if dd.day <= STEP*2 or abs((dd-QUAKE).days) <= 30:
        star = "  <== QUAKE" if abs((dd-QUAKE).days) <= 6 else ""
        bar = "-"*int(max(0,-v)*30) + "+"*int(max(0,v)*30)
        print(f"  {dd}  {v:+.3f}%  (npairs {n}, corr {c:.2f}) {bar}{star}", flush=True)
if len(co) and len(base):
    sig = (co.mean()-bm)/bsd
    print(f"\n>>> CO-SEISMIC = {sig:+.1f} sigma -> "
          f"{'VALIDATED: cross-correlation dv/v resolves the Ridgecrest drop' if sig < -2 else 'STILL not resolving -- method/data limit'}", flush=True)
