"""Reusable ambient-noise dv/v toolkit (single-station, vertical-component autocorrelation,
stretch method) with on-disk ACF caching so large multi-event runs are resumable.

dv/v = relative seismic velocity change. A velocity DECREASE (crack opening / damage / fluid)
shows as positive "stretch" of the coda -> we report dv/v in PERCENT, negative = slowdown.

Honest scope: single-station autocorr dv/v is a real, published technique (Brenguier et al.,
co-seismic drops; pre-seismic is contested). This toolkit is built to TEST it under control,
not to assume it works.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import io, os, time, random, threading, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import obspy

UA = {"User-Agent": "hazardpulse/0.1 (research)"}
CACHE = Path(os.environ.get("DVV_CACHE",
            str(Path(__file__).resolve().parents[2] / ".cache" / "dvv")))
CACHE.mkdir(parents=True, exist_ok=True)

FS_T = 20.0            # resample target Hz (Nyquist 10 >> 4 Hz bandpass top)
MAXLAG_S = 50.0
FMIN, FMAX = 0.5, 4.0
EARTH_KM = 6371.0

# FDSN dataselect nodes -- regional networks archive at their OWN node, NOT IRIS.
# (Confirmed: CI continuous data returns 0 bytes at IRIS, 397KB at SCEDC.)
NODES = {
    "IRIS":   "https://service.iris.edu/fdsnws/dataselect/1/query",
    "SCEDC":  "https://service.scedc.caltech.edu/fdsnws/dataselect/1/query",
    "NCEDC":  "https://service.ncedc.org/fdsnws/dataselect/1/query",
    "GEOFON": "https://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query",
    "ORFEUS": "https://www.orfeus-eu.org/fdsnws/dataselect/1/query",
    "INGV":   "https://webservices.ingv.it/fdsnws/dataselect/1/query",
    "NOA":    "https://eida.gein.noa.gr/fdsnws/dataselect/1/query",
    "AUSPASS":"http://auspass.edu.au:8080/fdsnws/dataselect/1/query",
    "GFZ":    "https://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query",
}
# network-code -> preferred node (avoids wasted requests on the wrong node)
NET_ROUTE = {
    "CI": "SCEDC", "AZ": "SCEDC", "CE": "SCEDC", "NP": "SCEDC", "SB": "SCEDC", "ZY": "SCEDC",
    "BK": "NCEDC", "NC": "NCEDC", "BG": "NCEDC", "WR": "NCEDC",
    "GE": "GEOFON", "GT": "IRIS",
    "KO": "ORFEUS", "HL": "NOA", "HT": "NOA", "HA": "NOA", "HC": "NOA", "HP": "NOA",
    "IV": "INGV", "MN": "INGV", "NI": "INGV", "OX": "INGV",
    "AU": "AUSPASS", "S1": "AUSPASS",
}
# fallback try-order after the routed node
FALLBACK = ["IRIS", "SCEDC", "NCEDC", "GEOFON", "ORFEUS", "INGV", "NOA"]
STATION_SVC = "https://service.iris.edu/fdsnws/station/1/query"  # metadata IS federated at IRIS

# per-(net,sta,cha) winning node, persisted so re-runs skip discovery
_NODE_CACHE_FP = CACHE / "_node_routes.json"
import json as _json
try:
    _NODE_CACHE = _json.loads(_NODE_CACHE_FP.read_text()) if _NODE_CACHE_FP.exists() else {}
except Exception:
    _NODE_CACHE = {}


def _node_order(net, sta, cha):
    key = f"{net}.{sta}.{cha}"
    order = []
    if key in _NODE_CACHE:
        order.append(_NODE_CACHE[key])
    routed = NET_ROUTE.get(net)
    if routed and routed not in order:
        order.append(routed)
    for n in FALLBACK:
        if n not in order:
            order.append(n)
    return key, order


# --- per-node request throttle (bulk-grade): public FDSN nodes 429 under batch load. We enforce
#     a minimum interval per node and retry transient (429/5xx/timeout) with exponential backoff,
#     while caching a "miss" ONLY for a genuine empty/204 answer (data truly absent). ---
MIN_INTERVAL = float(os.environ.get("DVV_MIN_INTERVAL", "0.18"))   # s between requests to one node
MAX_RETRIES = int(os.environ.get("DVV_MAX_RETRIES", "4"))
_TIMEOUT = 120
_GATE = {}                      # node -> [last_request_time, lock]
_GATE_LOCK = threading.Lock()
_TRANSIENT_CODES = {429, 500, 502, 503, 504}


def _throttle(node):
    with _GATE_LOCK:
        g = _GATE.setdefault(node, [0.0, threading.Lock()])
    with g[1]:
        wait = MIN_INTERVAL - (time.monotonic() - g[0])
        if wait > 0:
            time.sleep(wait)
        g[0] = time.monotonic()


def _cache_node(key, nm):
    if _NODE_CACHE.get(key) != nm:
        _NODE_CACHE[key] = nm
        with _GATE_LOCK:
            try:
                _NODE_CACHE_FP.write_text(_json.dumps(_NODE_CACHE))
            except Exception:
                pass


def _fetch_miniseed(net, sta, cha, s, e, min_bytes=40000):
    """Route per-network, throttle per node, retry transient errors with backoff.
    Returns (raw_bytes | None, transient_bool). transient=True means a network/429/5xx error
    occurred (do NOT cache a permanent miss); transient=False + None means genuine no-data."""
    key, order = _node_order(net, sta, cha)
    transient = False
    for nm in order:
        url = (f"{NODES[nm]}?net={net}&sta={sta}&loc=*&cha={cha}"
               f"&starttime={s}&endtime={e}&format=miniseed")
        for attempt in range(MAX_RETRIES):
            _throttle(nm)
            try:
                raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                             timeout=_TIMEOUT).read()
                if raw and len(raw) >= min_bytes:
                    _cache_node(key, nm)
                    return raw, False
                break                       # empty/204 = no data at THIS node -> try next node
            except urllib.error.HTTPError as ex:
                if ex.code in _TRANSIENT_CODES:
                    transient = True
                    time.sleep(min(8.0, 0.5 * 2 ** attempt) + random.random() * 0.3)
                    continue                # retry same node
                break                       # 204/404/400 = no data -> next node (no retry)
            except Exception:
                transient = True
                time.sleep(min(8.0, 0.5 * 2 ** attempt) + random.random() * 0.3)
                continue                    # timeout / URLError -> retry same node
    return None, transient


import re as _re
_ISO = _re.compile(r"\d{4}-\d{2}-\d{2}T")


def station_span(net, sta, cha):
    """Overall data-coverage [earliest_date, latest_date] for a station from the FDSN
    availability /extent service (one cheap query). Returns (earliest, latest) ISO dates, or None
    if no node serves it. Aggregates min/max across all epochs. Robust to empty Quality columns:
    parses the first two ISO-timestamp tokens per line (Earliest, Latest)."""
    _, order = _node_order(net, sta, cha)
    for nm in order:
        url = (NODES[nm].replace("/dataselect/1/query", "/availability/1/extent")
               + f"?net={net}&sta={sta}&cha={cha}&format=text")
        _throttle(nm)
        try:
            txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                         timeout=60).read().decode("utf-8", "replace")
        except Exception:
            continue
        earliest = latest = None
        for line in txt.splitlines():
            if not line or line.startswith("#"):
                continue
            iso = [t[:10] for t in line.split() if _ISO.match(t)]
            if len(iso) >= 2:
                e0, e1 = iso[0], iso[1]
                earliest = e0 if earliest is None else min(earliest, e0)
                latest = e1 if latest is None else max(latest, e1)
        if earliest:
            return (earliest, latest)
    return None


def covers(span, t0, t1):
    """True if availability `span`=(earliest,latest) covers the window [t0,t1] (ISO dates)."""
    if not span:
        return False
    return span[0] <= t0 and span[1] >= t1


def haversine(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r1)*np.cos(r2)*np.sin(dlon/2)**2
    return EARTH_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def acf_for_day(net, sta, cha, date, hours=4):
    """1-bit autocorrelation of an `hours`-long window (00:00 UTC). Cached to disk.
    Returns np.ndarray (lag samples up to MAXLAG_S*FS_T) or None."""
    tag = f"{net}.{sta}.{cha}.{date}.h{hours}.f{int(FS_T)}"
    fp = CACHE / f"{tag}.npy"
    miss = CACHE / f"{tag}.miss"
    if fp.exists():
        try:
            return np.load(fp)
        except Exception:
            pass
    if miss.exists():
        return None
    s = f"{date}T00:00:00"; e = f"{date}T{hours:02d}:00:00"
    raw, transient = _fetch_miniseed(net, sta, cha, s, e)
    if raw is None:
        if not transient:
            miss.touch()
        return None
    try:
        st = obspy.read(io.BytesIO(raw)); st.merge(fill_value=0)
        tr = st[0]
        tr.detrend("linear"); tr.taper(0.02)
        tr.filter("bandpass", freqmin=FMIN, freqmax=FMAX, zerophase=True)
        if tr.stats.sampling_rate > FS_T + 1e-6:
            tr.resample(FS_T)
        x = tr.data.astype(float)
        if len(x) < FS_T * 3600:
            miss.touch(); return None
        x = np.sign(x)
        n = len(x)
        f = np.fft.rfft(x, 2*n)
        ac = np.fft.irfft(f*np.conj(f))[:int(FS_T*MAXLAG_S)]
        ac = ac / (np.abs(ac[1:int(FS_T*2)]).max() + 1e-9)
        np.save(fp, ac.astype(np.float32))
        return ac
    except Exception:
        miss.touch(); return None


def series_day(net, sta, cha, date, hours=4, fmin=0.5, fmax=2.0):
    """Return preprocessed noise series (detrend/taper/bandpass/clip-transients/resample FS_T),
    NOT whitened/1-bit -- that happens at cross-correlation time. Cached to disk. None if no data.
    For station-pair cross-correlation dv/v (the sensitive method)."""
    tag = f"{net}.{sta}.{cha}.{date}.h{hours}.ser{fmin}_{fmax}.f{int(FS_T)}"
    fp = CACHE / f"{tag}.npy"; miss = CACHE / f"{tag}.miss"
    if fp.exists():
        try:
            return np.load(fp)
        except Exception:
            pass
    if miss.exists():
        return None
    s = f"{date}T00:00:00"; e = f"{date}T{hours:02d}:00:00"
    raw, transient = _fetch_miniseed(net, sta, cha, s, e)
    if raw is None:
        if not transient:
            miss.touch()    # only cache a miss when a node answered "no data" (not on transient errors)
        return None
    try:
        st = obspy.read(io.BytesIO(raw)); st.merge(fill_value=0)
        tr = st[0]
        tr.detrend("linear"); tr.taper(0.02)
        tr.filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True)
        if tr.stats.sampling_rate > FS_T + 1e-6:
            tr.resample(FS_T)
        x = tr.data.astype(np.float64)
        need = int(FS_T * hours * 3600)
        if len(x) < FS_T * 3600:
            miss.touch(); return None
        if len(x) < need:                       # pad short windows to fixed length for alignment
            x = np.concatenate([x, np.zeros(need - len(x))])
        else:
            x = x[:need]
        mad = np.median(np.abs(x - np.median(x))) + 1e-12
        x = np.clip(x, -8 * mad, 8 * mad)       # suppress earthquake transients
        np.save(fp, x.astype(np.float32))
        return x
    except Exception:
        miss.touch(); return None


def dvv_vs_ref(acf, ref, coda=(5, 30), grid=0.03, npts=121):
    """Stretch-method dv/v of `acf` vs `ref` over coda window (s). Returns (dvv_pct, corr)."""
    c0, c1 = int(FS_T*coda[0]), int(FS_T*coda[1])
    m = len(ref); xi = np.arange(m)
    best = (-2.0, 0.0)
    for eps in np.linspace(-grid, grid, npts):
        rs = np.interp(xi, xi*(1+eps), ref)
        c = np.corrcoef(acf[c0:c1], rs[c0:c1])[0, 1]
        if c > best[0]:
            best = (c, -eps)          # dv/v = -stretch
    return best[1]*100.0, best[0]


def _whiten(X, freqs, fmin, fmax):
    amp = np.abs(X)
    sm = np.convolve(amp, np.ones(21) / 21, "same")
    W = np.where(sm > 0, X / sm, 0.0)
    W[(freqs < fmin) | (freqs > fmax)] = 0.0
    return W


def cross_correlate(x, y, fs, maxlag_s, fmin, fmax):
    """Whitened noise cross-correlation -> symmetric CCF (lag -ML..+ML), peak-normalized.
    THE validated dv/v method (resolved Ridgecrest co-seismic drop at -2.3 sigma)."""
    n = max(len(x), len(y)); N = 2 * n
    fr = np.fft.rfftfreq(N, 1 / fs)
    Xw = _whiten(np.fft.rfft(x, N), fr, fmin, fmax)
    Yw = _whiten(np.fft.rfft(y, N), fr, fmin, fmax)
    cc = np.fft.irfft(np.conj(Xw) * Yw, N)
    ML = int(fs * maxlag_s)
    cc = np.concatenate([cc[-ML:], cc[:ML + 1]])
    return cc / (np.abs(cc).max() + 1e-9)


def dvv_stretch(cc, ref, fs, maxlag_s, coda_s=(4.0, 30.0), grid=0.02, npts=161):
    """Stretch-method dv/v of a symmetric CCF vs reference over the (two-sided) coda. (%, corr)."""
    ML = int(fs * maxlag_s)
    lags = (np.arange(2 * ML + 1) - ML) / fs
    mask = (np.abs(lags) >= coda_s[0]) & (np.abs(lags) <= coda_s[1])
    idx = np.arange(2 * ML + 1).astype(float); center = float(ML)
    best = (-2.0, 0.0)
    for eps in np.linspace(-grid, grid, npts):
        rs = np.interp(idx, (idx - center) * (1 + eps) + center, ref)
        c = np.corrcoef(cc[mask], rs[mask])[0, 1]
        if c > best[0]:
            best = (c, -eps)
    return best[1] * 100.0, best[0]


def list_dates(start, end, step_days):
    import datetime as dt
    out = []; d = start
    while d < end:
        out.append(d.isoformat()); d += dt.timedelta(days=step_days)
    return out


def find_stations(lat, lon, maxradius_deg, t_start, t_end, channels="HHZ,BHZ"):
    """Query FDSN station service for stations near (lat,lon) operating across [t_start,t_end].
    Returns list of dicts {net,sta,cha,lat,lon,dist_km,start,end} sorted by distance."""
    u = (f"{STATION_SVC}?latitude={lat}&longitude={lon}&maxradius={maxradius_deg}"
         f"&channel={channels}&starttime={t_start}&endtime={t_end}"
         f"&level=channel&format=text&includerestricted=false")
    try:
        txt = urllib.request.urlopen(urllib.request.Request(u, headers=UA),
                                     timeout=60).read().decode("utf-8", "replace")
    except Exception:
        return []
    rows = []
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split("|")
        if len(p) < 16:
            continue
        try:
            net, sta, loc, cha = p[0], p[1], p[2], p[3]
            slat, slon = float(p[4]), float(p[5])
            sstart, send = p[15], p[16] if len(p) > 16 else ""
        except Exception:
            continue
        rows.append({
            "net": net, "sta": sta, "cha": cha,
            "lat": slat, "lon": slon,
            "dist_km": float(haversine(lat, lon, slat, slon)),
            "start": sstart, "end": send,
        })
    rows.sort(key=lambda r: r["dist_km"])
    return rows
