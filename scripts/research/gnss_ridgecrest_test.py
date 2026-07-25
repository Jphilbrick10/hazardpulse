"""GNSS common-mode-filtered deformation test on Ridgecrest M7.1 (2019-07-06), CA cluster.

Positive control built in: Ridgecrest produced cm-level COSEISMIC offsets at near stations, so
the pipeline MUST show a step at the quake -> proves it can see deformation. Then the real test:
is there anomalous PRE-quake residual motion (near-epicenter vs regional common mode) above what
the SAME stations show in a quiet window 3 years earlier (negative control)?

tenv3 cols: yr=2, east(m)=8, north(m)=10, up(m)=12, lat=20, lon=21 (header line skipped).
"""
import glob, os, datetime as dt
from pathlib import Path
import numpy as np

GD = Path(__file__).resolve().parents[2] / ".cache" / "earthquake" / "gnss"
EPI = (35.770, -117.599); QUAKE_YR = 2019 + (dt.date(2019,7,6).timetuple().tm_yday)/365.25
BOX = dict(latmin=34.0, latmax=38.5, lonmin=-120.5, lonmax=-115.5)

def hav(la1,lo1,la2,lo2):
    r1,r2=np.radians(la1),np.radians(la2); dla=np.radians(la2-la1); dlo=np.radians(lo2-lo1)
    a=np.sin(dla/2)**2+np.cos(r1)*np.cos(r2)*np.sin(dlo/2)**2
    return 6371*2*np.arcsin(np.sqrt(a))

# load CA stations
stations={}
for fp in glob.glob(str(GD/'*.tenv3')):
    yr=[]; e=[]; n=[]; lat=lon=None
    try:
        with open(fp) as f:
            f.readline()
            for line in f:
                p=line.split()
                if len(p)<22: continue
                if lat is None:
                    lat=float(p[20]); lon=float(p[21]); lon=lon-360 if lon>180 else lon
                    if not (BOX['latmin']<lat<BOX['latmax'] and BOX['lonmin']<lon<BOX['lonmax']):
                        break
                yr.append(float(p[2])); e.append(float(p[8])); n.append(float(p[10]))
    except Exception:
        continue
    if lat is None or len(yr)<500: continue
    if not (BOX['latmin']<lat<BOX['latmax'] and BOX['lonmin']<lon<BOX['lonmax']): continue
    stations[os.path.basename(fp)[:-6]]=dict(yr=np.array(yr),e=np.array(e)*1000,n=np.array(n)*1000,
                                             lat=lat,lon=lon,dist=hav(*EPI,lat,lon))
print(f"loaded {len(stations)} CA GNSS stations near Ridgecrest", flush=True)
if len(stations)<5:
    print("too few stations"); raise SystemExit

# common grid (daily) over analysis windows; detrend each station by a baseline-window linear fit
def analyze(qyr, label):
    # windows in decimal years
    base0, base1 = qyr-3.0, qyr-0.5
    grid=np.arange(qyr-2.0, qyr+0.2, 1/365.25)
    res={}  # station -> (e_res, n_res) on grid
    for s,d in stations.items():
        m=(d['yr']>base0-0.2)&(d['yr']<qyr+0.3)
        if m.sum()<200: continue
        yr=d['yr'][m]
        # detrend on baseline window only (secular velocity), apply to whole
        bm=(yr>base0)&(yr<base1)
        if bm.sum()<100: continue
        for comp,arr in (('e',d['e'][m]),('n',d['n'][m])):
            A=np.vstack([yr[bm],np.ones(bm.sum())]).T
            cf=np.linalg.lstsq(A,arr[bm],rcond=None)[0]
            resid=arr-(yr*cf[0]+cf[1])
            gi=np.interp(grid,yr,resid,left=np.nan,right=np.nan)
            res.setdefault(s,{})[comp]=gi
    if len(res)<5:
        print(f"  [{label}] too few usable"); return None
    # common-mode = median across stations per epoch
    E=np.vstack([res[s]['e'] for s in res]); N=np.vstack([res[s]['n'] for s in res])
    cmE=np.nanmedian(E,axis=0); cmN=np.nanmedian(N,axis=0)
    near=[s for s in res if stations[s]['dist']<60]; far=[s for s in res if stations[s]['dist']>=60]
    def horiz_anom(slist, t0, t1):
        # MEDIAN (robust) horizontal residual displacement magnitude (cmc) over [t0,t1]
        idx=(grid>=t0)&(grid<t1)
        vals=[]
        for s in slist:
            de=res[s]['e'][idx]-cmE[idx]; dn=res[s]['n'][idx]-cmN[idx]
            v=np.sqrt(np.nanmean(de)**2+np.nanmean(dn)**2)
            if not np.isnan(v): vals.append(v)
        return np.median(vals) if vals else np.nan   # median over stations = robust to 1 bad station
    print(f"  [{label}] {len(res)} usable ({len(near)} near<60km, {len(far)} far)", flush=True)
    # pre window ENDS 7d before mainshock to EXCLUDE the M6.4 foreshock (2019-07-04) coseismic step
    pre=horiz_anom(near, qyr-0.27, qyr-7/365.25)   # [~100d, -7d] pre, foreshock excluded
    post=horiz_anom(near, qyr, qyr+0.1)            # ~35d post (coseismic step = positive control)
    far_pre=horiz_anom(far, qyr-0.27, qyr-7/365.25)
    # regional scatter: sd of near-station horizontal cmc residual over baseline
    base_vals=[]
    for s in near:
        de=res[s]['e']-cmE; dn=res[s]['n']-cmN
        bi=(grid>base0)&(grid<base1)
        base_vals.append(np.nanstd(np.sqrt(de[bi]**2+dn[bi]**2)))
    sd=np.nanmean(base_vals) if base_vals else np.nan
    print(f"     near pre-quake(100d) horiz cmc: {pre:.2f} mm | post(35d): {post:.2f} mm | "
          f"far pre: {far_pre:.2f} mm | baseline sd ~{sd:.2f} mm", flush=True)
    return dict(pre=pre, post=post, far_pre=far_pre, sd=sd, n_near=len(near))

print("\n=== REAL: Ridgecrest 2019-07-06 ===", flush=True)
real=analyze(QUAKE_YR, "real")
print("\n=== NEGATIVE CONTROL: same stations, 3 years earlier (quiet) ===", flush=True)
ctrl=analyze(QUAKE_YR-3.0, "control")

print("\n"+"="*64, flush=True)
if real:
    print(f"POSITIVE CONTROL: post-quake near-station step {real['post']:.2f} mm vs baseline sd {real['sd']:.2f} mm "
          f"-> {'VISIBLE (pipeline sees deformation)' if real['post']>3*real['sd'] else 'NOT clearly visible'}", flush=True)
    pre_excess = real['pre'] - (ctrl['pre'] if ctrl else 0)
    print(f"PRE-QUAKE TEST: near pre-quake {real['pre']:.2f} mm  vs  control pre {ctrl['pre'] if ctrl else float('nan'):.2f} mm  "
          f"(excess {pre_excess:+.2f} mm)", flush=True)
    print(f"  near-vs-far pre-quake: {real['pre']:.2f} vs {real['far_pre']:.2f} mm", flush=True)
    verdict = ("PRE-SLIP SIGNAL (near>control AND >baseline)" if (ctrl and real['pre']>ctrl['pre']+real['sd'] and real['pre']>2*real['sd'])
               else "NULL (no pre-quake deformation beyond control/noise)")
    print(f"  VERDICT: {verdict}", flush=True)
