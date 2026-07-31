#!/usr/bin/env python3
"""Do the classic catalog precursors -- b-value, AMR (accelerating moment release),
natural-time (Varotsos kappa1), Coulomb stress (GCMT focal mechs), tidal phase -- carry
OPERATIONAL signal BEYOND the multi-scale rate/quiescence/stress context the deep model
already uses?

Runs on the EXACT cached operational sample set (no rebuild): each sample's lat/lon are
recovered from context channels 6-7 and ref time from T. We compute the new features per
sample, then measure (a) univariate operational AUC and (b) whether a GBT on
[existing 14 context features + new features] beats a GBT on [context only], out of sample.
"""
import sys, os, time, datetime as dt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
# Resolve the repo from this file's own location: the previous hardcoded
# workstation path broke every other checkout and published the operator's
# directory layout from a public repository.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))

SEC_DAY = 86400.0
NPZ = os.environ.get(
    "HAZARDPULSE_EQ_NPZ",
    os.path.join(_REPO, ".cache", "earthquake",
                 "deepop_v3_my2025_m5.0_lr100_ld30_K192_ir100_am8_g2.npz"))
VAL0 = dt.datetime(2018,1,1,tzinfo=dt.timezone.utc).timestamp()
TEST0 = dt.datetime(2020,1,1,tzinfo=dt.timezone.utc).timestamp()

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

# ---- worker globals ----
_LAT=_LON=_T=_M0=_E=None     # catalog arrays; _M0 GCMT, _E energies
_GLA=_GLO=_GM0=_GT=None      # GCMT arrays

def _epoch_from_iso_z(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z","+00:00")).timestamp()
    except ValueError:
        return None

def _winit():
    global _LAT,_LON,_T,_E,_GLA,_GLO,_GM0,_GT
    from hazardpulse.earthquake.definitive_model import load_usgs_catalog, CatalogArrays
    cl = load_usgs_catalog(min_year=2000, max_year=2025, min_mag=2.5)
    cat = CatalogArrays(cl, verbose=False)
    _LAT,_LON,_T = cat.lats, cat.lons, cat.times
    _E = 10.0**(1.5*cat.mags + 4.8)        # radiated energy proxy (J)
    globals()['_MAG'] = cat.mags
    # GCMT focal-mechanism catalog for Coulomb proxy
    try:
        from hazardpulse.data.earthquake import load_gcmt_catalog
        g = load_gcmt_catalog()
        gla=[]; glo=[]; gm0=[]; gt=[]
        for r in g:
            try:
                la=float(r["lat"]); lo=float(r["lon"]); mw=float(r["Mw"])
            except Exception: continue
            if mw < 6.0: continue
            epoch=_epoch_from_iso_z(r.get("time"))
            if epoch is None: continue
            try:
                moment=float(r.get("scalar_moment") or 10.0**(1.5*mw+9.1))
            except Exception:
                moment=10.0**(1.5*mw+9.1)
            gla.append(la); glo.append(lo); gm0.append(moment); gt.append(epoch)
        _GLA=np.array(gla); _GLO=np.array(glo); _GM0=np.array(gm0); _GT=np.array(gt)
    except Exception:
        _GLA=np.array([]); _GLO=np.array([]); _GM0=np.array([]); _GT=np.array([])

def _hav(lat,lon,lats,lons):
    rlat,rlon=np.radians(lat),np.radians(lon); rla,rlo=np.radians(lats),np.radians(lons)
    dlon=rlo-rlon
    a=np.sin((rla-rlat)/2)**2+np.cos(rlat)*np.cos(rla)*np.sin(dlon/2)**2
    return 6371.0*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def _feat(arg):
    lat,lon,ref = arg
    mag=globals()['_MAG']
    # local window: events within ~6deg box, then radius filter
    box=((_T<ref)&(np.abs(_LAT-lat)<6)&(np.abs(_LON-lon)<6))
    bi=np.where(box)[0]
    out=dict(bval=np.nan, amr_ratio=np.nan, amr_curv=np.nan, nt_k1=np.nan,
             coulomb=np.nan, tidal_c=np.nan, tidal_fn=np.nan)
    if bi.size:
        d=_hav(lat,lon,_LAT[bi],_LON[bi]); days=(ref-_T[bi])/SEC_DAY
        # --- b-value (Aki MLE) within 150km / 5yr ---
        sel=(d<150)&(days<5*365)
        mm=mag[bi][sel]
        if mm.size>=25:
            mc=mm.min()
            mean_m=mm.mean()
            if mean_m>mc: out["bval"]=np.log10(np.e)/(mean_m-(mc-0.05))
        # --- AMR: Benioff strain accel within 250km / 3yr ---
        sa=(d<250)&(days<3*365)
        if sa.sum()>=15:
            ee=np.sqrt(_E[bi][sa]); tt=ref-_T[bi][sa]  # seconds before ref
            order=np.argsort(-tt)                       # oldest..newest
            S=np.cumsum(ee[order]); tau=(tt[order])/SEC_DAY  # days-before, descending
            x=(-tau)                                    # increasing time
            # recent-half vs older-half Benioff rate ratio
            half=len(S)//2
            r_old=S[half]-S[0]; r_new=S[-1]-S[half]
            out["amr_ratio"]=np.log1p(r_new)-np.log1p(r_old+1e-9)
            # curvature: residual of last point vs linear fit of S over time (normalized)
            A=np.vstack([x,np.ones_like(x)]).T
            coef,_,_,_=np.linalg.lstsq(A,S,rcond=None)
            pred=A@coef; resid=S-pred
            out["amr_curv"]=float(resid[-1]/(S[-1]+1e-9))   # >0: recent above-trend (accel)
        # --- natural-time kappa1 (last 50 events within 150km) ---
        sn=(d<150)
        if sn.sum()>=20:
            tn=_T[bi][sn]; en=_E[bi][sn]
            o=np.argsort(tn)[-50:]                       # most recent up to 50
            e=en[o]; N=len(e); chi=np.arange(1,N+1)/N
            p=e/e.sum()
            k1=float((p*chi**2).sum()-((p*chi).sum())**2)
            out["nt_k1"]=k1
        # --- causal Coulomb stress proxy from prior GCMT M6+ within 500km ---
        if _GLA.size:
            causal=_GT<ref
            gm0=_GM0[causal]
            gd=_hav(lat,lon,_GLA[causal],_GLO[causal]) if causal.any() else np.array([])
            ing=gd<500
            if ing.any():
                r_m=np.maximum(gd[ing]*1000.0,5000.0)    # meters, floor 5km
                out["coulomb"]=float(np.log1p((gm0[ing]/r_m**3).sum()))
    # --- tidal phase (lunar synodic + fortnightly) ---
    ref_new_moon=dt.datetime(2000,1,6,18,14,tzinfo=dt.timezone.utc).timestamp()
    phase=((ref-ref_new_moon)/(29.53059*SEC_DAY))%1.0
    out["tidal_c"]=np.cos(2*np.pi*phase)
    out["tidal_fn"]=np.cos(4*np.pi*phase)               # fortnightly
    return out

def main():
    print(f"loading {os.path.basename(NPZ)} ...", flush=True)
    z=np.load(NPZ); X=z["X"]; Y=z["Y"].astype(int); T=z["T"]
    lat=X[:,-1,6]*90.0; lon=X[:,-1,7]*180.0
    ctx=X[:,-1,6:20].astype(np.float32)                 # 14 existing context features
    print(f"  {len(Y)} samples, {Y.sum()} pos ({Y.mean():.2%}); recovered lat/lon/ref", flush=True)
    args=list(zip(lat,lon,T))
    feats_keys=["bval","amr_ratio","amr_curv","nt_k1","coulomb","tidal_c","tidal_fn"]
    cache=os.path.join(os.path.dirname(NPZ),"catalog_features_incremental_causal.npz")
    if os.path.exists(cache):
        F=np.load(cache)["F"]; print("  [cache] loaded features", flush=True)
    else:
        F=np.full((len(args),len(feats_keys)),np.nan,np.float32)
        t0=time.time()
        with ProcessPoolExecutor(max_workers=8, initializer=_winit) as ex:
            for i,d in enumerate(ex.map(_feat,args,chunksize=128)):
                F[i]=[d[k] for k in feats_keys]
                if (i+1)%20000==0:
                    rate=(i+1)/(time.time()-t0)
                    print(f"    {i+1}/{len(args)} ({rate:.0f}/s, ETA {(len(args)-i)/rate/60:.1f}min)",flush=True)
        np.savez_compressed(cache,F=F)
        print(f"  computed features in {(time.time()-t0)/60:.1f}min",flush=True)

    tr=T<VAL0; te=T>=TEST0
    print(f"\n  train {tr.sum()} | test {te.sum()} ({Y[te].sum()} pos)",flush=True)
    # univariate operational AUC on TEST (impute nan to median; flip sign so higher=more hazard)
    print("\n  UNIVARIATE operational AUC on held-out test (0.5=no signal):",flush=True)
    med=np.nanmedian(F[tr],axis=0)
    for j,k in enumerate(feats_keys):
        col=F[:,j].copy(); col[np.isnan(col)]=med[j]
        a=auc(Y[te],col[te])
        a2=max(a,1-a) if not np.isnan(a) else a
        direction = "+" if a>=0.5 else "-"
        print(f"    {k:11s}: AUC {a:.4f}  (|signal| {a2:.4f}, dir {direction})",flush=True)

    # GBT incremental test
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    except Exception:
        print("  sklearn unavailable -- skipping GBT incremental test"); return
    def fill(a):
        b=a.copy()
        for j in range(b.shape[1]):
            c=b[:,j]; c[np.isnan(c)]=med[j] if j<len(med) else np.nanmedian(c)
        return b
    Cf=ctx; Ff=fill(F)
    base_X=Cf; aug_X=np.hstack([Cf,Ff])
    def fit_auc(Xtr,Xte):
        m=HGB(max_iter=300,learning_rate=0.06,max_depth=3,l2_regularization=1.0,
              early_stopping=True,validation_fraction=0.15,random_state=0)
        m.fit(Xtr[tr],Y[tr])
        return auc(Y[te],m.predict_proba(Xte[te])[:,1])
    print("\n  GBT INCREMENTAL test (operational AUC on test):",flush=True)
    a_base=fit_auc(base_X,base_X)
    a_aug =fit_auc(aug_X,aug_X)
    print(f"    context-14 only        : {a_base:.4f}",flush=True)
    print(f"    context-14 + new feats : {a_aug:.4f}   (delta {a_aug-a_base:+.4f})",flush=True)
    # which new feature group helps most: add one at a time
    print("\n  marginal add (context-14 + ONE new feature):",flush=True)
    for j,k in enumerate(feats_keys):
        xj=np.hstack([Cf,Ff[:,[j]]])
        aj=fit_auc(xj,xj)
        print(f"    +{k:11s}: {aj:.4f}  (delta {aj-a_base:+.4f})",flush=True)
    print("\n  NOTE: context-14 GBT approximates (does not equal) the deep GRU model (~0.73);",flush=True)
    print("  the INCREMENTAL delta is the honest 'do these classic precursors add signal' test.",flush=True)

if __name__=="__main__":
    main()
