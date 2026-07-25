#!/usr/bin/env python3
"""ETAS benchmark: does the deep operational model BEAT the gold-standard statistical
aftershock forecaster (Ogata ETAS)? Same operational task, same cached sample set.

ETAS conditional intensity at (cell, t):
  lambda = mu_bg(cell) + SUM_{i: t_i<t, r_i<R} K * 10^(alpha*(M_i - M0)) / ((t-t_i)/day + c)^p
We rank active cells by the 30-day expected count and report operational AUC vs the deep 0.73.
"""
import sys, os, time, datetime as dt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, r"C:\Users\Josh\Projects\hazardpulse\src")
SEC_DAY=86400.0
NPZ=r"C:\Users\Josh\Projects\hazardpulse\.cache\earthquake\deepop_v3_my2025_m5.0_lr100_ld30_K192_ir100_am8_g2.npz"
VAL0=dt.datetime(2018,1,1,tzinfo=dt.timezone.utc).timestamp()
TEST0=dt.datetime(2020,1,1,tzinfo=dt.timezone.utc).timestamp()
# Ogata ETAS params (typical global-ish values; M0=completeness)
K,ALPHA,C,P,M0 = 0.018, 0.9, 0.01, 1.10, 2.5
R_KM=100.0

def auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float)
    if y.sum()==0 or y.sum()==len(y):return float("nan")
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(1,len(s)+1);n1=y.sum();n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

_LAT=_LON=_T=_MAG=None
def _winit():
    global _LAT,_LON,_T,_MAG
    from hazardpulse.earthquake.definitive_model import load_usgs_catalog, CatalogArrays
    cat=CatalogArrays(load_usgs_catalog(min_year=2000,max_year=2025,min_mag=2.5),verbose=False)
    _LAT,_LON,_T,_MAG=cat.lats,cat.lons,cat.times,cat.mags

def _hav(lat,lon,lats,lons):
    rlat,rlon=np.radians(lat),np.radians(lon);rla,rlo=np.radians(lats),np.radians(lons);dlon=rlo-rlon
    a=np.sin((rla-rlat)/2)**2+np.cos(rlat)*np.cos(rla)*np.sin(dlon/2)**2
    return 6371.0*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def _etas(arg):
    lat,lon,ref=arg
    box=((_T<ref)&(np.abs(_LAT-lat)<3)&(np.abs(_LON-lon)<3))
    bi=np.where(box)[0]
    if bi.size==0: return 0.0
    d=_hav(lat,lon,_LAT[bi],_LON[bi]); inr=d<R_KM
    if not inr.any(): return 0.0
    days=(ref-_T[bi][inr])/SEC_DAY; mg=_MAG[bi][inr]
    days=np.maximum(days,0.0)
    # aftershock intensity (per day) at ref
    aft=(K*10**(ALPHA*(mg-M0))/((days+C)**P)).sum()
    # background daily rate from 1-5yr window
    bg=((days>=365)&(days<5*365)).sum()/(4*365.0)
    lam=bg+aft
    # 30-day expected count ~ lam*30 (first-order; ranking-invariant to the constant)
    return float(np.log1p(lam*30.0))

def main():
    z=np.load(NPZ);X=z["X"];Y=z["Y"].astype(int);T=z["T"]
    lat=X[:,-1,6]*90.0; lon=X[:,-1,7]*180.0
    args=list(zip(lat,lon,T))
    print(f"{len(Y)} samples; computing ETAS intensity (parallel)...",flush=True)
    S=np.zeros(len(args)); t0=time.time()
    with ProcessPoolExecutor(max_workers=4,initializer=_winit) as ex:
        for i,v in enumerate(ex.map(_etas,args,chunksize=256)):
            S[i]=v
            if (i+1)%20000==0: print(f"  {i+1}/{len(args)} ({(i+1)/(time.time()-t0):.0f}/s)",flush=True)
    te=T>=TEST0
    a=auc(Y[te],S[te])
    print(f"\nETAS operational AUC (held-out test): {a:.4f}",flush=True)
    print(f"  deep operational model: ~0.73 | context-14 GBT: 0.719 | climatology baseline ~0.57-0.60",flush=True)
    print(f"  -> deep model {'BEATS' if 0.73>a+0.01 else 'ties/loses to'} ETAS by ~{0.73-a:+.3f}",flush=True)
    # also: does ETAS add to the deep context? (stack)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        ctx=X[:,-1,6:20].astype(np.float32); tr=T<VAL0
        def fit(XX):
            m=HGB(max_iter=300,learning_rate=0.06,max_depth=3,l2_regularization=1.0,
                  early_stopping=True,validation_fraction=0.15,random_state=0);m.fit(XX[tr],Y[tr])
            return auc(Y[te],m.predict_proba(XX[te])[:,1])
        b=fit(ctx); g=fit(np.hstack([ctx,S[:,None]]))
        print(f"\n  context-14 GBT {b:.4f} -> +ETAS {g:.4f} (delta {g-b:+.4f})",flush=True)
    except Exception as e:
        print("  (sklearn step skipped:",e,")")

if __name__=="__main__":
    main()
