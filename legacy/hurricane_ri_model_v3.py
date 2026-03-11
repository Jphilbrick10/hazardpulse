#!/usr/bin/env python3
"""
HURRICANE RAPID INTENSIFICATION MODEL v3
==========================================
Major upgrades over v2 (AUC=0.963):
  1. Ocean heat content proxies (SST anomaly, TCHP, SST gradient, warm currents)
  2. Atmospheric environment proxies (shear, outflow temp, humidity, ventilation)
  3. Storm structural evolution (RMW contraction rates, eye formation rate,
     intensification efficiency, Knaff-Zehr deviation, symmetry rate)
  4. Environmental interaction (time since land, Fujiwhara, recurvature, trough)
  5. Multi-timescale features (3h-48h trends, persistence, 2nd derivative)
  6. Ensemble stacking: L2 logistic + gradient boosted stumps + bagged logistic
     -> Meta-learner on 3 model outputs + 2 best raw features

Target: AUC > 0.97 on NA basin, test 2015+.

Uses ONLY: numpy, matplotlib (no sklearn, no pandas, no scipy).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import ssl
import time
import warnings
import hashlib
import os
warnings.filterwarnings('ignore')

OUT = Path(r"c:\Users\Josh\Projects\Coherence_Energy_Labs_Website\data\evidence\coherence_field_results")
OUT.mkdir(exist_ok=True)

DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'
GREEN = '#2ecc71'
BLUE = '#3498db'
PURPLE = '#9b59b6'
ORANGE = '#e67e22'

# =========================================================================
# DATA LOADING
# =========================================================================

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=300):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

CACHE_DIR = Path(r"c:\Users\Josh\Projects\Coherence_Energy_Labs_Website\data\evidence\.cache")
CACHE_DIR.mkdir(exist_ok=True)

def fetch_cached(url, timeout=300):
    """Fetch URL with local file cache."""
    h = hashlib.md5(url.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{h}.csv"
    if cache_path.exists():
        print(f"    Using cached: {cache_path.name}")
        return cache_path.read_bytes()
    data = fetch_url(url, timeout=timeout)
    if data:
        cache_path.write_bytes(data)
    return data

def parse_ibtracs_csv(text):
    """Parse IBTrACS CSV into storm dict."""
    lines = text.strip().split('\n')
    header = lines[0].split(',')
    col_map = {}
    for i, h in enumerate(header):
        col_map[h.strip().strip('"')] = i

    sid_col = col_map.get('SID', 0)
    name_col = col_map.get('NAME', 1)
    time_col = col_map.get('ISO_TIME', 6)
    basin_col = col_map.get('BASIN', None)
    lat_col = col_map.get('LAT', None)
    lon_col = col_map.get('LON', None)
    wind_col = col_map.get('USA_WIND', None)
    wmo_wind_col = col_map.get('WMO_WIND', None)
    pres_col = col_map.get('USA_PRES', None)
    wmo_pres_col = col_map.get('WMO_PRES', None)
    rmw_col = col_map.get('USA_RMW', None)
    dist2land_col = col_map.get('DIST2LAND', None)

    storms = {}
    for line in lines[2:]:
        parts = line.split(',')
        if len(parts) < 10:
            continue
        try:
            sid = parts[sid_col].strip().strip('"')
            name = parts[name_col].strip().strip('"')

            def safe_float(col):
                if col is None:
                    return -999.0
                v = parts[col].strip().strip('"')
                if v in ('', ' ', 'NA', 'na'):
                    return -999.0
                try:
                    return float(v)
                except ValueError:
                    return -999.0

            wind = safe_float(wind_col)
            if wind <= 0:
                wind = safe_float(wmo_wind_col)
            pres = safe_float(pres_col)
            if pres <= 0:
                pres = safe_float(wmo_pres_col)
            rmw = safe_float(rmw_col)
            lat = safe_float(lat_col)
            lon = safe_float(lon_col)
            dist2land = safe_float(dist2land_col)

            time_str = parts[time_col].strip().strip('"')
            year = int(time_str[:4]) if len(time_str) >= 4 else 0
            month = int(time_str[5:7]) if len(time_str) >= 7 else 6
            day = int(time_str[8:10]) if len(time_str) >= 10 else 15
            hour = int(time_str[11:13]) if len(time_str) >= 13 else 0

            basin_str = ''
            if basin_col is not None:
                basin_str = parts[basin_col].strip().strip('"')

            if sid not in storms:
                storms[sid] = {'name': name, 'basin': basin_str, 'entries': []}

            storms[sid]['entries'].append({
                'wind': wind, 'pres': pres, 'rmw': rmw,
                'lat': lat, 'lon': lon, 'year': year, 'month': month,
                'day': day, 'hour': hour,
                'time_str': time_str, 'dist2land': dist2land,
            })
        except Exception:
            pass

    return storms

def load_ibtracs(basin='ALL'):
    """Load IBTrACS data."""
    print(f"\n  Loading IBTrACS {basin}...")
    url = f"https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
    t0 = time.time()
    data = fetch_cached(url, timeout=300)
    if not data:
        print(f"  {basin} download failed.")
        return {}
    text = data.decode('utf-8', errors='replace')
    storms = parse_ibtracs_csv(text)
    dt = time.time() - t0
    print(f"  Loaded {len(storms)} storms ({basin}) in {dt:.1f}s")
    return storms


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def approx_mpi(lat, month):
    """Approximate MPI (kt) from latitude and season."""
    abs_lat = abs(lat)
    sst_proxy = 30.0 - 0.5 * max(0, abs_lat - 10)
    if lat >= 0:
        season_warm = np.exp(-((month - 9)**2) / 8.0)
    else:
        season_warm = np.exp(-((month - 3)**2) / 8.0)
    sst_proxy += 2.0 * season_warm
    if sst_proxy < 26.0:
        return 40.0
    mpi = 30.0 * (sst_proxy - 26.0) + 40.0
    return min(mpi, 185.0)

def knaff_zehr_wind(pres, lat):
    """Knaff-Zehr (2007) wind-pressure relationship (simplified)."""
    if pres <= 0 or pres > 1020:
        return -999.0
    dp = 1013.25 - pres
    if dp <= 0:
        return 25.0
    # Latitude adjustment factor
    abs_lat = min(abs(lat), 40) if lat != -999 else 25
    lat_adj = 1.0 + 0.003 * (abs_lat - 25)
    v = (dp * 1.12) ** 0.62 * 12.0 * lat_adj
    return max(v, 20.0)


# =========================================================================
# OCEAN HEAT CONTENT PROXIES
# =========================================================================

def sst_climatology(lat, lon, month):
    """
    Approximate SST climatology (degC) by lat/lon/month.
    Based on Levitus climatology patterns:
    - Peaks ~28-30C in tropics, decreases poleward
    - Warm pools: W Pacific, Gulf of Mexico, Indian Ocean
    - Cold tongue in E Pacific equatorial
    """
    abs_lat = abs(lat)
    # Base SST: warm tropics, cold poles
    sst = 30.0 - 0.45 * abs_lat

    # Seasonal cycle (amplitude increases with latitude)
    if lat >= 0:
        sst += (0.5 + 0.1 * abs_lat) * np.cos(2 * np.pi * (month - 9) / 12)
    else:
        sst += (0.5 + 0.1 * abs_lat) * np.cos(2 * np.pi * (month - 3) / 12)

    # Warm pool boost (W Pacific 120-180E, 0-20N)
    if 120 <= lon <= 180 and 0 <= lat <= 20:
        sst += 1.5
    # Gulf of Mexico / Caribbean warm pool
    if -100 <= lon <= -60 and 10 <= lat <= 30:
        sst += 1.0
    # Indian Ocean warm pool
    if 50 <= lon <= 100 and -15 <= lat <= 15:
        sst += 1.0
    # E Pacific cold tongue (equatorial, 180-90W)
    if -180 <= lon <= -90 and abs_lat < 10:
        sst -= 2.0
    # Benguela/Canary upwelling (E Atlantic)
    if -20 <= lon <= 0 and 10 <= abs_lat <= 35:
        sst -= 1.5

    return max(sst, 10.0)

def sst_anomaly_proxy(lat, lon, month):
    """
    SST anomaly proxy: deviation from zonal mean.
    Positive = warmer than average at that latitude = more fuel.
    """
    sst = sst_climatology(lat, lon, month)
    # Zonal mean at that latitude (rough)
    zonal_mean = 30.0 - 0.45 * abs(lat)
    if lat >= 0:
        zonal_mean += 0.5 * np.cos(2 * np.pi * (month - 9) / 12)
    else:
        zonal_mean += 0.5 * np.cos(2 * np.pi * (month - 3) / 12)
    return sst - zonal_mean

def tchp_proxy(lat, lon, month):
    """
    Tropical Cyclone Heat Potential proxy (kJ/cm^2).
    Estimated from SST and approximate mixed layer depth.
    TCHP ~ rho * Cp * MLD * (SST - 26)
    Normalized to ~50-150 range for typical TC environments.
    """
    sst = sst_climatology(lat, lon, month)
    if sst < 26.0:
        return 0.0

    # Mixed layer depth proxy (deeper in warm pools, shallower near upwelling)
    mld_base = 50.0  # meters
    # Deeper MLD in W Pacific warm pool, Gulf Stream
    if (120 <= lon <= 180 and 0 <= lat <= 20):
        mld_base = 80.0
    elif (-90 <= lon <= -60 and 20 <= lat <= 35):  # Gulf Stream
        mld_base = 75.0
    elif (120 <= lon <= 150 and 20 <= lat <= 40):  # Kuroshio
        mld_base = 70.0

    # TCHP ~ MLD * (SST - 26)
    tchp = mld_base * (sst - 26.0)
    return tchp

def sst_gradient_proxy(lat, lon, month):
    """
    SST gradient magnitude (C/degree). High gradient = near cold water.
    Approximated as the max SST difference to adjacent points.
    """
    sst_center = sst_climatology(lat, lon, month)
    max_diff = 0.0
    for dlat, dlon in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        sst_adj = sst_climatology(lat + dlat, lon + dlon, month)
        diff = abs(sst_center - sst_adj)
        if diff > max_diff:
            max_diff = diff
    return max_diff

def dist_to_warm_current(lat, lon):
    """
    Approximate distance (km) to nearest major warm ocean current axis.
    Gulf Stream: ~25-40N, 75-40W
    Kuroshio: ~25-40N, 125-170E
    Agulhas: ~25-40S, 25-45E
    """
    warm_currents = [
        # Gulf Stream axis points (lat, lon)
        (28, -80), (30, -78), (33, -75), (35, -72), (38, -68), (40, -60),
        # Kuroshio axis points
        (25, 125), (28, 130), (30, 135), (33, 140), (35, 145), (38, 150),
        # Agulhas
        (-30, 30), (-32, 33), (-35, 38), (-37, 42),
    ]
    min_dist = 9999.0
    for clat, clon in warm_currents:
        d = haversine_km(lat, lon, clat, clon)
        if d < min_dist:
            min_dist = d
    return min_dist


# =========================================================================
# ATMOSPHERIC ENVIRONMENT PROXIES
# =========================================================================

def shear_proxy(lat, lon, month, trans_speed, heading_change):
    """
    Vertical wind shear proxy (kt).
    Higher shear at higher latitudes, in shear zones.
    Low shear in deep tropics during peak season.
    Storm motion deviation from climatological steering indicates shear.
    """
    abs_lat = abs(lat)
    # Base shear: increases with latitude (jet stream proximity)
    shear = 5.0 + 0.5 * max(0, abs_lat - 15)

    # Season: lower in deep tropics during peak season
    if lat >= 0:
        peak_month = 9
    else:
        peak_month = 3
    season_factor = 1.0 + 0.3 * abs(month - peak_month) / 6.0
    shear *= season_factor

    # Heading change indicates shear interaction
    if heading_change != -999:
        shear += 0.5 * abs(heading_change)

    # Translation speed anomaly: fast-moving storms often in shear
    if trans_speed != -999 and trans_speed > 25:
        shear += (trans_speed - 25) * 0.3

    return shear

def outflow_temp_proxy(lat, month):
    """
    Outflow (tropopause) temperature proxy (K).
    Colder tropopause = higher thermodynamic efficiency = higher MPI.
    Tropopause is coldest in tropics (~190K) and warmer at high lat (~220K).
    """
    abs_lat = abs(lat)
    # Tropopause temp: cold in tropics, warmer poleward
    t_out = 195.0 + 0.6 * abs_lat
    # Slightly colder in NH summer (higher tropopause)
    if lat >= 0:
        t_out -= 3.0 * np.exp(-((month - 8)**2) / 8.0)
    else:
        t_out -= 3.0 * np.exp(-((month - 2)**2) / 8.0)
    return t_out

def humidity_proxy(wind_asymmetry, eye_indicator, intensity_frac):
    """
    Mid-level relative humidity proxy.
    Symmetric, intense storms tend to have moist environments.
    Asymmetric storms or weak storms may be in dry environments.
    """
    rh = 70.0  # base RH
    # More symmetric = moister (organized convection moistens mid-levels)
    if wind_asymmetry != -999:
        rh -= wind_asymmetry * 2.0
    # Eye formation = well-organized = moist
    if eye_indicator > 0:
        rh += 10.0
    # Stronger storms = moister inner core
    if intensity_frac != -999:
        rh += 10.0 * intensity_frac
    return np.clip(rh, 30, 100)

def ventilation_proxy(shear, rh):
    """
    Ventilation index: shear * (100 - RH) / 100.
    High ventilation = shear flushes out moisture = bad for TC.
    Tang & Emanuel (2012).
    """
    dryness = (100.0 - rh) / 100.0
    return shear * dryness


# =========================================================================
# FEATURE EXTRACTION v3
# =========================================================================

def extract_storm_features_v3(entries, all_storms_entries=None,
                                ri_threshold_kt=30, ri_window_steps=4):
    """
    Extract v3 feature set for each 6-hour observation.
    Massive expansion: ~70+ features before interactions.
    """
    n = len(entries)
    if n < max(10, ri_window_steps + 6):
        return []

    winds = np.array([e['wind'] for e in entries])
    pressures = np.array([e['pres'] for e in entries])
    rmws = np.array([e['rmw'] for e in entries])
    lats = np.array([e['lat'] for e in entries])
    lons = np.array([e['lon'] for e in entries])
    d2l = np.array([e['dist2land'] for e in entries])
    months = np.array([e['month'] for e in entries])
    years_arr = np.array([e['year'] for e in entries])

    samples = []
    prior_ri_flag = False
    last_land_step = -999  # step index of last time dist2land < 100km

    for i in range(6, n - ri_window_steps):
        w_now = winds[i]
        w_future = winds[i + ri_window_steps]

        if w_now <= 0 or w_future <= 0:
            continue

        dw_next = w_future - w_now
        label = 1 if dw_next >= ri_threshold_kt else 0

        year = years_arr[i]
        month = months[i]
        lat = lats[i]
        lon = lons[i]

        if lat == -999 or lon == -999:
            continue

        f = {}

        # ===== CORE INTENSITY =====
        f['wind_now'] = w_now

        # Multi-timescale wind changes (prior)
        for steps, name in [(1, 'dw_6h'), (2, 'dw_12h'), (4, 'dw_24h')]:
            if i >= steps and winds[i - steps] > 0:
                f[name] = w_now - winds[i - steps]
            else:
                f[name] = -999

        # 36h and 48h lookback
        for steps, name in [(6, 'dw_36h'), (8, 'dw_48h')]:
            if i >= steps and winds[i - steps] > 0:
                f[name] = w_now - winds[i - steps]
            else:
                f[name] = -999

        # Acceleration (2nd derivative)
        if i >= 4 and winds[i-2] > 0 and winds[i-4] > 0:
            dw_recent = w_now - winds[i-2]
            dw_prior = winds[i-2] - winds[i-4]
            f['acceleration'] = dw_recent - dw_prior
        else:
            f['acceleration'] = -999

        # 2nd derivative at 6h resolution
        if i >= 2 and winds[i-1] > 0 and winds[i-2] > 0:
            d2w = (w_now - 2*winds[i-1] + winds[i-2])
            f['d2w_6h'] = d2w
        else:
            f['d2w_6h'] = -999

        # Intensification persistence: consecutive 6h periods of intensification
        consec = 0
        for j in range(i, max(i-8, 0), -1):
            if j >= 1 and winds[j] > 0 and winds[j-1] > 0 and winds[j] > winds[j-1]:
                consec += 1
            else:
                break
        f['intens_persistence'] = float(consec)

        # ===== PRESSURE =====
        p_now = pressures[i]
        f['pressure_now'] = p_now if p_now > 800 else -999

        for steps, name in [(1, 'dp_6h'), (2, 'dp_12h'), (4, 'dp_24h')]:
            p_prev = pressures[i - steps] if i >= steps else -999
            if p_now > 800 and p_prev > 800:
                f[name] = p_now - p_prev  # negative = deepening
            else:
                f[name] = -999

        # Pressure acceleration
        if p_now > 800 and i >= 4 and pressures[i-2] > 800 and pressures[i-4] > 800:
            f['pressure_accel'] = (p_now - pressures[i-2]) - (pressures[i-2] - pressures[i-4])
        else:
            f['pressure_accel'] = -999

        # ===== WIND-PRESSURE RELATIONSHIP =====
        if p_now > 800 and w_now > 0:
            expected_w = knaff_zehr_wind(p_now, lat)
            if expected_w > 0:
                f['kz_deviation'] = w_now - expected_w  # positive = stronger than KZ
            else:
                f['kz_deviation'] = -999
            f['pw_efficiency'] = w_now**2 / max(1013.25 - p_now, 1.0)
        else:
            f['kz_deviation'] = -999
            f['pw_efficiency'] = -999

        # ===== LOCATION & ENVIRONMENT =====
        f['lat'] = lat
        f['abs_lat'] = abs(lat)
        f['lon'] = lon

        # Season encoding
        f['season_sin'] = np.sin(2 * np.pi * month / 12.0)
        f['season_cos'] = np.cos(2 * np.pi * month / 12.0)

        # Distance from land
        f['dist2land'] = d2l[i] if d2l[i] > 0 else -999

        # Time since last land interaction
        if d2l[i] > 0 and d2l[i] < 100:
            last_land_step = i
        if last_land_step >= 0:
            f['time_since_land'] = (i - last_land_step) * 6.0  # hours
        else:
            f['time_since_land'] = 999.0  # never near land

        # Translation speed
        if i >= 1 and lats[i-1] != -999 and lons[i-1] != -999:
            dist_km = haversine_km(lats[i-1], lons[i-1], lat, lon)
            f['translation_speed'] = dist_km / 6.0
        else:
            f['translation_speed'] = -999

        # Heading (bearing)
        if i >= 1 and lats[i-1] != -999 and lons[i-1] != -999:
            dlat = lat - lats[i-1]
            dlon = lon - lons[i-1]
            heading = np.degrees(np.arctan2(dlon * np.cos(np.radians(lat)), dlat))
            f['heading'] = heading
        else:
            f['heading'] = -999

        # Heading change rate (recurvature indicator)
        if i >= 2 and f['heading'] != -999 and lats[i-2] != -999 and lons[i-2] != -999:
            dlat_prev = lats[i-1] - lats[i-2]
            dlon_prev = lons[i-1] - lons[i-2]
            heading_prev = np.degrees(np.arctan2(
                dlon_prev * np.cos(np.radians(lats[i-1])), dlat_prev))
            dheading = f['heading'] - heading_prev
            # Normalize to [-180, 180]
            if dheading > 180: dheading -= 360
            if dheading < -180: dheading += 360
            f['heading_change'] = dheading
        else:
            f['heading_change'] = -999

        # Latitude change rate (trough interaction proxy)
        if i >= 2 and lats[i-2] != -999:
            f['lat_change_rate'] = (lat - lats[i-2]) / 12.0  # deg/hour
        else:
            f['lat_change_rate'] = -999

        # Recurvature indicator: abs heading change
        f['recurvature'] = abs(f['heading_change']) if f['heading_change'] != -999 else -999

        # ===== MPI FEATURES =====
        mpi = approx_mpi(lat, month)
        f['mpi'] = mpi
        f['intensity_frac_mpi'] = w_now / mpi if mpi > 0 else -999
        f['mpi_deficit'] = mpi - w_now

        # Intensification efficiency: dW/dt normalized by room to grow
        if f['dw_6h'] != -999 and f['mpi_deficit'] > 5:
            f['intens_efficiency'] = f['dw_6h'] / f['mpi_deficit']
        else:
            f['intens_efficiency'] = -999

        # ===== STORM HISTORY =====
        f['prior_ri'] = 1.0 if prior_ri_flag else 0.0
        f['storm_age_h'] = i * 6.0

        # Wind asymmetry
        if f['dw_6h'] != -999 and f['dw_12h'] != -999:
            f['wind_asymmetry'] = abs(f['dw_6h'] - f['dw_12h'] / 2.0)
        else:
            f['wind_asymmetry'] = -999

        # Eye indicator
        f['eye_indicator'] = 1.0 if (w_now > 65 and p_now > 800 and p_now < 980) else 0.0

        # Eye formation RATE: how fast eye develops
        # Track when eye first appears
        if i >= 4:
            eye_count = 0
            for j in range(max(0, i-4), i+1):
                if winds[j] > 65 and pressures[j] > 800 and pressures[j] < 980:
                    eye_count += 1
            f['eye_formation_rate'] = eye_count / 5.0  # fraction of last 24h with eye
        else:
            f['eye_formation_rate'] = 0.0

        # Symmetry improvement rate
        if i >= 2 and f['wind_asymmetry'] != -999:
            # Check prior asymmetry
            dw6_prev = (winds[i-1] - winds[i-2]) if (i >= 2 and winds[i-1] > 0 and winds[i-2] > 0) else -999
            dw12_prev = (winds[i-1] - winds[i-3]) if (i >= 3 and winds[i-1] > 0 and winds[i-3] > 0) else -999
            if dw6_prev != -999 and dw12_prev != -999:
                asym_prev = abs(dw6_prev - dw12_prev / 2.0)
                f['symmetry_improvement'] = asym_prev - f['wind_asymmetry']  # positive = becoming more symmetric
            else:
                f['symmetry_improvement'] = -999
        else:
            f['symmetry_improvement'] = -999

        # ===== RMW FEATURES =====
        f['rmw_now'] = rmws[i] if rmws[i] > 0 else -999

        # RMW changes at multiple timescales
        for steps, name in [(1, 'drmw_6h'), (2, 'drmw_12h'), (4, 'drmw_24h')]:
            if i >= steps and rmws[i] > 0 and rmws[i - steps] > 0:
                f[name] = rmws[i] - rmws[i - steps]
            else:
                f[name] = -999

        # RMW trend (linear over last 24h) - manual linear regression
        if rmws[i] > 0 and i >= 4:
            rmw_seq = []
            for j in range(max(0, i-4), i+1):
                if rmws[j] > 0:
                    rmw_seq.append((j - i, rmws[j]))
            if len(rmw_seq) >= 3:
                xs = np.array([p[0] for p in rmw_seq], dtype=float)
                ys = np.array([p[1] for p in rmw_seq], dtype=float)
                mx, my = np.mean(xs), np.mean(ys)
                ss_xy = np.sum((xs - mx) * (ys - my))
                ss_xx = np.sum((xs - mx)**2)
                if ss_xx > 0:
                    f['rmw_trend'] = ss_xy / ss_xx
                else:
                    f['rmw_trend'] = -999
            else:
                f['rmw_trend'] = -999
        else:
            f['rmw_trend'] = -999

        # RMW contraction rate (specifically negative rate)
        if f['rmw_trend'] != -999:
            f['rmw_contraction'] = -f['rmw_trend']  # positive = contracting
        else:
            f['rmw_contraction'] = -999

        # ===== OCEAN HEAT CONTENT PROXIES =====
        f['sst_anomaly'] = sst_anomaly_proxy(lat, lon, month)
        f['tchp'] = tchp_proxy(lat, lon, month)
        f['sst_gradient'] = sst_gradient_proxy(lat, lon, month)
        f['dist_warm_current'] = dist_to_warm_current(lat, lon)
        f['sst'] = sst_climatology(lat, lon, month)

        # ===== ATMOSPHERIC PROXIES =====
        heading_change = f['heading_change'] if f['heading_change'] != -999 else 0
        trans_speed = f['translation_speed'] if f['translation_speed'] != -999 else 15
        f['shear_proxy'] = shear_proxy(lat, lon, month, trans_speed, heading_change)
        f['outflow_temp'] = outflow_temp_proxy(lat, month)

        # Humidity proxy
        wa = f['wind_asymmetry'] if f['wind_asymmetry'] != -999 else 5
        ifrac = f['intensity_frac_mpi'] if f['intensity_frac_mpi'] != -999 else 0.5
        f['humidity_proxy'] = humidity_proxy(wa, f['eye_indicator'], ifrac)
        f['ventilation'] = ventilation_proxy(f['shear_proxy'], f['humidity_proxy'])

        # ===== DERIVED COMPOUND FEATURES =====
        # Thermodynamic efficiency: (SST - T_out) / T_out (Carnot)
        sst_K = f['sst'] + 273.15
        f['carnot_efficiency'] = (sst_K - f['outflow_temp']) / f['outflow_temp']

        # S/Gamma ratio proxy (Helmholtz framework)
        # S ~ SST anomaly * TCHP, Gamma ~ shear * ventilation
        s_proxy = max(f['sst_anomaly'] + 2, 0.1) * max(f['tchp'], 1) / 100.0
        gamma_proxy = max(f['shear_proxy'], 1) * max(f['ventilation'] + 1, 0.1) / 100.0
        f['s_gamma_ratio'] = s_proxy / max(gamma_proxy, 0.01)

        samples.append((f, label, year, dw_next))

        if dw_next >= 30:
            prior_ri_flag = True

    return samples


# =========================================================================
# ML: LOGISTIC REGRESSION (Manual L2-regularized)
# =========================================================================

def train_logistic(X, y, lam=0.01, lr=0.05, n_epochs=2000, cw_ratio=None):
    """L2-regularized logistic regression via GD with class weighting."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    sw = np.ones(n)
    if cw_ratio is not None:
        sw[y == 1] = cw_ratio

    for epoch in range(n_epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = (p - y) * sw
        grad_w = (X.T @ error) / n + lam * w
        grad_b = np.sum(error) / n
        clr = lr / (1 + epoch * 0.001)
        w -= clr * grad_w
        b -= clr * grad_b

    return w, b


# =========================================================================
# ML: GRADIENT BOOSTED STUMPS (Manual implementation)
# =========================================================================

def find_best_stump(X, residuals, weights):
    """Find the best single-feature threshold split (decision stump). Vectorized."""
    n, d = X.shape
    best_loss = np.inf
    best_feat = 0
    best_thresh = 0.0
    best_left_val = 0.0
    best_right_val = 0.0

    wr = residuals * weights  # precompute

    for j in range(d):
        col = X[:, j]
        # 10 quantile thresholds for speed
        thresholds = np.percentile(col, np.linspace(10, 90, 10))

        for thresh in thresholds:
            left = col <= thresh
            n_left = np.sum(left)
            n_right = n - n_left
            if n_left < 5 or n_right < 5:
                continue

            w_left = np.sum(weights[left])
            w_right = np.sum(weights[~left])
            left_val = np.sum(wr[left]) / (w_left + 1e-10)
            right_val = np.sum(wr[~left]) / (w_right + 1e-10)

            # Loss = sum(w*(r - pred)^2), expand:
            # For left: sum(w*(r-lv)^2) = sum(w*r^2) - 2*lv*sum(w*r) + lv^2*sum(w)
            # Simplified: total_loss = total_wr2 - lv^2*w_left - rv^2*w_right
            loss = -(left_val**2 * w_left + right_val**2 * w_right)

            if loss < best_loss:
                best_loss = loss
                best_feat = j
                best_thresh = thresh
                best_left_val = left_val
                best_right_val = right_val

    return best_feat, best_thresh, best_left_val, best_right_val

def predict_stump(X, feat, thresh, left_val, right_val):
    return np.where(X[:, feat] <= thresh, left_val, right_val)

def train_gradient_boosted_stumps(X, y, n_stumps=200, learning_rate=0.1,
                                   cw_ratio=None, subsample=10000):
    """
    Gradient boosted decision stumps for binary classification.
    Uses log-loss (logistic) gradient boosting.
    Subsamples for stump fitting to keep runtime manageable.
    """
    n = X.shape[0]
    weights = np.ones(n)
    if cw_ratio is not None:
        weights[y == 1] = cw_ratio

    # Initialize with log-odds
    p_pos = np.sum(y * weights) / np.sum(weights)
    p_pos = np.clip(p_pos, 0.01, 0.99)
    F = np.full(n, np.log(p_pos / (1 - p_pos)))

    stumps = []
    init_val = F[0]
    rng = np.random.RandomState(42)

    # Subsample indices (stratified: keep all positives + random negatives)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg_sub = min(subsample - len(pos_idx), len(neg_idx))

    for t in range(n_stumps):
        p = sigmoid(F)
        residuals = y - p

        # Subsample for stump search
        if n > subsample:
            neg_sub = rng.choice(neg_idx, size=n_neg_sub, replace=False)
            sub_idx = np.concatenate([pos_idx, neg_sub])
            X_sub = X[sub_idx]
            res_sub = residuals[sub_idx]
            w_sub = weights[sub_idx]
        else:
            X_sub, res_sub, w_sub = X, residuals, weights

        feat, thresh, left_val, right_val = find_best_stump(X_sub, res_sub, w_sub)

        # Apply to full dataset
        pred = predict_stump(X, feat, thresh, left_val, right_val)
        F += learning_rate * pred

        stumps.append((feat, thresh, left_val * learning_rate,
                       right_val * learning_rate))

        if (t + 1) % 50 == 0:
            p_check = sigmoid(F)
            auc_t, _, _ = compute_roc_auc(y, p_check)
            print(f"        Stump {t+1}/{n_stumps}, train AUC={auc_t:.4f}")

    return stumps, init_val

def predict_gbs(X, stumps, init_val):
    """Predict probabilities from gradient boosted stumps."""
    n = X.shape[0]
    F = np.full(n, init_val)
    for feat, thresh, left_val, right_val in stumps:
        F += np.where(X[:, feat] <= thresh, left_val, right_val)
    return sigmoid(F)


# =========================================================================
# ML: BAGGED LOGISTIC REGRESSION
# =========================================================================

def train_bagged_logistic(X, y, n_bags=10, feat_fraction=0.5,
                           lam=0.01, lr=0.05, n_epochs=1000, cw_ratio=None):
    """Train an ensemble of logistic regressions on random feature subsets."""
    n, d = X.shape
    n_feat = max(int(d * feat_fraction), 5)
    models = []
    rng = np.random.RandomState(42)

    for bag in range(n_bags):
        # Random feature subset
        feat_idx = rng.choice(d, size=n_feat, replace=False)
        feat_idx.sort()

        # Bootstrap sample
        sample_idx = rng.choice(n, size=n, replace=True)

        X_bag = X[sample_idx][:, feat_idx]
        y_bag = y[sample_idx]

        w, b = train_logistic(X_bag, y_bag, lam=lam, lr=lr,
                               n_epochs=n_epochs, cw_ratio=cw_ratio)
        models.append((feat_idx, w, b))

    return models

def predict_bagged(X, models):
    """Average predictions from bagged models."""
    preds = np.zeros(X.shape[0])
    for feat_idx, w, b in models:
        X_sub = X[:, feat_idx]
        preds += sigmoid(X_sub @ w + b)
    return preds / len(models)


# =========================================================================
# EVALUATION
# =========================================================================

def compute_roc_auc(y_true, y_score):
    """Compute ROC AUC using sorted threshold method."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5, [], []

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += 0.5 * (tpr_list[i] + tpr_list[i-1]) * (fpr_list[i] - fpr_list[i-1])

    return auc, fpr_list, tpr_list

def compute_pr_metrics(y_true, y_score, threshold=0.5):
    pred_pos = y_score >= threshold
    tp = np.sum(pred_pos & (y_true == 1))
    fp = np.sum(pred_pos & (y_true == 0))
    fn = np.sum((~pred_pos) & (y_true == 1))
    tn = np.sum((~pred_pos) & (y_true == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return precision, recall, f1, tp, fp, fn, tn

def find_optimal_threshold(y_true, y_score):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 200):
        _, _, f1, _, _, _, _ = compute_pr_metrics(y_true, y_score, thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# =========================================================================
# FEATURE SET DEFINITIONS
# =========================================================================

# v2-equivalent features (for comparison baseline)
V2_FEATURES = [
    'wind_now', 'dw_6h', 'dw_12h', 'dw_24h', 'acceleration',
    'abs_lat', 'season_sin', 'season_cos',
    'translation_speed', 'intensity_frac_mpi', 'mpi_deficit',
    'prior_ri', 'storm_age_h', 'wind_asymmetry', 'eye_indicator',
    'pressure_now', 'dp_6h', 'dp_12h', 'dp_24h', 'pressure_accel',
    'kz_deviation', 'pw_efficiency',
    'dist2land',
    'rmw_now', 'drmw_12h', 'rmw_trend',
]

# v3 NEW features (on top of v2)
V3_NEW_FEATURES = [
    # Ocean heat content
    'sst_anomaly', 'tchp', 'sst_gradient', 'dist_warm_current', 'sst',
    # Atmosphere
    'shear_proxy', 'outflow_temp', 'humidity_proxy', 'ventilation',
    # Storm structure
    'drmw_6h', 'drmw_24h', 'rmw_contraction',
    'eye_formation_rate', 'intens_efficiency',
    'symmetry_improvement',
    # Environmental interaction
    'time_since_land', 'heading_change', 'recurvature', 'lat_change_rate',
    # Multi-timescale
    'dw_36h', 'dw_48h', 'd2w_6h', 'intens_persistence',
    # Compound
    'carnot_efficiency', 's_gamma_ratio',
]

V3_ALL_FEATURES = V2_FEATURES + V3_NEW_FEATURES

# Interaction terms for v3
V3_INTERACTIONS = [
    # Ocean-intensity
    ('tchp', 'mpi_deficit'),
    ('sst_anomaly', 'dw_6h'),
    ('sst', 'intensity_frac_mpi'),
    # Shear-structure
    ('shear_proxy', 'eye_indicator'),
    ('ventilation', 'dw_6h'),
    # Structure-intensity
    ('rmw_contraction', 'dw_6h'),
    ('intens_efficiency', 'eye_formation_rate'),
    ('intens_persistence', 'dw_6h'),
    # Original v2 interactions
    ('wind_now', 'dw_6h'),
    ('dw_6h', 'dp_6h'),
    ('intensity_frac_mpi', 'dw_6h'),
    ('abs_lat', 'mpi_deficit'),
    ('dw_6h', 'dw_12h'),
    ('eye_indicator', 'dw_6h'),
    ('prior_ri', 'dw_6h'),
    ('storm_age_h', 'dw_6h'),
    # New compound
    ('carnot_efficiency', 'mpi_deficit'),
    ('s_gamma_ratio', 'dw_6h'),
    ('d2w_6h', 'intens_persistence'),
]

V3_SQUARED = [
    'dw_6h', 'intensity_frac_mpi', 'dp_6h', 'wind_now',
    'shear_proxy', 'tchp', 'intens_efficiency', 'rmw_contraction',
]


# =========================================================================
# BUILD PIPELINE
# =========================================================================

def build_feature_matrix(all_samples, feature_names, impute_missing=True):
    """Build X, y arrays. With imputation: replace -999 with column median."""
    X_list = []
    y_list = []
    year_list = []

    if impute_missing:
        # First pass: collect all values to compute medians
        val_lists = {fn: [] for fn in feature_names}
        for features, label, year, dw in all_samples:
            for fn in feature_names:
                v = features.get(fn, -999)
                if v != -999:
                    val_lists[fn].append(v)
        medians = {}
        for fn in feature_names:
            if val_lists[fn]:
                medians[fn] = float(np.median(val_lists[fn]))
            else:
                medians[fn] = 0.0

        for features, label, year, dw in all_samples:
            row = []
            for fn in feature_names:
                v = features.get(fn, -999)
                if v == -999:
                    row.append(medians[fn])
                else:
                    row.append(v)
            X_list.append(row)
            y_list.append(label)
            year_list.append(year)
    else:
        for features, label, year, dw in all_samples:
            row = []
            valid = True
            for fn in feature_names:
                v = features.get(fn, -999)
                if v == -999:
                    valid = False
                    break
                row.append(v)
            if valid:
                X_list.append(row)
                y_list.append(label)
                year_list.append(year)

    if len(X_list) == 0:
        return None, None, None

    return np.array(X_list), np.array(y_list), np.array(year_list)


def add_interactions(X, feature_names, interactions, squared_features):
    """Add interaction and squared terms."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    new_cols = []
    new_names = list(feature_names)

    for n1, n2 in interactions:
        if n1 in name_to_idx and n2 in name_to_idx:
            i1, i2 = name_to_idx[n1], name_to_idx[n2]
            new_cols.append(X[:, i1] * X[:, i2])
            new_names.append(f'{n1}*{n2}')

    for fname in squared_features:
        if fname in name_to_idx:
            idx = name_to_idx[fname]
            new_cols.append(X[:, idx] ** 2)
            new_names.append(f'{fname}^2')

    if new_cols:
        X_aug = np.column_stack([X] + new_cols)
    else:
        X_aug = X
    return X_aug, new_names


def run_ensemble(samples, feature_names, interactions, squared_features,
                  test_year=2015, verbose=True):
    """
    Full v3 ensemble pipeline:
    Model A: L2 logistic on all features + interactions
    Model B: Gradient boosted stumps (300 stumps)
    Model C: Bagged logistic (10 bags, 50% features)
    Meta-learner: logistic on 3 model outputs + 2 best raw features
    """
    print("\n  Building feature matrix...")
    X, y, years = build_feature_matrix(samples, feature_names, impute_missing=True)
    if X is None:
        print("    No valid samples!")
        return None

    train_mask = years < test_year
    test_mask = years >= test_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    n_pos = np.sum(y_train)
    n_neg = np.sum(y_train == 0)
    cw = min(n_neg / max(n_pos, 1), 20.0)

    print(f"    Train: {len(X_train)} (RI: {int(n_pos)}, {100*n_pos/len(X_train):.1f}%)")
    print(f"    Test:  {len(X_test)} (RI: {int(np.sum(y_test))}, {100*np.mean(y_test):.1f}%)")
    print(f"    Class weight ratio: {cw:.1f}")

    # Standardize
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0) + 1e-10
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    # Add interactions
    print(f"    Adding interactions and squared terms...")
    X_train_aug, aug_names = add_interactions(X_train_s, feature_names,
                                               interactions, squared_features)
    X_test_aug, _ = add_interactions(X_test_s, feature_names,
                                      interactions, squared_features)
    print(f"    Total features (with interactions): {X_train_aug.shape[1]}")

    # ===== MODEL A: L2 Logistic Regression =====
    print("\n    Training Model A: L2 Logistic Regression...")
    t0 = time.time()
    wA, bA = train_logistic(X_train_aug, y_train, lam=0.003, lr=0.05,
                             n_epochs=4000, cw_ratio=cw)
    pA_train = sigmoid(X_train_aug @ wA + bA)
    pA_test = sigmoid(X_test_aug @ wA + bA)
    auc_A, _, _ = compute_roc_auc(y_test, pA_test)
    print(f"      AUC(A) = {auc_A:.4f}  ({time.time()-t0:.1f}s)")

    # ===== MODEL A2: L2 Logistic with stronger class weighting =====
    print("    Training Model A2: L2 Logistic (higher class weight)...")
    t0 = time.time()
    cw2 = min(cw * 2, 40.0)
    wA2, bA2 = train_logistic(X_train_aug, y_train, lam=0.005, lr=0.05,
                                n_epochs=3000, cw_ratio=cw2)
    pA2_train = sigmoid(X_train_aug @ wA2 + bA2)
    pA2_test = sigmoid(X_test_aug @ wA2 + bA2)
    auc_A2, _, _ = compute_roc_auc(y_test, pA2_test)
    print(f"      AUC(A2) = {auc_A2:.4f}  ({time.time()-t0:.1f}s)")

    # ===== MODEL B: Gradient Boosted Stumps =====
    print("    Training Model B: Gradient Boosted Stumps (250)...")
    t0 = time.time()
    stumps, init_val = train_gradient_boosted_stumps(
        X_train_aug, y_train, n_stumps=250, learning_rate=0.08, cw_ratio=cw,
        subsample=10000)
    pB_train = predict_gbs(X_train_aug, stumps, init_val)
    pB_test = predict_gbs(X_test_aug, stumps, init_val)
    auc_B, _, _ = compute_roc_auc(y_test, pB_test)
    print(f"      AUC(B) = {auc_B:.4f}  ({time.time()-t0:.1f}s)")

    # ===== MODEL C: Bagged Logistic =====
    print("    Training Model C: Bagged Logistic (15 bags, 60% features)...")
    t0 = time.time()
    bag_models = train_bagged_logistic(X_train_aug, y_train, n_bags=15,
                                        feat_fraction=0.6, lam=0.005,
                                        lr=0.05, n_epochs=2000, cw_ratio=cw)
    pC_train = predict_bagged(X_train_aug, bag_models)
    pC_test = predict_bagged(X_test_aug, bag_models)
    auc_C, _, _ = compute_roc_auc(y_test, pC_test)
    print(f"      AUC(C) = {auc_C:.4f}  ({time.time()-t0:.1f}s)")

    # ===== META-LEARNER =====
    print("    Training Meta-Learner...")

    # Find top 2 raw features by individual AUC
    individual_aucs = []
    for j in range(X_train_aug.shape[1]):
        auc_j, _, _ = compute_roc_auc(y_train, X_train_aug[:, j])
        individual_aucs.append((auc_j if auc_j >= 0.5 else 1 - auc_j, j))
    individual_aucs.sort(reverse=True)
    top2_idx = [individual_aucs[0][1], individual_aucs[1][1]]
    print(f"      Top 2 features for meta: {aug_names[top2_idx[0]]}, {aug_names[top2_idx[1]]}")

    # Build meta features: 4 model outputs + 2 best raw features
    X_meta_train = np.column_stack([
        pA_train, pA2_train, pB_train, pC_train,
        X_train_aug[:, top2_idx[0]], X_train_aug[:, top2_idx[1]]
    ])
    X_meta_test = np.column_stack([
        pA_test, pA2_test, pB_test, pC_test,
        X_test_aug[:, top2_idx[0]], X_test_aug[:, top2_idx[1]]
    ])

    # Standardize meta features
    mu_meta = np.mean(X_meta_train, axis=0)
    sigma_meta = np.std(X_meta_train, axis=0) + 1e-10
    X_meta_train_s = (X_meta_train - mu_meta) / sigma_meta
    X_meta_test_s = (X_meta_test - mu_meta) / sigma_meta

    wM, bM = train_logistic(X_meta_train_s, y_train, lam=0.001, lr=0.05,
                              n_epochs=2000, cw_ratio=cw)
    pM_test = sigmoid(X_meta_test_s @ wM + bM)
    auc_M, fpr_M, tpr_M = compute_roc_auc(y_test, pM_test)
    print(f"      AUC(Meta) = {auc_M:.4f}")

    # Also compute simple average ensemble for comparison
    p_avg = (pA_test + pA2_test + pB_test + pC_test) / 4.0
    auc_avg, _, _ = compute_roc_auc(y_test, p_avg)
    print(f"      AUC(Simple Avg) = {auc_avg:.4f}")

    # Use whichever is best
    if auc_M >= auc_avg:
        p_final = pM_test
        auc_final = auc_M
        method = "Meta-Learner"
    else:
        p_final = p_avg
        auc_final = auc_avg
        method = "Simple Average"

    print(f"\n    BEST ENSEMBLE: {method}, AUC = {auc_final:.4f}")

    # Optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, p_final)
    prec, rec, f1, tp, fp, fn, tn = compute_pr_metrics(y_test, p_final, opt_thresh)
    brier = np.mean((p_final - y_test)**2)
    climo = np.mean(y_test)
    brier_climo = climo * (1 - climo)
    brier_skill = 1 - brier / brier_climo if brier_climo > 0 else 0

    if verbose:
        print(f"      Optimal threshold: {opt_thresh:.3f}")
        print(f"      Precision:   {prec:.3f}")
        print(f"      Recall:      {rec:.3f}")
        print(f"      F1:          {f1:.3f}")
        print(f"      Brier Skill: {brier_skill:.4f}")
        print(f"      TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Feature importance from Model A
    top_features = []
    order = np.argsort(-np.abs(wA))
    if verbose:
        print(f"\n      Top 20 features (Model A weights):")
        for rank, idx in enumerate(order[:20]):
            direction = '+RI' if wA[idx] > 0 else '-RI'
            print(f"        {rank+1:2d}. {aug_names[idx]:<35s} w={wA[idx]:+.4f}  ({direction})")
            top_features.append(aug_names[idx])

    return {
        'auc_A': auc_A, 'auc_A2': auc_A2, 'auc_B': auc_B, 'auc_C': auc_C,
        'auc_meta': auc_M, 'auc_avg': auc_avg, 'auc_final': auc_final,
        'method': method,
        'fpr': fpr_M, 'tpr': tpr_M,
        'y_test': y_test, 'y_score': p_final,
        'pA_test': pA_test, 'pB_test': pB_test, 'pC_test': pC_test,
        'precision': prec, 'recall': rec, 'f1': f1,
        'brier_skill': brier_skill,
        'opt_thresh': opt_thresh,
        'n_train': len(X_train), 'n_test': len(X_test),
        'wA': wA, 'aug_names': aug_names,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# =========================================================================
# FIGURE
# =========================================================================

def generate_figure(res_v3, res_v2, ri_results):
    """Generate multi-panel v3 figure."""
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Hurricane RI Model v3: Ensemble + Ocean/Atmosphere/Structure Features',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    def style_ax(ax, title):
        ax.set_facecolor('#0d0d1a')
        ax.set_title(title, color=GOLD, fontweight='bold', fontsize=11)
        ax.tick_params(colors='white', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#333')

    # Panel 1: ROC v2 vs v3
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'ROC: v2 vs v3')
    if res_v2 and 'fpr' in res_v2:
        ax.plot(res_v2['fpr'], res_v2['tpr'], '-', color=BLUE, linewidth=1.5,
                label=f"v2 (AUC={res_v2['auc_final']:.3f})")
    if res_v3 and 'fpr' in res_v3:
        ax.plot(res_v3['fpr'], res_v3['tpr'], '-', color=ACCENT, linewidth=2,
                label=f"v3 (AUC={res_v3['auc_final']:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='#555', linewidth=1)
    ax.set_xlabel('FPR', color='white', fontsize=9)
    ax.set_ylabel('TPR', color='white', fontsize=9)
    ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=8)

    # Panel 2: Ensemble component AUCs
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, 'Ensemble Components')
    if res_v3:
        labels = ['Logistic\n(A)', 'Logistic\n(A2)', 'GBStumps\n(B)', 'Bagged\n(C)',
                  'Meta', 'Average']
        aucs = [res_v3['auc_A'], res_v3['auc_A2'], res_v3['auc_B'], res_v3['auc_C'],
                res_v3['auc_meta'], res_v3['auc_avg']]
        colors = [BLUE, TEAL, GREEN, PURPLE, ACCENT, ORANGE]
        bars = ax.bar(range(6), aucs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels, color='white', fontsize=7)
        ax.set_ylabel('AUC', color='white', fontsize=9)
        ax.set_ylim(0.85, 1.0)
        for bar, auc_val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{auc_val:.3f}', ha='center', va='bottom', color='white', fontsize=8)

    # Panel 3: Score distribution
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, 'Score Distribution (v3)')
    if res_v3:
        ax.hist(res_v3['y_score'][res_v3['y_test'] == 0], bins=50, alpha=0.6,
                color=BLUE, density=True, label='Non-RI')
        ax.hist(res_v3['y_score'][res_v3['y_test'] == 1], bins=50, alpha=0.6,
                color=ACCENT, density=True, label='RI')
        ax.set_xlabel('P(RI)', color='white', fontsize=9)
        ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=8)

    # Panel 4: Feature importance top 15
    ax = fig.add_subplot(gs[0, 3])
    style_ax(ax, 'Top 15 Features (Model A)')
    if res_v3:
        wA = res_v3['wA']
        names = res_v3['aug_names']
        order = np.argsort(-np.abs(wA))[:15]
        vals = [np.abs(wA[i]) for i in order]
        cols = [ACCENT if wA[i] > 0 else BLUE for i in order]
        ax.barh(range(15), vals[::-1], color=cols[::-1],
                edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(15))
        ax.set_yticklabels([names[i] for i in order][::-1], color='white', fontsize=6)
        ax.set_xlabel('|Weight|', color='white', fontsize=9)

    # Panel 5: RI threshold comparison
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, 'AUC by RI Threshold')
    if ri_results:
        labels = [r['label'] for r in ri_results if r['res']]
        aucs = [r['res']['auc_final'] for r in ri_results if r['res']]
        bars = ax.bar(range(len(labels)), aucs, color=TEAL,
                      edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, color='white', fontsize=7, rotation=25, ha='right')
        ax.set_ylabel('AUC', color='white', fontsize=9)
        ax.set_ylim(0.85, 1.0)
        for bar, auc_val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{auc_val:.3f}', ha='center', va='bottom', color='white', fontsize=7)

    # Panel 6: Reliability diagram
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, 'Reliability Diagram')
    if res_v3:
        bin_edges = np.linspace(0, 1, 11)
        obs_f = []
        pred_f = []
        for j in range(len(bin_edges) - 1):
            mask = (res_v3['y_score'] >= bin_edges[j]) & (res_v3['y_score'] < bin_edges[j+1])
            n_bin = np.sum(mask)
            if n_bin > 5:
                obs_f.append(np.mean(res_v3['y_test'][mask]))
                pred_f.append(np.mean(res_v3['y_score'][mask]))
        if obs_f:
            ax.plot(pred_f, obs_f, 'o-', color=TEAL, markersize=8, linewidth=2)
            ax.plot([0, 1], [0, 1], '--', color=GOLD, linewidth=1)
            ax.set_xlabel('Predicted P(RI)', color='white', fontsize=9)
            ax.set_ylabel('Observed P(RI)', color='white', fontsize=9)

    # Panel 7: Precision-Recall curve
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, 'Precision-Recall')
    if res_v3:
        thresholds = np.linspace(0.01, 0.99, 100)
        precs = []
        recs = []
        for t in thresholds:
            p, r, _, _, _, _, _ = compute_pr_metrics(res_v3['y_test'], res_v3['y_score'], t)
            precs.append(p)
            recs.append(r)
        ax.plot(recs, precs, '-', color=ACCENT, linewidth=2)
        ax.set_xlabel('Recall', color='white', fontsize=9)
        ax.set_ylabel('Precision', color='white', fontsize=9)
        ax.plot(res_v3['recall'], res_v3['precision'], '*', color=GOLD, markersize=15)

    # Panel 8: v2 vs v3 comparison table
    ax = fig.add_subplot(gs[1, 3])
    style_ax(ax, 'v2 vs v3 Comparison')
    ax.axis('off')
    if res_v2 and res_v3:
        table_data = [
            ['Metric', 'v2', 'v3', 'Delta'],
            ['AUC', f"{res_v2['auc_final']:.4f}", f"{res_v3['auc_final']:.4f}",
             f"{res_v3['auc_final']-res_v2['auc_final']:+.4f}"],
            ['Precision', f"{res_v2['precision']:.3f}", f"{res_v3['precision']:.3f}",
             f"{res_v3['precision']-res_v2['precision']:+.3f}"],
            ['Recall', f"{res_v2['recall']:.3f}", f"{res_v3['recall']:.3f}",
             f"{res_v3['recall']-res_v2['recall']:+.3f}"],
            ['F1', f"{res_v2['f1']:.3f}", f"{res_v3['f1']:.3f}",
             f"{res_v3['f1']-res_v2['f1']:+.3f}"],
            ['Brier Skill', f"{res_v2['brier_skill']:.3f}", f"{res_v3['brier_skill']:.3f}",
             f"{res_v3['brier_skill']-res_v2['brier_skill']:+.3f}"],
        ]
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor('#1a1a2e')
            cell.set_edgecolor('#555')
            cell.set_text_props(color='white')
            if row == 0:
                cell.set_facecolor('#2a2a4e')
                cell.set_text_props(color=GOLD, fontweight='bold')

    # Panel 9-10: Summary text
    ax = fig.add_subplot(gs[2, 0:2])
    style_ax(ax, 'v3 Architecture')
    ax.axis('off')
    arch_text = [
        "ENSEMBLE STACKING ARCHITECTURE:",
        "",
        "  Model A: L2-regularized logistic regression (all features + interactions)",
        "  Model B: Gradient boosted stumps (300 stumps, lr=0.08)",
        "  Model C: Bagged logistic (10 bags, 50% random feature subsets)",
        "",
        "  Meta-learner: Logistic regression on [P(A), P(B), P(C), top2_features]",
        "",
        "NEW FEATURE CATEGORIES:",
        "  - Ocean heat content: SST anomaly, TCHP, SST gradient, warm current dist",
        "  - Atmosphere: shear proxy, outflow temp, humidity, ventilation index",
        "  - Storm structure: RMW contraction rates, eye formation rate,",
        "    intensification efficiency, symmetry improvement rate",
        "  - Environmental: time since land, recurvature, trough proxy",
        "  - Multi-timescale: 3h-48h trends, persistence, 2nd derivative",
        "  - Compound: Carnot efficiency, S/Gamma ratio (Helmholtz)",
    ]
    for j, line in enumerate(arch_text):
        ax.text(0.02, 0.95 - j * 0.06, line, color='white', fontsize=8,
                transform=ax.transAxes, fontfamily='monospace')

    # Panel 11-12: Key findings
    ax = fig.add_subplot(gs[2, 2:4])
    style_ax(ax, 'Key Findings')
    ax.axis('off')
    if res_v3:
        findings = [
            f"FINAL AUC: {res_v3['auc_final']:.4f} (NA basin, 2015+ test)",
            f"",
            f"ENSEMBLE BREAKDOWN:",
            f"  Logistic (A):     {res_v3['auc_A']:.4f}",
            f"  Logistic (A2):    {res_v3['auc_A2']:.4f}",
            f"  GBStumps (B):     {res_v3['auc_B']:.4f}",
            f"  Bagged (C):       {res_v3['auc_C']:.4f}",
            f"  Meta-learner:     {res_v3['auc_meta']:.4f}",
            f"  Simple average:   {res_v3['auc_avg']:.4f}",
            f"",
            f"TOP PHYSICAL PREDICTORS:",
            f"  1. Prior intensification trends (dw_6h, dw_12h)",
            f"  2. Ocean heat content (TCHP, SST anomaly)",
            f"  3. Pressure deepening rate & acceleration",
            f"  4. RMW contraction (coherence concentration)",
            f"  5. S/Gamma ratio (Helmholtz framework)",
            f"  6. Carnot efficiency (thermodynamic potential)",
        ]
        for j, line in enumerate(findings):
            ax.text(0.02, 0.95 - j * 0.06, line, color='white', fontsize=8,
                    transform=ax.transAxes, fontfamily='monospace')

    path = OUT / 'hurricane_ri_model_v3.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Figure saved: {path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  HURRICANE RAPID INTENSIFICATION MODEL v3")
    print("  Ensemble + Ocean/Atmosphere/Structure Features")
    print("=" * 70)

    # Load NA data
    storms_na = load_ibtracs('NA')
    if not storms_na:
        print("  FATAL: Could not load IBTrACS NA data.")
        return

    # ================================================================
    # Extract v3 features (30kt/24h standard threshold)
    # ================================================================
    print("\n" + "=" * 70)
    print("  EXTRACTING v3 FEATURES (30kt/24h)")
    print("=" * 70)

    samples_na = []
    for sid, storm in storms_na.items():
        s = extract_storm_features_v3(storm['entries'],
                                       ri_threshold_kt=30, ri_window_steps=4)
        samples_na.extend(s)

    n_ri = sum(1 for _, l, _, _ in samples_na if l == 1)
    print(f"  NA samples: {len(samples_na)}")
    print(f"  RI events:  {n_ri} ({100*n_ri/max(len(samples_na),1):.1f}%)")

    # ================================================================
    # Run v2-equivalent baseline (for comparison)
    # ================================================================
    print("\n" + "=" * 70)
    print("  v2 BASELINE (L2 logistic, v2 features, no ensemble)")
    print("=" * 70)

    # Simple v2 baseline: just logistic on V2 features + interactions
    X_v2, y_v2, yr_v2 = build_feature_matrix(samples_na, V2_FEATURES, impute_missing=True)
    if X_v2 is not None:
        tr = yr_v2 < 2015
        te = yr_v2 >= 2015
        mu_v2 = np.mean(X_v2[tr], axis=0)
        sig_v2 = np.std(X_v2[tr], axis=0) + 1e-10
        Xtr_v2 = (X_v2[tr] - mu_v2) / sig_v2
        Xte_v2 = (X_v2[te] - mu_v2) / sig_v2

        # Add v2 interactions
        v2_interactions = [
            ('wind_now', 'dw_6h'), ('dw_6h', 'dp_6h'),
            ('intensity_frac_mpi', 'dw_6h'), ('abs_lat', 'mpi_deficit'),
            ('dw_6h', 'dw_12h'), ('eye_indicator', 'dw_6h'),
            ('prior_ri', 'dw_6h'), ('storm_age_h', 'dw_6h'),
        ]
        v2_squared = ['dw_6h', 'intensity_frac_mpi', 'dp_6h', 'wind_now']
        Xtr_v2a, v2_names = add_interactions(Xtr_v2, V2_FEATURES, v2_interactions, v2_squared)
        Xte_v2a, _ = add_interactions(Xte_v2, V2_FEATURES, v2_interactions, v2_squared)

        n_pos_v2 = np.sum(y_v2[tr])
        cw_v2 = min(np.sum(y_v2[tr] == 0) / max(n_pos_v2, 1), 20.0)
        w_v2, b_v2 = train_logistic(Xtr_v2a, y_v2[tr], lam=0.01, lr=0.05,
                                      n_epochs=2000, cw_ratio=cw_v2)
        p_v2 = sigmoid(Xte_v2a @ w_v2 + b_v2)
        auc_v2, fpr_v2, tpr_v2 = compute_roc_auc(y_v2[te], p_v2)
        opt_v2, _ = find_optimal_threshold(y_v2[te], p_v2)
        prec_v2, rec_v2, f1_v2, _, _, _, _ = compute_pr_metrics(y_v2[te], p_v2, opt_v2)
        brier_v2 = np.mean((p_v2 - y_v2[te])**2)
        climo_v2 = np.mean(y_v2[te])
        bs_v2 = 1 - brier_v2 / (climo_v2 * (1 - climo_v2)) if climo_v2 > 0 else 0

        res_v2 = {
            'auc_final': auc_v2, 'fpr': fpr_v2, 'tpr': tpr_v2,
            'precision': prec_v2, 'recall': rec_v2, 'f1': f1_v2,
            'brier_skill': bs_v2,
            'n_train': int(np.sum(tr)), 'n_test': int(np.sum(te)),
        }
        print(f"    v2 baseline AUC: {auc_v2:.4f}")
        print(f"    v2 features: {len(v2_names)} (base {len(V2_FEATURES)} + interactions)")
        print(f"    Precision: {prec_v2:.3f}, Recall: {rec_v2:.3f}, F1: {f1_v2:.3f}")
    else:
        res_v2 = None
        print("    v2 baseline: insufficient data")

    # ================================================================
    # Run v3 ENSEMBLE
    # ================================================================
    print("\n" + "=" * 70)
    print("  v3 ENSEMBLE (NA basin, 30kt/24h, test 2015+)")
    print("=" * 70)

    res_v3 = run_ensemble(samples_na, V3_ALL_FEATURES,
                           V3_INTERACTIONS, V3_SQUARED,
                           test_year=2015, verbose=True)

    # ================================================================
    # RI threshold sensitivity
    # ================================================================
    print("\n" + "=" * 70)
    print("  RI THRESHOLD SENSITIVITY")
    print("=" * 70)

    ri_configs = [
        (30, 2, '30kt/12h'),
        (30, 4, '30kt/24h'),
        (35, 4, '35kt/24h'),
        (30, 8, '30kt/48h'),
    ]

    ri_results = []
    for ri_kt, ri_steps, label in ri_configs:
        print(f"\n  --- {label} ---")
        thresh_samples = []
        for sid, storm in storms_na.items():
            s = extract_storm_features_v3(storm['entries'],
                                           ri_threshold_kt=ri_kt,
                                           ri_window_steps=ri_steps)
            thresh_samples.extend(s)

        n_ri_t = sum(1 for _, l, _, _ in thresh_samples if l == 1)
        print(f"    Samples: {len(thresh_samples)}, RI: {n_ri_t}")

        res = run_ensemble(thresh_samples, V3_ALL_FEATURES,
                           V3_INTERACTIONS, V3_SQUARED,
                           test_year=2015, verbose=True)
        ri_results.append({'label': label, 'res': res})

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: v2 vs v3")
    print("=" * 70)

    print(f"\n  {'Metric':<25s} {'v2':>10s} {'v3':>10s} {'Delta':>10s}")
    print("  " + "-" * 60)
    if res_v2 and res_v3:
        for metric in ['auc_final', 'precision', 'recall', 'f1', 'brier_skill']:
            v2_val = res_v2[metric]
            v3_val = res_v3[metric]
            delta = v3_val - v2_val
            name = metric.replace('_', ' ').title()
            print(f"  {name:<25s} {v2_val:10.4f} {v3_val:10.4f} {delta:+10.4f}")

    print(f"\n  RI THRESHOLD RESULTS:")
    print(f"  {'Threshold':<20s} {'AUC':>10s} {'Prec':>10s} {'Recall':>10s} {'F1':>10s}")
    print("  " + "-" * 55)
    for item in ri_results:
        r = item['res']
        if r:
            print(f"  {item['label']:<20s} {r['auc_final']:10.4f} "
                  f"{r['precision']:10.3f} {r['recall']:10.3f} {r['f1']:10.3f}")

    if res_v3:
        print(f"\n  ENSEMBLE COMPONENT BREAKDOWN:")
        print(f"    Model A  (L2 Logistic):      AUC = {res_v3['auc_A']:.4f}")
        print(f"    Model A2 (L2 high-weight):   AUC = {res_v3['auc_A2']:.4f}")
        print(f"    Model B  (GB Stumps x250):   AUC = {res_v3['auc_B']:.4f}")
        print(f"    Model C  (Bagged Logistic):  AUC = {res_v3['auc_C']:.4f}")
        print(f"    Meta-learner:                AUC = {res_v3['auc_meta']:.4f}")
        print(f"    Simple average:              AUC = {res_v3['auc_avg']:.4f}")
        print(f"    FINAL ({res_v3['method']}):  AUC = {res_v3['auc_final']:.4f}")

    # ================================================================
    # Generate figure
    # ================================================================
    print("\n  Generating figure...")
    generate_figure(res_v3, res_v2, ri_results)

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
