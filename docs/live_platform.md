# Live Natural Hazard Prediction Platform - Architecture

## Real-Time Data Feeds

### Earthquakes (USGS - no API key needed)
- `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_hour.geojson` - M4.5+ last hour, updated every minute
- `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson` - M2.5+ last day, every 15 min
- `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson` - M4.5+ last 7 days
- FDSN historical: `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=...&minmagnitude=4.0`

### Hurricanes (NHC/NOAA)
- NHC GIS data: `https://www.nhc.noaa.gov/gis/`
- ATCF best track: `https://ftp.nhc.noaa.gov/atcf/btk/`
- ATCF model forecasts: `https://ftp.nhc.noaa.gov/atcf/aid_public/`
- IBTrACS: `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv`
- NHC advisories every 6h (00, 06, 12, 18 UTC); every 3h near landfall

### Tornadoes (SPC/NWS)
- Today's reports: `https://www.spc.noaa.gov/climo/reports/today_torn.csv`
- NWS Alerts API: `https://api.weather.gov/alerts/active?event=Tornado%20Warning`
- SPC outlooks: `https://www.spc.noaa.gov/products/outlook/archive/YYYY/day1otlk_YYYYMMDD_1200.lyr.geojson`
- Historical: `https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Prediction backend | Python 3.11+ / FastAPI on VPS ($20-40/mo) |
| Database | PostgreSQL 16 + TimescaleDB |
| Task scheduler | APScheduler (periodic data ingestion) |
| Maps | MapLibre GL JS (open-source, free, WebGL) |
| Charts | D3.js + Observable Plot |
| Frontend hosting | Cloudflare Workers (existing setup) |
| API proxy | Cloudflare Worker → FastAPI, or KV edge cache |
| Prediction integrity | SHA-256 hash chain + GitHub commit anchoring |

---

## Prediction Schedule

| Hazard | Frequency | Trigger | Output |
|---|---|---|---|
| Earthquake | Every 6 hours | + immediate on M5+ | P(M6+ in 30/90/365 days) per 3° cell |
| Hurricane RI | Each advisory (6h) | Each active TC | P(RI in 12h/24h/48h) per storm |
| Tornado formation | Daily (in season) | + on severe weather | P(tornado today) per 2° CONUS cell |
| Tornado severity | On each report | New tornado report | Predicted EF from width |

---

## Verification System (Critical for Credibility)

### Hash Chain
Every prediction → SHA-256(content + prev_hash) → immutable chain.

### External Anchoring
Hourly: latest hash committed to public GitHub repo (independent timestamp).
Optional: OpenTimestamps (Bitcoin-anchored proof of existence).

### Verification Pipeline
- Earthquakes: every 6h, match predictions to USGS M6+ events in each cell
- Hurricanes: after each advisory, check if RI occurred (Δwind ≥ 30kt/24h)
- Tornadoes: nightly, match cell-day predictions to SPC storm reports

### Live Accuracy Metrics
- AUC (updated hourly)
- Brier Score and Brier Skill Score vs climatology
- Reliability diagram (predicted vs observed frequency)
- Sharpness (distribution of predicted probabilities)
- Log Loss
- Hit/miss log table (every prediction matched to outcome)

---

## Database Schema

```sql
-- Predictions (TimescaleDB hypertable)
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hazard_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_hash TEXT NOT NULL,
    prev_hash TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    grid_cell_id TEXT,
    storm_id TEXT,
    probability DOUBLE PRECISION NOT NULL,
    threshold TEXT NOT NULL,
    window_hours INTEGER NOT NULL,
    confidence_lo DOUBLE PRECISION,
    confidence_hi DOUBLE PRECISION,
    features JSONB,
    anchor_hash TEXT,
    anchor_type TEXT
);

-- Events (what actually happened)
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,
    hazard_type TEXT NOT NULL,
    magnitude DOUBLE PRECISION,
    wind_speed_kt DOUBLE PRECISION,
    ef_rating INTEGER,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    source_id TEXT,
    raw_data JSONB
);

-- Verifications (prediction ↔ outcome)
CREATE TABLE verifications (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(id),
    event_id BIGINT REFERENCES events(id),
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    outcome TEXT NOT NULL,
    brier_contrib DOUBLE PRECISION,
    log_score DOUBLE PRECISION
);
```

---

## Dashboard Pages

### `/predictions/earthquakes/`
- Global MapLibre GL map with 3° cells colored by P(M6+ in 30 days)
- "Highest Risk Cells" ranked list
- Recent M6+ hit/miss indicators
- Running AUC with trend, Brier score, reliability diagram

### `/predictions/hurricanes/`
- Basin map with active storms + RI probability bars
- Storm detail: intensity time series, feature gauges, CEL vs NHC vs SHIPS comparison
- Season summary with running accuracy

### `/predictions/tornadoes/`
- CONUS 2° grid map with P(tornado today)
- Active warnings overlay from NWS API
- SPC outlook overlay for comparison
- Real-time severity prediction on new reports

### `/predictions/accuracy/`
- Three-panel layout (one per hazard)
- Running AUC, Brier Score, BSS, reliability diagrams, ROC curves
- Public prediction ledger with hash verification tool
- Head-to-head comparison charts vs operational models

---

## Legal Requirements

**Primary disclaimer (always visible):**
> "These predictions are experimental research outputs from Coherence Energy Labs. They are NOT official forecasts. For official earthquake information, see USGS. For hurricane forecasts, see NHC. For tornado warnings, see NWS. Always follow official guidance."

**Key rules:**
- Never use "warning," "alert," or "imminent" (reserved for NWS under 18 U.S.C. §1038)
- Frame as "elevated probability" not "prediction of event"
- Always show uncertainty bounds and comparison to base rate
- Earthquake cells ≥ 3° (~300km), tornado cells ≥ 2° (~200km) - no neighborhood-level claims

---

## Implementation Phases

| Phase | Scope | Timeline |
|---|---|---|
| 1 | PostgreSQL + FastAPI + data ingestion + hash chain | Weeks 1-3 |
| 2 | Earthquake dashboard + verification pipeline | Weeks 4-6 |
| 3 | Hurricane dashboard + RI predictions | Weeks 7-9 |
| 4 | Tornado dashboard + formation/severity | Weeks 10-12 |
| 5 | Unified accuracy dashboard + public ledger + launch | Weeks 13-14 |

**Cost:** ~$20-40/month VPS + free Cloudflare Workers tier. <1GB/year prediction storage.
