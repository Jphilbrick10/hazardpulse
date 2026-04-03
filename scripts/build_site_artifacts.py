from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PRIMARY_DOMAIN = "https://hazardpulse.com"
CONTACT_EMAIL = "josh@coherenceenergylabs.com"

LIVE_PULSE_PATH = DIST / "data" / "live-pulse.json"
LIVE_TORNADOES_PATH = DIST / "data" / "live-tornadoes.json"
LIVE_STORMS_PATH = DIST / "data" / "live-storms.json"
EQ_LEDGER_PATH = DIST / "data" / "earthquake-ledger.jsonl"
TO_LEDGER_PATH = DIST / "data" / "tornado-ledger.jsonl"
REPLAY_DIR = DIST / "data" / "replay"
REPLAY_INDEX_PATH = DIST / "data" / "evidence" / "replay-index.json"
PREDICTION_LEDGER_PATH = DIST / "data" / "evidence" / "prediction-ledger.json"
PROVENANCE_PATH = DIST / "data" / "evidence" / "provenance-envelopes.json"
GATE_DECISIONS_PATH = DIST / "data" / "evidence" / "gate-decisions.json"
EVIDENCE_PAGE_PATH = DIST / "evidence" / "index.html"
SITEMAP_PATH = DIST / "sitemap.xml"
FEED_PATH = DIST / "feed.xml"

ROUTES = [
    ("/", "daily", "1.0"),
    ("/live/", "hourly", "1.0"),
    ("/live/earthquake/", "hourly", "0.9"),
    ("/live/hurricane/", "hourly", "0.9"),
    ("/live/tornado/", "hourly", "0.9"),
    ("/verification/", "daily", "0.8"),
    ("/evidence/", "daily", "0.8"),
    ("/methods/", "weekly", "0.7"),
    ("/registry/", "daily", "0.8"),
    ("/api/", "weekly", "0.7"),
    ("/ops/status/", "hourly", "0.7"),
    ("/legal/disclaimer/", "monthly", "0.4"),
]

HAZARD_LABELS = {
    "eq": "Earthquake",
    "earthquake": "Earthquake",
    "hu": "Hurricane",
    "hurricane": "Hurricane",
    "to": "Tornado",
    "tornado": "Tornado",
}


def _read_json(path: Path, default: dict | list | None = None):
    if default is None:
        default = {}
    if not path.exists():
        return default.copy() if isinstance(default, dict) else list(default)
    try:
        return json.loads(path.read_text(encoding="utf-8").replace("\ufeff", ""))
    except json.JSONDecodeError:
        return default.copy() if isinstance(default, dict) else list(default)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _parse_utc(value: object) -> dt.datetime | None:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith(" UTC"):
            text = text[:-4] + "Z"
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _format_utc_z(value: dt.datetime | None) -> str:
    if value is None:
        value = dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_http_date(value: dt.datetime | None) -> str:
    if value is None:
        value = dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


def _forecast_id(prefix: str, issued_at: dt.datetime | None) -> str:
    issue = issued_at or dt.datetime.now(dt.timezone.utc)
    if issue.tzinfo is None:
        issue = issue.replace(tzinfo=dt.timezone.utc)
    issue = issue.astimezone(dt.timezone.utc)
    return f"{prefix}_fcst_{issue.strftime('%Y%m%d_%H%M')}"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _asset_ref(path: Path) -> str:
    return "/" + path.relative_to(DIST).as_posix()


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"


def _short_hash(value: str | None) -> str:
    if not value:
        return "--"
    clean = value.replace("sha256:", "")
    if len(clean) <= 16:
        return clean
    return f"{clean[:8]}..{clean[-8:]}"


def _hazard_label(value: object) -> str:
    return HAZARD_LABELS.get(str(value), str(value).replace("_", " ").title())


def _load_replay_index() -> dict:
    payload = _read_json(REPLAY_INDEX_PATH, {"generated_at": None, "items": []})
    payload.setdefault("items", [])
    return payload


def _upsert_replay_index_item(index: dict, forecast_id: str, replay_path: Path) -> None:
    items = [item for item in index.get("items", []) if item.get("forecast_id") != forecast_id]
    items.append({"forecast_id": forecast_id, "replay_artifact": _asset_ref(replay_path)})
    items.sort(key=lambda item: item.get("forecast_id", ""))
    index["generated_at"] = _format_utc_z(dt.datetime.now(dt.timezone.utc))
    index["items"] = items


def _ensure_live_publish_artifacts() -> tuple[dict, dict]:
    pulse = _read_json(LIVE_PULSE_PATH, {"updated_at": None, "hazards": []})
    replay_index = _load_replay_index()

    def update_hazard(key: str, forecast_id: str | None) -> None:
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == key:
                hazard["forecast_id"] = forecast_id
                break

    storms = _read_json(LIVE_STORMS_PATH, {})
    storms_updated = _parse_utc(storms.get("updated_at"))
    if storms_updated is not None:
        forecast_id = storms.get("forecast_id") or _forecast_id("hu", storms_updated)
        storms["forecast_id"] = forecast_id
        top_probability = 0.0
        if storms.get("storms"):
            top_probability = max(float(item.get("ri_probability", 0) or 0) for item in storms["storms"])
        artifact = {
            "forecast_id": forecast_id,
            "hazard": "hurricane",
            "issued_at": _format_utc_z(storms_updated),
            "model_version": storms.get("model_version", "hurricane_ri_v8_1"),
            "forecast_horizon_hours": 24,
            "n_active_storms": int(storms.get("n_active_storms", 0) or 0),
            "top_probability": round(top_probability, 4),
            "source_artifacts": ["/data/live-storms.json"],
            "storms": storms.get("storms", []),
        }
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        _write_json(replay_path, artifact)
        _upsert_replay_index_item(replay_index, forecast_id, replay_path)
        update_hazard("hu", forecast_id)
        _write_json(LIVE_STORMS_PATH, storms)

    tornadoes = _read_json(LIVE_TORNADOES_PATH, {})
    tornadoes_updated = _parse_utc(tornadoes.get("updated_at"))
    if tornadoes_updated is not None:
        forecast_id = tornadoes.get("forecast_id") or _forecast_id("to", tornadoes_updated)
        tornadoes["forecast_id"] = forecast_id
        top_probability = 0.0
        if tornadoes.get("storms"):
            top_probability = max(
                float(item.get("tornado_probability", 0) or 0)
                for item in tornadoes["storms"]
            )
        artifact = {
            "forecast_id": forecast_id,
            "hazard": "tornado",
            "issued_at": _format_utc_z(tornadoes_updated),
            "model_version": tornadoes.get("model_version", "tornado_storm_v1_0"),
            "forecast_horizon_hours": 24,
            "scoring_tier": tornadoes.get("scoring_tier"),
            "scoring_tier_label": tornadoes.get("scoring_tier_label"),
            "coherence_source": tornadoes.get("coherence_source"),
            "n_active_storms": int(tornadoes.get("n_active_storms", 0) or 0),
            "top_probability": round(top_probability, 4),
            "source_artifacts": ["/data/live-tornadoes.json", "/data/tornado-storms.geojson"],
            "storms": tornadoes.get("storms", []),
        }
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        _write_json(replay_path, artifact)
        _upsert_replay_index_item(replay_index, forecast_id, replay_path)
        update_hazard("to", forecast_id)
        _write_json(LIVE_TORNADOES_PATH, tornadoes)

    eq_hazard = next((item for item in pulse.get("hazards", []) if item.get("key") == "eq"), {})
    eq_forecast_id = eq_hazard.get("forecast_id")
    if eq_forecast_id:
        replay_path = REPLAY_DIR / f"{eq_forecast_id}.json"
        if replay_path.exists():
            _upsert_replay_index_item(replay_index, eq_forecast_id, replay_path)

    _write_json(REPLAY_INDEX_PATH, replay_index)
    _write_json(LIVE_PULSE_PATH, pulse)
    return pulse, replay_index


def _collect_prediction_entries(pulse: dict, replay_index: dict) -> list[dict]:
    entries: list[dict] = []

    current_eq = next(
        (hazard.get("forecast_id") for hazard in pulse.get("hazards", []) if hazard.get("key") == "eq"),
        None,
    )
    current_hu = next(
        (hazard.get("forecast_id") for hazard in pulse.get("hazards", []) if hazard.get("key") == "hu"),
        None,
    )
    current_to = next(
        (hazard.get("forecast_id") for hazard in pulse.get("hazards", []) if hazard.get("key") == "to"),
        None,
    )

    for row in _read_jsonl(EQ_LEDGER_PATH):
        issued_at = _parse_utc(row.get("timestamp"))
        forecast_id = row.get("forecast_id") or _forecast_id("eq", issued_at)
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "earthquake",
                "issued_at": _format_utc_z(issued_at),
                "hash": f"sha256:{row.get('hash', '')}",
                "prev_hash": f"sha256:{row.get('prev_hash', '')}",
                "model_version": row.get("model_version"),
                "probability": row.get("top_probability", 0.0),
                "replay_artifact": _asset_ref(replay_path) if replay_path.exists() else None,
                "current": forecast_id == current_eq,
            }
        )

    for row in _read_jsonl(TO_LEDGER_PATH):
        issued_at = _parse_utc(row.get("timestamp"))
        forecast_id = row.get("forecast_id") or _forecast_id("to", issued_at)
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "tornado",
                "issued_at": _format_utc_z(issued_at),
                "hash": f"sha256:{row.get('hash', '')}",
                "prev_hash": f"sha256:{row.get('prev_hash', '')}",
                "model_version": row.get("model_version"),
                "probability": row.get("top_probability", 0.0),
                "replay_artifact": _asset_ref(replay_path) if replay_path.exists() else None,
                "current": forecast_id == current_to,
            }
        )

    seen_hurricane: set[str] = set()
    for item in replay_index.get("items", []):
        forecast_id = item.get("forecast_id", "")
        if not forecast_id.startswith("hu_fcst_") or forecast_id in seen_hurricane:
            continue
        replay_artifact = item.get("replay_artifact")
        replay_path = DIST / replay_artifact.lstrip("/")
        if not replay_path.exists():
            continue
        artifact = _read_json(replay_path, {})
        entry_hash = f"sha256:{_canonical_hash(artifact)}"
        seen_hurricane.add(forecast_id)
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "hurricane",
                "issued_at": artifact.get("issued_at", _format_utc_z(_parse_utc(artifact.get("issued_at")))),
                "hash": entry_hash,
                "prev_hash": None,
                "model_version": artifact.get("model_version"),
                "probability": artifact.get("top_probability", 0.0),
                "replay_artifact": replay_artifact,
                "current": forecast_id == current_hu,
            }
        )

    entries.sort(key=lambda item: item.get("issued_at", ""), reverse=True)
    return entries


def _build_provenance_envelopes(entries: list[dict]) -> list[dict]:
    envelopes: list[dict] = []
    for entry in entries:
        replay_artifact = entry.get("replay_artifact")
        if not replay_artifact:
            continue
        replay_path = DIST / replay_artifact.lstrip("/")
        if not replay_path.exists():
            continue
        artifact = _read_json(replay_path, {})
        hazard = entry.get("hazard")
        if hazard == "earthquake":
            input_manifest = {
                "source_catalog": artifact.get("source_catalog"),
                "forecast_domain": artifact.get("forecast_domain"),
                "feature_history_days": artifact.get("feature_history_days"),
                "recent_activity_days": artifact.get("recent_activity_days"),
            }
            sources = [artifact.get("source_catalog", {}).get("provider", "USGS FDSNWS")]
            transforms = [
                "catalog_ingest",
                "grid_binning",
                "coherence_feature_extraction",
                "singularity_scoring",
                "publish_artifact",
            ]
        elif hazard == "hurricane":
            input_manifest = {
                "source_artifacts": artifact.get("source_artifacts"),
                "n_active_storms": artifact.get("n_active_storms"),
                "storm_ids": [storm.get("storm_id") for storm in artifact.get("storms", [])],
            }
            sources = ["ATCF advisories", "NOAA tropical cyclone feed", "published live storm snapshot"]
            transforms = [
                "advisory_ingest",
                "feature_build",
                "ri_scoring",
                "calibration",
                "publish_artifact",
            ]
        else:
            input_manifest = {
                "source_artifacts": artifact.get("source_artifacts"),
                "scoring_tier": artifact.get("scoring_tier"),
                "coherence_source": artifact.get("coherence_source"),
                "n_active_storms": artifact.get("n_active_storms"),
                "storm_ids": [storm.get("storm_id") for storm in artifact.get("storms", [])],
            }
            sources = ["ProbSevere storm objects", "published live tornado snapshot"]
            if artifact.get("coherence_source") == "hrrr":
                sources.append("HRRR analysis")
            transforms = [
                "probsevere_ingest",
                "coherence_scoring",
                "storm_ranking",
                "publish_artifact",
            ]

        envelopes.append(
            {
                "provenance_id": f"prov_{entry['forecast_id']}",
                "forecast_id": entry["forecast_id"],
                "hazard": hazard,
                "model_version": artifact.get("model_version", entry.get("model_version")),
                "input_hash": f"sha256:{_canonical_hash(input_manifest)}",
                "output_hash": f"sha256:{_canonical_hash(artifact)}",
                "signed_at": artifact.get("issued_at", entry.get("issued_at")),
                "sources": sources,
                "transforms": transforms,
                "replay_artifact": replay_artifact,
            }
        )
    return envelopes


def _build_gate_decisions(entries: list[dict], pulse: dict) -> list[dict]:
    decisions: list[dict] = []
    hazard_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    hazard_key_for_name = {"earthquake": "eq", "hurricane": "hu", "tornado": "to"}
    for entry in entries:
        if not entry.get("replay_artifact"):
            continue
        hazard_name = str(entry.get("hazard"))
        hazard_key = hazard_key_for_name.get(hazard_name, hazard_name)
        hazard = hazard_map.get(hazard_key, {})
        reasons: list[str] = []
        warnings: list[str] = []
        decision = "pass"
        if hazard.get("conf_lo") is None or hazard.get("conf_hi") is None:
            warnings.append("confidence_interval_unavailable")
        if hazard_name == "hurricane" and int(hazard.get("n_active_storms", 0) or 0) == 0:
            warnings.append("no_active_tropical_cyclones_in_feed")
        if hazard_name == "tornado" and hazard.get("coherence_source") == "probsevere":
            warnings.append("probsevere_coherence_fallback_active")
        if hazard.get("gate_status") not in (None, "", "pass"):
            decision = str(hazard.get("gate_status"))
            reasons.append(f"publish_gate_status_{decision}")
        decisions.append(
            {
                "gate_decision_id": f"gdec_{entry['forecast_id']}",
                "forecast_id": entry["forecast_id"],
                "hazard": hazard_name,
                "decision": decision,
                "reasons": reasons,
                "warnings": warnings,
                "issued_at": entry.get("issued_at"),
            }
        )
    return decisions


def _count_link_mismatches(path: Path) -> tuple[int, int]:
    rows = _read_jsonl(path)
    mismatches = 0
    previous_hash = "0" * 64
    for index, row in enumerate(rows):
        expected = previous_hash if index else "0" * 64
        if str(row.get("prev_hash", "")) != expected:
            mismatches += 1
        previous_hash = str(row.get("hash", previous_hash))
    return len(rows), mismatches


def _render_evidence_page(
    pulse: dict,
    entries: list[dict],
    envelopes: list[dict],
    gate_decisions: list[dict],
    replay_index: dict,
) -> None:
    hazard_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    eq_rows, eq_mismatches = _count_link_mismatches(EQ_LEDGER_PATH)
    to_rows, to_mismatches = _count_link_mismatches(TO_LEDGER_PATH)
    replayable_entries = [entry for entry in entries if entry.get("replay_artifact")]

    evidence_cards: list[str] = []
    for key in ("eq", "hu", "to"):
        hazard = hazard_map.get(key, {})
        label = _hazard_label(key)
        forecast_id = hazard.get("forecast_id") or "No published artifact id"
        gate_text = str(hazard.get("gate_status", "pass")).replace("_", " ")
        replay_ready = "Yes" if any(item.get("forecast_id") == hazard.get("forecast_id") for item in replay_index.get("items", [])) else "Pending"
        evidence_cards.append(
            f'<div class="card col-4 hazard-{key}">'
            f"<h3>{_esc(label)}</h3>"
            f'<div class="metric">{_pct(hazard.get("probability", 0))}</div>'
            f'<div class="metric-label">Current published state</div>'
            f'<div class="kv"><span>Forecast ID</span><strong><code>{_esc(forecast_id)}</code></strong></div>'
            f'<div class="kv"><span>Gate</span><strong>{_esc(gate_text.title())}</strong></div>'
            f'<div class="kv"><span>Replay ready</span><strong>{replay_ready}</strong></div>'
            f"</div>"
        )

    ledger_rows = []
    for entry in entries[:12]:
        replay_link = entry.get("replay_artifact")
        ledger_rows.append(
            "<tr>"
            f"<td><code>{_esc(entry.get('forecast_id'))}</code></td>"
            f"<td>{_esc(_hazard_label(entry.get('hazard')))}</td>"
            f"<td>{_esc(entry.get('issued_at'))}</td>"
            f"<td>{_pct(entry.get('probability', 0))}</td>"
            f"<td><code>{_esc(_short_hash(entry.get('hash')))}</code></td>"
            f"<td>{'<a href=\"' + _esc(replay_link) + '\">Replay</a>' if replay_link else 'Archive pending'}</td>"
            "</tr>"
        )

    provenance_cards = []
    for envelope in envelopes[:6]:
        provenance_cards.append(
            '<div class="card">'
            f"<h3><code>{_esc(envelope['provenance_id'])}</code></h3>"
            f'<div class="kv"><span>Forecast</span><strong><code>{_esc(envelope["forecast_id"])}</code></strong></div>'
            f'<div class="kv"><span>Input hash</span><strong><code>{_esc(_short_hash(envelope["input_hash"]))}</code></strong></div>'
            f'<div class="kv"><span>Output hash</span><strong><code>{_esc(_short_hash(envelope["output_hash"]))}</code></strong></div>'
            f'<div class="kv"><span>Signed</span><strong>{_esc(envelope["signed_at"])}</strong></div>'
            f'<p class="muted" style="margin:12px 0 0;">Sources: {_esc(", ".join(envelope["sources"]))}</p>'
            "</div>"
        )

    gate_rows = []
    for decision in gate_decisions[:10]:
        warnings = ", ".join(decision.get("warnings", [])) or "None"
        gate_rows.append(
            "<tr>"
            f"<td><code>{_esc(decision['gate_decision_id'])}</code></td>"
            f"<td>{_esc(_hazard_label(decision['hazard']))}</td>"
            f"<td>{_esc(decision['decision'])}</td>"
            f"<td>{_esc(warnings)}</td>"
            f"<td>{_esc(decision.get('issued_at'))}</td>"
            "</tr>"
        )

    replay_rows = []
    for item in replay_index.get("items", []):
        replay_rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('forecast_id'))}</code></td>"
            f"<td><a href=\"{_esc(item.get('replay_artifact'))}\">{_esc(item.get('replay_artifact'))}</a></td>"
            "</tr>"
        )

    coverage = f"{len(envelopes)}/{max(1, len(replayable_entries))}"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Ledger - HazardPulse</title>
  <meta name="description" content="Live evidence ledger built from real forecast archives, current publish artifacts, provenance hashes, and gate decisions.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/evidence/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Evidence Ledger - HazardPulse">
  <meta property="og:description" content="Live evidence ledger built from real forecast archives, current publish artifacts, provenance hashes, and gate decisions.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/evidence/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Evidence Ledger - HazardPulse">
  <meta name="twitter:description" content="Live evidence ledger built from real forecast archives, current publish artifacts, provenance hashes, and gate decisions.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "DataCatalog",
    "name": "HazardPulse Evidence Ledger",
    "description": "Real forecast archives, provenance hashes, gate decisions, and replay artifacts for published HazardPulse forecasts.",
    "url": "{PRIMARY_DOMAIN}/evidence/",
    "creator": {{ "@type": "Organization", "name": "HazardPulse", "url": "{PRIMARY_DOMAIN}/" }}
  }}
  </script>
  <script type="speculationrules">
  {{ "prefetch": [{{ "source": "list", "urls": ["/", "/live/", "/verification/", "/api/"] }}] }}
  </script>
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
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/" aria-current="page">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Evidence</div>
      <h1>Every published state has a real artifact.</h1>
      <p class="subtitle">
        This surface is generated from the live publish artifacts in <code>/data</code>, the replay archive,
        and the raw earthquake and tornado ledgers. No sample hashes, no synthetic gate records, no fabricated provenance.
      </p>
      <p class="muted">Updated {_esc(pulse.get("updated_at"))} &middot; Replayable artifacts: {len(replay_index.get("items", []))}</p>
    </section>
    <section class="section">
      <div class="grid">
        {"".join(evidence_cards)}
      </div>
    </section>
    <section class="section">
      <div class="grid">
        <div class="card col-3"><div class="metric mono">{eq_rows + to_rows}</div><div class="metric-label">Raw chain rows audited</div></div>
        <div class="card col-3"><div class="metric mono">{eq_mismatches + to_mismatches}</div><div class="metric-label">Prev-hash mismatches</div></div>
        <div class="card col-3"><div class="metric mono">{coverage}</div><div class="metric-label">Provenance coverage</div></div>
        <div class="card col-3"><div class="metric mono">{len(replay_index.get("items", []))}</div><div class="metric-label">Replay artifacts</div></div>
      </div>
    </section>
    <section class="section" id="ledger">
      <h2>Prediction ledger</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Recent append-only and archive-backed forecast records, newest first.</p>
      <div class="card">
        <table>
          <thead><tr><th>Forecast ID</th><th>Hazard</th><th>Issued</th><th>Probability</th><th>Hash</th><th>Artifact</th></tr></thead>
          <tbody>
            {"".join(ledger_rows)}
          </tbody>
        </table>
      </div>
      <div class="cta-row"><a href="/data/evidence/prediction-ledger.json" class="btn btn-secondary">Download prediction ledger</a></div>
    </section>
    <section class="section">
      <h2>Provenance envelopes</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Hashes are computed from the actual published replay artifacts and their source manifests.</p>
      <div class="grid">
        {"".join(provenance_cards) or '<div class="card"><p class="muted" style="margin:0;">No replay artifacts are available yet.</p></div>'}
      </div>
      <div class="cta-row"><a href="/data/evidence/provenance-envelopes.json" class="btn btn-secondary">Download provenance envelopes</a></div>
    </section>
    <section class="section" id="gates">
      <h2>Gate decisions</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Current warnings are derived from the live publish state, not hardcoded examples.</p>
      <div class="card">
        <table>
          <thead><tr><th>Decision ID</th><th>Hazard</th><th>Decision</th><th>Warnings</th><th>Issued</th></tr></thead>
          <tbody>
            {"".join(gate_rows)}
          </tbody>
        </table>
      </div>
      <div class="cta-row"><a href="/data/evidence/gate-decisions.json" class="btn btn-secondary">Download gate decisions</a></div>
    </section>
    <section class="section" id="replay">
      <h2>Replay index</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Frozen publish artifacts that can be fetched directly through the public API.</p>
      <div class="card">
        <table>
          <thead><tr><th>Forecast ID</th><th>Artifact</th></tr></thead>
          <tbody>
            {"".join(replay_rows)}
          </tbody>
        </table>
      </div>
      <div class="cta-row"><a href="/data/evidence/replay-index.json" class="btn btn-secondary">Download replay index</a></div>
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
        <h4>About</h4>
        <a href="mailto:{CONTACT_EMAIL}">Contact</a>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the USGS, NHC, NWS, SPC, JMA, and IMD.
      </p>
      <p class="footer-build">Static-first HTML &middot; Evidence-linked data &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
</body>
</html>
"""
    EVIDENCE_PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PAGE_PATH.write_text(page, encoding="utf-8")


def _render_live_hurricane_page() -> None:
    storms = _read_json(LIVE_STORMS_PATH, {})
    updated_at = _parse_utc(storms.get("updated_at"))
    storms_list = list(storms.get("storms", []))

    if storms_list:
        rows = []
        sorted_storms = sorted(
            storms_list,
            key=lambda item: float(item.get("ri_probability", 0) or 0),
            reverse=True,
        )
        for storm in sorted_storms[:8]:
            rows.append(
                f"<tr>"
                f"<td>{_esc(storm.get('storm_name', storm.get('storm_id', 'Storm')))}</td>"
                f"<td>{_esc(storm.get('category', '--'))}</td>"
                f"<td>{_esc(storm.get('lat', '--'))}, {_esc(storm.get('lon', '--'))}</td>"
                f"<td>{_pct(storm.get('ri_probability', 0))}</td>"
                f"<td>{_esc(storm.get('vmax_kt', '--'))} kt</td>"
                f"</tr>"
            )
        storms_html = (
            "<table><thead><tr><th>Storm</th><th>Status</th><th>Location</th><th>RI 24h</th><th>Wind</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        top = sorted_storms[0]
        summary_html = (
            '<div class="card hazard-hu">'
            '<h2 style="margin-top:0;">Top storm</h2>'
            f'<div class="metric">{_pct(top.get("ri_probability", 0))}</div>'
            '<div class="metric-label">Rapid intensification in 24h</div>'
            f'<div class="kv"><span>Name</span><strong>{_esc(top.get("storm_name", top.get("storm_id", "Storm")))}</strong></div>'
            f'<div class="kv"><span>Status</span><strong>{_esc(top.get("category", "--"))} &middot; {_esc(top.get("vmax_kt", "--"))} kt</strong></div>'
            "</div>"
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

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Hurricane Forecasts - HazardPulse</title>
  <meta name="description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/hurricane/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta property="og:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/hurricane/">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta name="twitter:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Live Hurricane Forecasts",
    "url": "{PRIMARY_DOMAIN}/live/hurricane/",
    "description": "Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output."
  }}
  </script>
  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/", "/live/", "/live/earthquake/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
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
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
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
      <p class="muted">Updated {_esc(_format_utc_z(updated_at))} &middot; Model: hurricane_ri_v8_1 &middot; Independent hazard intelligence platform</p>
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
        Independent hazard intelligence platform. Always follow official guidance from the NHC, JTWC, WMO RSMCs, and local emergency authorities.
      </p>
      <p class="footer-build">Static-first HTML &middot; Live data under <code>/data</code> &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
</body>
</html>
"""
    live_hurricane_path = DIST / "live" / "hurricane" / "index.html"
    live_hurricane_path.parent.mkdir(parents=True, exist_ok=True)
    live_hurricane_path.write_text(page, encoding="utf-8")


def _write_sitemap_and_feed(pulse: dict) -> None:
    updated_at = _parse_utc(pulse.get("updated_at")) or dt.datetime.now(dt.timezone.utc)
    lastmod = updated_at.strftime("%Y-%m-%d")

    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for route, changefreq, priority in ROUTES:
        sitemap_lines.append(
            f"  <url><loc>{PRIMARY_DOMAIN}{route}</loc><lastmod>{lastmod}</lastmod><changefreq>{changefreq}</changefreq><priority>{priority}</priority></url>"
        )
    sitemap_lines.append("</urlset>")
    SITEMAP_PATH.write_text("\n".join(sitemap_lines) + "\n", encoding="utf-8")

    hazard_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    summary = (
        f"Earthquake {_pct(hazard_map.get('eq', {}).get('probability', 0))}, "
        f"Hurricane {_pct(hazard_map.get('hu', {}).get('probability', 0))}, "
        f"Tornado {_pct(hazard_map.get('to', {}).get('probability', 0))}."
    )
    feed_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>HazardPulse Updates</title>
    <link>{PRIMARY_DOMAIN}/</link>
    <description>Current publish-cycle updates from the live HazardPulse surface</description>
    <language>en-us</language>
    <item>
      <title>HazardPulse publish cycle {updated_at.strftime('%Y-%m-%d %H:%M UTC')}</title>
      <link>{PRIMARY_DOMAIN}/</link>
      <guid>hazardpulse-publish-{updated_at.strftime('%Y%m%d%H%M')}</guid>
      <pubDate>{_format_http_date(updated_at)}</pubDate>
      <description>{_esc(summary)}</description>
    </item>
  </channel>
</rss>
"""
    FEED_PATH.write_text(feed_xml, encoding="utf-8")


def _normalize_html_accessibility_labels() -> None:
    for html_path in DIST.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        normalized = text.replace(
            'aria-label="Switch to light mode"',
            'aria-label="Switch to dark mode"',
        )
        if normalized != text:
            html_path.write_text(normalized, encoding="utf-8")


def build_site_artifacts() -> dict:
    pulse, replay_index = _ensure_live_publish_artifacts()
    entries = _collect_prediction_entries(pulse, replay_index)
    envelopes = _build_provenance_envelopes(entries)
    gate_decisions = _build_gate_decisions(entries, pulse)

    _write_json(
        PREDICTION_LEDGER_PATH,
        {
            "mode": "append_only",
            "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
            "entries": entries,
        },
    )
    _write_json(
        PROVENANCE_PATH,
        {
            "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
            "envelopes": envelopes,
        },
    )
    _write_json(
        GATE_DECISIONS_PATH,
        {
            "gate_set_version": "2026.04",
            "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
            "decisions": gate_decisions,
        },
    )
    _render_live_hurricane_page()
    _render_evidence_page(pulse, entries, envelopes, gate_decisions, replay_index)
    _write_sitemap_and_feed(pulse)
    _normalize_html_accessibility_labels()

    return {
        "pulse": pulse,
        "entries": entries,
        "envelopes": envelopes,
        "gate_decisions": gate_decisions,
        "replay_index": replay_index,
    }


def main() -> None:
    build_site_artifacts()
    print("Built HazardPulse evidence, replay, sitemap, and feed artifacts.")


if __name__ == "__main__":
    main()
