/**
 * HazardPulse Cloudflare Worker
 *
 * - Serves static assets from dist/
 * - Exposes public beta JSON/SSE API routes backed by dist/data/*
 * - Applies edge geolocation personalization to HTML responses
 */

const API_VERSION = "v1";
const PRIMARY_DOMAIN = "https://hazardpulse.com";
const ALERT_THRESHOLDS = { critical: 50, severe: 150, warning: 400, watch: 800 };
const HTML_CACHE_CONTROL = "private, no-cache, no-store, must-revalidate";
const HTML_CONTENT_SECURITY_POLICY =
  "default-src 'self'; " +
  "script-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net; " +
  "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; " +
  "img-src 'self' data: https://tile.openstreetmap.org https://*.tile.openstreetmap.org https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com; " +
  "font-src 'self' data: https://fonts.gstatic.com; " +
  "connect-src 'self' https://tile.openstreetmap.org https://*.tile.openstreetmap.org https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com https://noaa-mrms-pds.s3.amazonaws.com; " +
  "frame-ancestors 'none'; base-uri 'self'; form-action 'self' mailto:; object-src 'none'; upgrade-insecure-requests";

const AGENCY_LINKS = {
  earthquake:
    '<a href="https://earthquake.usgs.gov/" rel="noopener">USGS</a> &middot; ' +
    '<a href="https://www.emsc-csem.org/" rel="noopener">EMSC</a>',
  hurricane:
    '<a href="https://www.nhc.noaa.gov/" rel="noopener">NHC</a> &middot; ' +
    '<a href="https://www.metoc.navy.mil/jtwc/jtwc.html" rel="noopener">JTWC</a>',
  tornado:
    '<a href="https://www.spc.noaa.gov/" rel="noopener">SPC</a> &middot; ' +
    '<a href="https://www.weather.gov/" rel="noopener">NWS</a>',
};

function isFiniteCoordinate(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function hasReliableGeo(geo) {
  if (!geo) return false;
  if (!isFiniteCoordinate(geo.latitude) || !isFiniteCoordinate(geo.longitude)) {
    return false;
  }
  if (geo.latitude < -90 || geo.latitude > 90) return false;
  if (geo.longitude < -180 || geo.longitude > 180) return false;
  if (Math.abs(geo.latitude) < 0.25 && Math.abs(geo.longitude) < 0.25) {
    return false;
  }
  return Boolean(
    geo.city || geo.country || geo.region || geo.timezone || geo.continent
  );
}

function normalizeGeo(cf = {}) {
  const latitude =
    cf.latitude !== undefined && cf.latitude !== null ? Number(cf.latitude) : null;
  const longitude =
    cf.longitude !== undefined && cf.longitude !== null ? Number(cf.longitude) : null;

  const geo = {
    latitude: isFiniteCoordinate(latitude) ? latitude : null,
    longitude: isFiniteCoordinate(longitude) ? longitude : null,
    city: cf.city || null,
    country: cf.country || null,
    region: cf.region || null,
    continent: cf.continent || null,
    timezone: cf.timezone || null,
  };
  geo.isReliable = hasReliableGeo(geo);
  return geo;
}

function withSecurityHeaders(
  response,
  { cacheControl, xRobotsTag, contentSecurityPolicy, vary } = {}
) {
  const secured = new Response(response.body, response);
  secured.headers.set(
    "Strict-Transport-Security",
    "max-age=63072000; includeSubDomains; preload"
  );
  secured.headers.set(
    "Content-Security-Policy",
    contentSecurityPolicy || HTML_CONTENT_SECURITY_POLICY
  );
  secured.headers.set("Cross-Origin-Embedder-Policy", "credentialless");
  secured.headers.set("Cross-Origin-Opener-Policy", "same-origin");
  secured.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), interest-cohort=(), payment=(), usb=(), bluetooth=(), serial=()"
  );
  secured.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  secured.headers.set("X-Content-Type-Options", "nosniff");
  secured.headers.set("X-Frame-Options", "DENY");
  secured.headers.set("X-Build-Mode", "observatory-v7");
  if (cacheControl) secured.headers.set("Cache-Control", cacheControl);
  if (cacheControl && cacheControl.includes("no-store")) {
    secured.headers.set("CDN-Cache-Control", "no-store");
    secured.headers.set("Cloudflare-CDN-Cache-Control", "no-store");
    secured.headers.set("Surrogate-Control", "no-store");
  }
  if (xRobotsTag) secured.headers.set("X-Robots-Tag", xRobotsTag);
  if (vary) secured.headers.set("Vary", vary);
  return secured;
}

function isLikelyDocumentRequest(request, pathname) {
  if (pathname === "/") return true;
  if (pathname.endsWith("/")) return true;
  const lastSegment = pathname.split("/").pop() || "";
  if (!lastSegment.includes(".")) return true;
  const accept = request.headers.get("accept") || "";
  return accept.includes("text/html");
}

async function notFoundResponse(env, request) {
  if (isLikelyDocumentRequest(request, new URL(request.url).pathname)) {
    const notFoundAsset = await fetchAsset(env, "/404.html");
    if (notFoundAsset.ok) {
      return withSecurityHeaders(
        new Response(notFoundAsset.body, {
          status: 404,
          headers: notFoundAsset.headers,
        }),
        {
          cacheControl: "no-store",
          xRobotsTag: "noindex, nofollow",
          contentSecurityPolicy: HTML_CONTENT_SECURITY_POLICY,
        }
      );
    }
  }

  return withSecurityHeaders(
    new Response("Not found", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    }),
    {
      cacheControl: "no-store",
      xRobotsTag: "noindex, nofollow",
    }
  );
}

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const radiusKm = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return radiusKm * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function eqProjectX(lon, mapWidth) {
  return ((lon + 180) / 360) * mapWidth;
}

function eqProjectY(lat, mapHeight) {
  return ((90 - lat) / 180) * mapHeight;
}

function jsonResponse(body, status = 200, cacheControl = "public, max-age=300") {
  return withSecurityHeaders(
    new Response(JSON.stringify(body, null, 2), {
      status,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": cacheControl,
        "Access-Control-Allow-Origin": "*",
      },
    }),
    {
      cacheControl,
      xRobotsTag: "noindex, nofollow",
    }
  );
}

function sseResponse(eventName, payload) {
  const body = `event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`;
  return withSecurityHeaders(
    new Response(body, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Access-Control-Allow-Origin": "*",
      },
    }),
    {
      cacheControl: "no-cache",
      xRobotsTag: "noindex, nofollow",
    }
  );
}

function apiEnvelope(data, cacheTtl, extraMeta = {}) {
  return {
    meta: {
      version: API_VERSION,
      generated_at: new Date().toISOString(),
      cache_ttl: cacheTtl,
      ...extraMeta,
    },
    data,
  };
}

function errorEnvelope(code, message, status = 404) {
  return jsonResponse(
    {
      meta: {
        version: API_VERSION,
        generated_at: new Date().toISOString(),
      },
      error: { code, message },
    },
    status,
    "no-cache"
  );
}

function assetRequest(pathname) {
  return new Request(new URL(pathname, PRIMARY_DOMAIN).toString());
}

async function fetchAsset(env, pathname) {
  const request = assetRequest(pathname);
  try {
    const assetResponse = await env.ASSETS.fetch(request);
    if (assetResponse.ok) return assetResponse;
  } catch {}

  try {
    return await fetch(new Request(new URL(pathname, PRIMARY_DOMAIN).toString()));
  } catch {
    return new Response(null, { status: 404 });
  }
}

async function fetchAssetJson(env, pathname) {
  const response = await fetchAsset(env, pathname);
  if (!response.ok) return null;
  try {
    const text = await response.text();
    return JSON.parse(text.replace(/^\uFEFF/, ""));
  } catch {
    return null;
  }
}

async function fetchAssetText(env, pathname) {
  const response = await fetchAsset(env, pathname);
  if (!response.ok) return null;
  return response.text();
}

function computeThreatLevel(userLat, userLon, zones) {
  if (!zones.length) {
    return { alertLevel: "none", nearest: null, nearestDist: null };
  }

  let nearest = null;
  let nearestDist = Infinity;

  for (const zone of zones) {
    const dist = haversineKm(userLat, userLon, zone.lat, zone.lon);
    if (dist < nearestDist) {
      nearestDist = dist;
      nearest = zone;
    }
  }

  let alertLevel = "none";
  if (nearestDist < ALERT_THRESHOLDS.critical && nearest.prob >= 80) {
    alertLevel = "critical";
  } else if (nearestDist < ALERT_THRESHOLDS.severe && nearest.prob >= 51) {
    alertLevel = "severe";
  } else if (nearestDist < ALERT_THRESHOLDS.warning && nearest.prob >= 20) {
    alertLevel = "warning";
  } else if (nearestDist < ALERT_THRESHOLDS.watch && nearest.prob >= 5) {
    alertLevel = "watch";
  }

  return {
    alertLevel,
    nearest,
    nearestDist: nearest ? Math.round(nearestDist) : null,
  };
}

class BodyHandler {
  constructor(geo, threat) {
    this.geo = geo;
    this.threat = threat;
  }

  element(el) {
    try {
      const { geo, threat } = this;
      el.setAttribute("data-alert", threat.alertLevel);
      el.setAttribute("data-geo-valid", geo.isReliable ? "true" : "false");
      el.setAttribute(
        "data-lat",
        geo.isReliable && geo.latitude !== null ? geo.latitude : ""
      );
      el.setAttribute(
        "data-lon",
        geo.isReliable && geo.longitude !== null ? geo.longitude : ""
      );
      el.setAttribute("data-city", escapeHtml(geo.city) || "");
      el.setAttribute("data-country", escapeHtml(geo.country) || "");
      el.setAttribute("data-continent", escapeHtml(geo.continent) || "");
      if (threat.nearest) {
        el.setAttribute(
          "data-nearest-hazard",
          `${threat.nearest.type}:${escapeHtml(threat.nearest.name)}`
        );
      }

      const mapW = 960;
      const mapH = 480;
      const hasCoords =
        geo.isReliable && geo.latitude !== null && geo.longitude !== null;
      const ux = hasCoords ? eqProjectX(geo.longitude, mapW).toFixed(1) : "-9999";
      const uy = hasCoords ? eqProjectY(geo.latitude, mapH).toFixed(1) : "-9999";

      el.setAttribute(
        "style",
        `--user-x:${ux}px;--user-y:${uy}px;--user-lat:${
          hasCoords ? geo.latitude : ""
        };--user-lon:${hasCoords ? geo.longitude : ""}`
      );
    } catch {
      // noop
    }
  }
}

class EmergencyBannerHandler {
  constructor(threat, geo) {
    this.threat = threat;
    this.geo = geo;
  }

  element(el) {
    try {
      const { threat, geo } = this;
      if (!geo.isReliable) {
        return;
      }
      if (
        !threat.nearest ||
        threat.alertLevel === "none" ||
        threat.alertLevel === "watch" ||
        threat.alertLevel === "warning"
      ) {
        return;
      }

      const hz = threat.nearest;
      const typeLabel =
        hz.type === "hurricane"
          ? "Hurricane/Typhoon"
          : hz.type.charAt(0).toUpperCase() + hz.type.slice(1);
      const agencies = AGENCY_LINKS[hz.type] || "";

      const detailHtml = `<strong>${escapeHtml(typeLabel)}: ${escapeHtml(
        hz.name
      )}</strong> - ${hz.prob}% probability (${escapeHtml(
        hz.detail
      )}). You are approximately ${threat.nearestDist} km from this hazard zone.`;

      const titles = {
        critical: "Imminent danger - take action now",
        severe: "Significant hazard near your location",
        warning: "Hazard advisory for your region",
        watch: "Hazard being monitored near your area",
      };
      const actions = {
        critical: `A major ${escapeHtml(
          typeLabel.toLowerCase()
        )} signal is very close to your location. Follow official evacuation or shelter guidance immediately.`,
        severe: `A significant ${escapeHtml(
          typeLabel.toLowerCase()
        )} signal is near your location. Monitor official sources closely and be ready to act.`,
        warning: "An active hazard is in your broader region. Stay informed via official channels.",
        watch: "A hazard is being monitored in your general area. No action is needed yet.",
      };

      el.setInnerContent(
        `
        <div class="emergency-pulse"></div>
        <div class="emergency-body">
          <h2 class="emergency-title">${titles[threat.alertLevel] || "Hazard information"}</h2>
          <p class="emergency-location">Detected location: ${escapeHtml(
            geo.city || "Unknown"
          )}, ${escapeHtml(geo.country || "Unknown")}</p>
          <p class="emergency-detail">${detailHtml}</p>
          <p class="emergency-action">${actions[threat.alertLevel] || ""}</p>
          <div class="emergency-links">Official sources: ${agencies}</div>
        </div>
      `,
        { html: true }
      );
    } catch {
      // noop
    }
  }
}

class YourAreaHandler {
  constructor(threat, geo) {
    this.threat = threat;
    this.geo = geo;
  }

  element(el) {
    try {
      const { threat, geo } = this;
      if (!geo.isReliable) {
        el.setInnerContent(
          `
          <h2 id="your-area-heading">Your area</h2>
          <p class="muted">Approximate location is currently unavailable, so local distance estimates are hidden for this session.</p>
          <div class="card"><p class="muted">The global hazard view is still live. We only render a personal location marker when edge geolocation resolves cleanly.</p></div>
        `,
          { html: true }
        );
        return;
      }

      const city = escapeHtml(geo.city) || escapeHtml(geo.region) || "your area";
      const country = escapeHtml(geo.country) || "";
      const locationStr = country ? `${city}, ${country}` : city;

      if (!threat.nearest) {
        el.setInnerContent(
          `
          <h2 id="your-area-heading">Your area</h2>
          <p class="muted">We detected your approximate location as <strong>${locationStr}</strong>.</p>
          <div class="card"><p class="muted">No significant hazards are currently active near your location. Stay informed via official sources.</p></div>
        `,
          { html: true }
        );
        return;
      }

      const hz = threat.nearest;
      const typeLabel =
        hz.type === "hurricane"
          ? "Hurricane/Typhoon"
          : hz.type.charAt(0).toUpperCase() + hz.type.slice(1);
      const distStr =
        threat.nearestDist < 100
          ? `${threat.nearestDist} km`
          : `~${Math.round(threat.nearestDist / 10) * 10} km`;

      el.setInnerContent(
        `
        <h2 id="your-area-heading">Your area - ${locationStr}</h2>
        <p class="muted">Based on your approximate location. Distances are estimates.</p>
        <div class="grid">
          <div class="card col-4">
            <div class="metric">${distStr}</div>
            <div class="metric-label">to nearest active hazard</div>
            <div class="kv"><span>Hazard</span><strong>${escapeHtml(typeLabel)}: ${escapeHtml(
              hz.name
            )}</strong></div>
            <div class="kv"><span>Probability</span><strong>${hz.prob}%</strong></div>
            <div class="kv"><span>Detail</span><strong>${escapeHtml(hz.detail)}</strong></div>
          </div>
          <div class="card col-4">
            <div class="kv"><span>Your location</span><strong>${locationStr}</strong></div>
            <div class="kv"><span>Alert level</span><strong style="text-transform:capitalize;">${escapeHtml(
              threat.alertLevel
            )}</strong></div>
            <div class="kv"><span>Continent</span><strong>${escapeHtml(
              geo.continent || "Unknown"
            )}</strong></div>
            <div class="kv"><span>Timezone</span><strong>${escapeHtml(
              geo.timezone || "Unknown"
            )}</strong></div>
          </div>
          <div class="card col-4">
            <h3>What to do</h3>
            <p class="muted">${
              threat.alertLevel === "critical" || threat.alertLevel === "severe"
                ? "Monitor official sources. Have an emergency plan ready. Follow all government directives for your area."
                : "No immediate action is needed. Stay aware of conditions and rely on official guidance."
            }</p>
            <a href="${escapeHtml(hz.link)}" class="btn btn-secondary" style="margin-top:8px;">View full detail &rarr;</a>
          </div>
        </div>
      `,
        { html: true }
      );
    } catch {
      // noop
    }
  }
}

async function loadLiveEarthquakeZones(env) {
  const pulse = await fetchAssetJson(env, "/data/live-pulse.json");
  if (!pulse || !Array.isArray(pulse.hazards)) return [];

  const eq = pulse.hazards.find((hazard) => hazard.key === "eq");
  const forecastId = eq && eq.forecast_id;
  if (!forecastId) return [];

  const replay = await fetchAssetJson(env, `/data/replay/${forecastId}.json`);
  if (!replay || !Array.isArray(replay.active_cells)) return [];

  return replay.active_cells.slice(0, 12).map((cell, index) => ({
    lat: Number(cell.lat || 0),
    lon: Number(cell.lon || 0),
    r: 250,
    type: "earthquake",
    name: `Hotspot ${index + 1} (${Number(cell.lat || 0).toFixed(1)}°, ${Number(
      cell.lon || 0
    ).toFixed(1)}°)`,
    prob: Math.round(Number(cell.probability || 0) * 1000) / 10,
    country: "",
    detail: "M6.0+ in 30 days",
    link: "/live/earthquake/",
  }));
}

async function loadLiveHurricaneZones(env) {
  const data = await fetchAssetJson(env, "/data/live-storms.json");
  if (!data || !Array.isArray(data.storms)) return [];

  return data.storms.map((storm, index) => ({
    lat: Number(storm.lat || 0),
    lon: Number(storm.lon || 0),
    r: 300,
    type: "hurricane",
    name: `${storm.category || ""} ${storm.storm_name || storm.storm_id || "Storm"}`.trim(),
    prob: Math.round(Number(storm.ri_probability || 0) * 1000) / 10,
    country: storm.basin || "",
    detail: "Rapid intensification in 24 hours",
    link: "/live/hurricane/",
  }));
}

async function loadLiveTornadoZones(env) {
  const data = await fetchAssetJson(env, "/data/live-tornadoes.json");
  if (!data || !Array.isArray(data.storms)) return [];

  return data.storms.slice(0, 20).map((storm) => ({
    lat: Number(storm.lat || 0),
    lon: Number(storm.lon || 0),
    r: 120,
    type: "tornado",
    name: `Storm ${storm.storm_id || "--"}`,
    prob: Math.round(Number(storm.tornado_probability || 0) * 1000) / 10,
    country: "",
    detail: "Tornado formation in 24 hours",
    link: "/live/tornado/",
  }));
}

async function buildOpsSnapshot(env) {
  const pulse = await fetchAssetJson(env, "/data/live-pulse.json");
  const storms = await fetchAssetJson(env, "/data/live-storms.json");
  const tornadoes = await fetchAssetJson(env, "/data/live-tornadoes.json");

  return {
    status: "ok",
    domain: PRIMARY_DOMAIN,
    updated_at: pulse && pulse.updated_at ? pulse.updated_at : null,
    hazard_count: pulse && Array.isArray(pulse.hazards) ? pulse.hazards.length : 0,
    active_hurricanes: storms && typeof storms.n_active_storms === "number" ? storms.n_active_storms : 0,
    active_tornado_objects:
      tornadoes && typeof tornadoes.n_active_storms === "number" ? tornadoes.n_active_storms : 0,
  };
}

async function handleApiRequest(request, env) {
  const url = new URL(request.url);
  const path = url.pathname;

  if (path === "/api/v1/live/pulse") {
    const pulse = await fetchAssetJson(env, "/data/live-pulse.json");
    if (!pulse) return errorEnvelope("not_found", "Live pulse data is unavailable.");
    return jsonResponse(apiEnvelope(pulse, 300), 200, "public, max-age=300");
  }

  const liveMatch = path.match(/^\/api\/v1\/live\/(earthquake|hurricane|tornado)$/);
  if (liveMatch) {
    const hazard = liveMatch[1];
    if (hazard === "hurricane") {
      const storms = await fetchAssetJson(env, "/data/live-storms.json");
      if (!storms) return errorEnvelope("not_found", "Live hurricane data is unavailable.");
      return jsonResponse(apiEnvelope(storms, 300, { hazard }), 200, "public, max-age=300");
    }
    if (hazard === "tornado") {
      const tornadoes = await fetchAssetJson(env, "/data/live-tornadoes.json");
      if (!tornadoes) return errorEnvelope("not_found", "Live tornado data is unavailable.");
      return jsonResponse(apiEnvelope(tornadoes, 300, { hazard }), 200, "public, max-age=300");
    }

    const pulse = await fetchAssetJson(env, "/data/live-pulse.json");
    if (!pulse || !Array.isArray(pulse.hazards)) {
      return errorEnvelope("not_found", "Live earthquake data is unavailable.");
    }
    const eq = pulse.hazards.find((item) => item.key === "eq") || null;
    const forecastId = eq && eq.forecast_id;
    const replay = forecastId
      ? await fetchAssetJson(env, `/data/replay/${forecastId}.json`)
      : null;
    return jsonResponse(
      apiEnvelope({ summary: eq, replay }, 300, { hazard }),
      200,
      "public, max-age=300"
    );
  }

  const forecastMatch = path.match(/^\/api\/v1\/forecast\/([A-Za-z0-9_.-]+)$/);
  if (forecastMatch) {
    const forecastId = forecastMatch[1];
    const artifact = await fetchAssetJson(env, `/data/replay/${forecastId}.json`);
    if (!artifact) {
      return errorEnvelope("not_found", `Forecast ${forecastId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(artifact, 31536000, { forecast_id: forecastId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  const replayMatch = path.match(/^\/api\/v1\/replay\/([A-Za-z0-9_.-]+)$/);
  if (replayMatch) {
    const forecastId = replayMatch[1];
    const artifact = await fetchAssetJson(env, `/data/replay/${forecastId}.json`);
    if (!artifact) {
      return errorEnvelope("not_found", `Replay artifact ${forecastId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(artifact, 31536000, { forecast_id: forecastId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  const evidenceMatch = path.match(/^\/api\/v1\/evidence\/([A-Za-z0-9_.-]+)$/);
  if (evidenceMatch) {
    const provenanceId = evidenceMatch[1];
    const envelopes = await fetchAssetJson(env, "/data/evidence/provenance-envelopes.json");
    const match =
      envelopes &&
      Array.isArray(envelopes.envelopes) &&
      envelopes.envelopes.find((item) => item.provenance_id === provenanceId);
    if (!match) {
      return errorEnvelope("not_found", `Provenance envelope ${provenanceId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(match, 31536000, { provenance_id: provenanceId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  const gatesMatch = path.match(/^\/api\/v1\/gates\/([A-Za-z0-9_.-]+)$/);
  if (gatesMatch) {
    const gateDecisionId = gatesMatch[1];
    const gates = await fetchAssetJson(env, "/data/evidence/gate-decisions.json");
    const match =
      gates &&
      Array.isArray(gates.decisions) &&
      gates.decisions.find((item) => item.gate_decision_id === gateDecisionId);
    if (!match) {
      return errorEnvelope("not_found", `Gate decision ${gateDecisionId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(match, 31536000, { gate_decision_id: gateDecisionId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  if (path === "/api/v1/verification/summary") {
    const summary = await fetchAssetJson(env, "/data/verification-summary.json");
    if (!summary) return errorEnvelope("not_found", "Verification summary is unavailable.");
    return jsonResponse(apiEnvelope(summary, 3600), 200, "public, max-age=3600");
  }

  if (path === "/api/v1/registry/models") {
    const registry = await fetchAssetJson(env, "/data/model-registry.json");
    if (!registry) return errorEnvelope("not_found", "Model registry is unavailable.");
    return jsonResponse(apiEnvelope(registry, 3600), 200, "public, max-age=3600");
  }

  if (path === "/api/v1/ops/status") {
    const snapshot = await buildOpsSnapshot(env);
    return jsonResponse(apiEnvelope(snapshot, 300), 200, "public, max-age=300");
  }

  return null;
}

async function handleStreamRequest(request, env) {
  const path = new URL(request.url).pathname;

  if (path === "/stream/live/pulse") {
    const pulse = await fetchAssetJson(env, "/data/live-pulse.json");
    if (!pulse) return errorEnvelope("not_found", "Live pulse data is unavailable.");
    return sseResponse("live_pulse", apiEnvelope(pulse, 300));
  }

  if (path === "/stream/ops/status") {
    const snapshot = await buildOpsSnapshot(env);
    return sseResponse("ops_status", apiEnvelope(snapshot, 300));
  }

  return null;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    if (pathname === "/commercial-license" || pathname === "/commercial-license/") {
      return withSecurityHeaders(
        Response.redirect(`${PRIMARY_DOMAIN}/COMMERCIAL_LICENSE.md`, 302),
        {
          cacheControl: "public, max-age=3600",
          xRobotsTag: "noindex, nofollow",
        }
      );
    }

    if (pathname.startsWith("/api/")) {
      const apiResponse = await handleApiRequest(request, env);
      return apiResponse || errorEnvelope("not_found", "Unknown API endpoint.");
    }

    if (pathname.startsWith("/stream/")) {
      const streamResponse = await handleStreamRequest(request, env);
      return streamResponse || errorEnvelope("not_found", "Unknown stream endpoint.");
    }

    if (
      pathname.startsWith("/assets/") ||
      pathname.startsWith("/data/") ||
      pathname.endsWith(".xml") ||
      pathname.endsWith(".txt") ||
      pathname.endsWith(".json") ||
      pathname.endsWith(".md")
    ) {
      const assetResponse = await env.ASSETS.fetch(request);
      if (!assetResponse.ok) return notFoundResponse(env, request);
      const xRobotsTag =
        pathname.startsWith("/data/") ||
        pathname.endsWith(".json") ||
        pathname.endsWith(".md")
          ? "noindex, nofollow"
          : undefined;
      return withSecurityHeaders(assetResponse, { xRobotsTag });
    }

    let response;
    try {
      response = await env.ASSETS.fetch(request);
    } catch {
      return withSecurityHeaders(
        new Response("Service temporarily unavailable", {
          status: 502,
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        }),
        {
          cacheControl: "no-store",
          xRobotsTag: "noindex, nofollow",
        }
      );
    }

    if (!response.ok) {
      return notFoundResponse(env, request);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/html")) {
      return withSecurityHeaders(response);
    }

    const geo = normalizeGeo(request.cf || {});

    const [earthquakeZones, hurricaneZones, tornadoZones] = await Promise.all([
      loadLiveEarthquakeZones(env),
      loadLiveHurricaneZones(env),
      loadLiveTornadoZones(env),
    ]);
    const allZones = [...earthquakeZones, ...hurricaneZones, ...tornadoZones];

    let threat = { alertLevel: "none", nearest: null, nearestDist: null };
    if (geo.isReliable) {
      threat = computeThreatLevel(geo.latitude, geo.longitude, allZones);
    }

    const transformed = new HTMLRewriter()
      .on("body", new BodyHandler(geo, threat))
      .on(".emergency-banner", new EmergencyBannerHandler(threat, geo))
      .on(".your-area-section", new YourAreaHandler(threat, geo))
      .transform(response);

    return withSecurityHeaders(new Response(transformed.body, transformed), {
      cacheControl: HTML_CACHE_CONTROL,
      xRobotsTag: "index, follow",
      contentSecurityPolicy: HTML_CONTENT_SECURITY_POLICY,
      vary: "CF-IPCountry, Accept-Encoding",
    });
  },
};

if (typeof globalThis !== "undefined") {
  globalThis.__hazardpulse_worker_test = {
    BodyHandler,
    YourAreaHandler,
    hasReliableGeo,
    normalizeGeo,
  };
}
