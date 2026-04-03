import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const workerSource = readFileSync(path.join(root, "src", "worker.js"), "utf8");
const transformedSource = workerSource.replace(
  "export default",
  "globalThis.__worker_default ="
);

class HTMLRewriterStub {
  on() {
    return this;
  }

  transform(response) {
    return response;
  }
}

function mimeTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".xml") return "application/xml; charset=utf-8";
  if (ext === ".txt") return "text/plain; charset=utf-8";
  if (ext === ".md") return "text/markdown; charset=utf-8";
  if (ext === ".svg") return "image/svg+xml";
  if (ext === ".png") return "image/png";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".js") return "application/javascript; charset=utf-8";
  return "application/octet-stream";
}

function resolveAssetPath(urlPath) {
  let normalized = urlPath;
  if (!normalized || normalized === "/") normalized = "/index.html";
  if (normalized.endsWith("/")) normalized += "index.html";
  return path.join(root, "dist", ...normalized.split("/").filter(Boolean));
}

const context = vm.createContext({
  console,
  URL,
  Headers,
  Request,
  Response,
  HTMLRewriter: HTMLRewriterStub,
  fetch: async (input) => {
    const request = input instanceof Request ? input : new Request(input);
    const url = new URL(request.url);
    if (url.hostname !== "hazardpulse.com") {
      throw new Error(`unexpected fetch host: ${url.hostname}`);
    }
    const filePath = resolveAssetPath(url.pathname);
    if (!existsSync(filePath)) {
      return new Response("not found", { status: 404 });
    }
    return new Response(readFileSync(filePath), {
      status: 200,
      headers: { "Content-Type": mimeTypeFor(filePath) },
    });
  },
  globalThis: {},
});

vm.runInContext(transformedSource, context, { filename: "worker.js" });
const worker = context.globalThis.__worker_default;
const workerTest = context.globalThis.__hazardpulse_worker_test;
assert(worker && typeof worker.fetch === "function");
assert(workerTest);

const env = {
  ASSETS: {
    async fetch(request) {
      const url = new URL(request.url);
      if (url.hostname !== "hazardpulse.com") {
        throw new Error(`unexpected asset host: ${url.hostname}`);
      }
      const filePath = resolveAssetPath(url.pathname);
      if (!existsSync(filePath)) {
        return new Response("not found", { status: 404 });
      }
      return new Response(readFileSync(filePath), {
        status: 200,
        headers: { "Content-Type": mimeTypeFor(filePath) },
      });
    },
  },
};

async function expectJsonRoute(pathname, predicate) {
  const response = await worker.fetch(new Request(`https://hazardpulse.com${pathname}`), env);
  assert.equal(response.status, 200, `${pathname} should return 200`);
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.equal(response.headers.get("X-Frame-Options"), "DENY");
  assert.equal(response.headers.get("X-Robots-Tag"), "noindex, nofollow");
  const json = await response.json();
  assert.equal(json.meta.version, "v1");
  predicate(json.data);
}

function assertHtmlSecurityHeaders(
  response,
  expectedStatus = 200,
  expectedCacheControl = "private, no-cache, no-store, must-revalidate"
) {
  assert.equal(response.status, expectedStatus);
  assert.match(response.headers.get("Content-Security-Policy") || "", /frame-ancestors 'none'/);
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.equal(response.headers.get("X-Frame-Options"), "DENY");
  assert.equal(response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin");
  assert.equal(response.headers.get("Cache-Control"), expectedCacheControl);
}

const homeResponse = await worker.fetch(new Request("https://hazardpulse.com/"), env);
assertHtmlSecurityHeaders(homeResponse);
assert.match(await homeResponse.text(), /HazardPulse/);

const siteShellResponse = await worker.fetch(
  new Request("https://hazardpulse.com/assets/site-shell.js"),
  env
);
assert.equal(siteShellResponse.status, 200);
assert.match(siteShellResponse.headers.get("Content-Type") || "", /javascript/);

assert.equal(
  workerTest.hasReliableGeo({
    latitude: 0,
    longitude: 0,
    city: null,
    country: null,
    region: null,
    timezone: null,
    continent: null,
  }),
  false
);
assert.equal(
  workerTest.hasReliableGeo({
    latitude: 40.7128,
    longitude: -74.006,
    city: "New York",
    country: "US",
    region: "New York",
    timezone: "America/New_York",
    continent: "NA",
  }),
  true
);

assert.equal(
  workerTest.readThemePreference(
    new Request("https://hazardpulse.com/", {
      headers: { cookie: "hp_theme=dark; foo=bar" },
    })
  ),
  "dark"
);

const invalidGeo = workerTest.normalizeGeo({ latitude: "0", longitude: "0" });
const invalidBodyAttrs = new Map();
new workerTest.BodyHandler(invalidGeo, {
  alertLevel: "none",
  nearest: null,
  nearestDist: null,
}).element({
  setAttribute(name, value) {
    invalidBodyAttrs.set(name, value);
  },
});
assert.equal(invalidBodyAttrs.get("data-geo-valid"), "false");
assert.equal(invalidBodyAttrs.get("data-lat"), "");
assert.equal(invalidBodyAttrs.get("data-lon"), "");
assert.match(invalidBodyAttrs.get("style") || "", /--user-x:-9999px/);

const validGeo = workerTest.normalizeGeo({
  latitude: "40.7128",
  longitude: "-74.0060",
  city: "New York",
  country: "US",
  region: "New York",
  timezone: "America/New_York",
  continent: "NA",
});

const markerAttrs = new Map();
new workerTest.UserMarkerHandler(validGeo).element({
  setAttribute(name, value) {
    markerAttrs.set(name, value);
  },
});
assert.match(markerAttrs.get("transform") || "", /^translate\(\d+\.\d \d+\.\d\)$/);
assert.match(markerAttrs.get("aria-label") || "", /Your approximate location/);

const toggleAttrs = new Map();
new workerTest.ThemeToggleHandler("dark").element({
  setAttribute(name, value) {
    toggleAttrs.set(name, value);
  },
});
assert.equal(toggleAttrs.get("checked"), "checked");
assert.equal(toggleAttrs.get("aria-label"), "Switch to light mode");

let unavailableAreaHtml = "";
new workerTest.YourAreaHandler(
  { alertLevel: "none", nearest: null, nearestDist: null },
  invalidGeo
).element({
  setInnerContent(html) {
    unavailableAreaHtml = html;
  },
});
assert.match(unavailableAreaHtml, /Approximate location is currently unavailable/);

await expectJsonRoute("/api/v1/live/pulse", (data) => {
  assert.ok(Array.isArray(data.hazards));
});

await expectJsonRoute("/api/v1/live/hurricane", (data) => {
  assert.ok("n_active_storms" in data);
});

await expectJsonRoute("/api/v1/live/tornado", (data) => {
  assert.ok(Array.isArray(data.storms));
});

await expectJsonRoute("/api/v1/live/earthquake", (data) => {
  assert.ok(data.summary);
});

await expectJsonRoute("/api/v1/forecast/eq_fcst_20260402_0000", (data) => {
  assert.equal(data.forecast_id, "eq_fcst_20260402_0000");
});

await expectJsonRoute("/api/v1/verification/summary", (data) => {
  assert.ok(Array.isArray(data.hazards));
});

await expectJsonRoute("/api/v1/registry/models", (data) => {
  assert.ok(Array.isArray(data.models));
});

const provenancePayload = JSON.parse(
  readFileSync(path.join(root, "dist", "data", "evidence", "provenance-envelopes.json"), "utf8")
);
if (Array.isArray(provenancePayload.envelopes) && provenancePayload.envelopes.length > 0) {
  const firstEnvelope = provenancePayload.envelopes[0];
  await expectJsonRoute(`/api/v1/evidence/${firstEnvelope.provenance_id}`, (data) => {
    assert.equal(data.provenance_id, firstEnvelope.provenance_id);
  });
}

const gatePayload = JSON.parse(
  readFileSync(path.join(root, "dist", "data", "evidence", "gate-decisions.json"), "utf8")
);
if (Array.isArray(gatePayload.decisions) && gatePayload.decisions.length > 0) {
  const firstDecision = gatePayload.decisions[0];
  await expectJsonRoute(`/api/v1/gates/${firstDecision.gate_decision_id}`, (data) => {
    assert.equal(data.gate_decision_id, firstDecision.gate_decision_id);
  });
}

const sseResponse = await worker.fetch(
  new Request("https://hazardpulse.com/stream/live/pulse"),
  env
);
assert.equal(sseResponse.status, 200);
assert.match(sseResponse.headers.get("Content-Type") || "", /text\/event-stream/);
assert.equal(sseResponse.headers.get("X-Robots-Tag"), "noindex, nofollow");
assert.match(await sseResponse.text(), /event: live_pulse/);

const redirectResponse = await worker.fetch(
  new Request("https://hazardpulse.com/commercial-license/"),
  env
);
assert.equal(redirectResponse.status, 302);
assert.match(redirectResponse.headers.get("Location") || "", /COMMERCIAL_LICENSE\.md$/);
assert.equal(redirectResponse.headers.get("X-Robots-Tag"), "noindex, nofollow");

const missingResponse = await worker.fetch(
  new Request("https://hazardpulse.com/does-not-exist"),
  env
);
assertHtmlSecurityHeaders(missingResponse, 404, "no-store");
assert.equal(missingResponse.headers.get("X-Robots-Tag"), "noindex, nofollow");
assert.match(await missingResponse.text(), /404 - Page not found|Not found/);

const unknownApiResponse = await worker.fetch(
  new Request("https://hazardpulse.com/api/v1/unknown"),
  env
);
assert.equal(unknownApiResponse.status, 404);
const unknownApiJson = await unknownApiResponse.json();
assert.equal(unknownApiJson.error.code, "not_found");
assert.ok(unknownApiJson.error.trace_id);

console.log("worker api smoke checks passed");
