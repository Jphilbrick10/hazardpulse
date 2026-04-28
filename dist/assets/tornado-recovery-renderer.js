/**
 * Renders the per-tier AUC recovery curve from
 * /api/v1/verification/tornado-recovery.
 *
 * Mounts into #tornado-recovery-panel. Static fallback says "no data yet"
 * until the verification scorer publishes its first run.
 */
(function () {
  "use strict";

  function _esc(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function _fmtAuc(v) {
    if (v == null) return '<span class="muted">&mdash;</span>';
    return Number(v).toFixed(3);
  }

  function _fmtN(v) {
    if (v == null) return "&mdash;";
    return String(v);
  }

  function render(payload) {
    var mount = document.getElementById("tornado-recovery-panel");
    if (!mount) return;

    var byTierRec = payload.by_tier_recovery || {};
    var byTier = payload.by_tier || {};
    var tiers = Object.keys(byTierRec).sort();

    if (tiers.length === 0) {
      mount.innerHTML =
        '<p class="muted">Waiting for first verification run after the recent fix. ' +
        "Per-tier recovery curve will populate within a few hours.</p>";
      return;
    }

    var headRow =
      "<tr>" +
      "<th>Tier</th>" +
      "<th>last 24h</th>" +
      "<th>last 3d</th>" +
      "<th>last 7d</th>" +
      "<th>last 14d</th>" +
      "<th>all time</th>" +
      "</tr>";

    function _cell(buckets, label) {
      var b = buckets[label] || {};
      if (b.n_forecasts == null || b.n_forecasts === 0) {
        return '<td class="muted">no data</td>';
      }
      return (
        '<td class="mono">AUC=' + _fmtAuc(b.mean_auc) +
        '<br><span class="muted">n=' + _fmtN(b.n_forecasts) + "</span></td>"
      );
    }

    var bodyRows = tiers.map(function (tier) {
      var rec = byTierRec[tier] || {};
      var allTime = byTier[tier] || {};
      return (
        "<tr>" +
        "<td>" + _esc(tier) + "</td>" +
        _cell(rec, "last_24h") +
        _cell(rec, "last_3d") +
        _cell(rec, "last_7d") +
        _cell(rec, "last_14d") +
        '<td class="mono">AUC=' + _fmtAuc(allTime.mean_auc) +
        '<br><span class="muted">n=' + _fmtN(allTime.n_forecasts) + "</span></td>" +
        "</tr>"
      );
    });

    var fix = payload.fix_landed_at;
    var fixLine = fix
      ? '<p class="muted" style="margin-top:8px;">Last model fix landed ' + _esc(fix) +
        ". Forecasts issued after that timestamp roll into the recency buckets."
      : "";

    mount.innerHTML =
      '<table class="data-table"><thead>' + headRow + "</thead><tbody>" +
      bodyRows.join("") + "</tbody></table>" + fixLine;
  }

  function init() {
    var mount = document.getElementById("tornado-recovery-panel");
    if (!mount) return;
    fetch("/api/v1/verification/tornado-recovery")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (envelope) {
        var data = envelope && envelope.data ? envelope.data : envelope;
        render(data);
      })
      .catch(function () {
        mount.innerHTML =
          '<p class="muted">Recovery curve unavailable. The verification ' +
          "scorer publishes /data/tornado-recovery.json every 4 hours.</p>";
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
