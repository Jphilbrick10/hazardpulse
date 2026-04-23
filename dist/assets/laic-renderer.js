/**
 * Renders the LAIC cross-modality analyses table from
 * /api/v1/laic/summary into #laic-table-wrapper.
 *
 * Zero JS frameworks; runs against the static API endpoint emitted by
 * scripts/run_cross_modality_analyses.py.
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

  function _fmtFloat(v, digits) {
    if (v == null || Number.isNaN(v)) return "&mdash;";
    return Number(v).toFixed(digits || 3);
  }

  function _significance(p) {
    if (p == null || Number.isNaN(p)) return "";
    if (p < 0.001) return "***";
    if (p < 0.01) return "**";
    if (p < 0.05) return "*";
    return "";
  }

  function _hazardDot(hazard) {
    var cls = { earthquake: "eq", hurricane: "hu", tornado: "to", cme: "hu" }[hazard] || "";
    return cls ? '<span class="hazard-dot ' + cls + '"></span> ' : "";
  }

  function render(payload) {
    var wrapper = document.getElementById("laic-table-wrapper");
    if (!wrapper) return;

    var generatedAt = document.getElementById("generated-at");
    if (generatedAt && payload.generated_at) {
      generatedAt.textContent = "Generated " + payload.generated_at +
        " · " + payload.n_analyses + " analyses · " +
        payload.n_eq_events_used + " EQ events";
    }

    var rows = (payload.analyses || []).map(function (a) {
      if (a.status === "needs_curated_input") {
        return (
          "<tr>" +
          "<td>" + _hazardDot(a.hazard) + _esc(a.analysis_id) + "</td>" +
          '<td colspan="6" class="muted">' + _esc(a.notes || "Pending curated input list") + "</td>" +
          "</tr>"
        );
      }
      var delta = a.delta_mean;
      var p = a.welch_p_two_sided != null ? a.welch_p_two_sided : a.welch_p;
      var ciLo = a.bootstrap_ci_delta_lo != null ? a.bootstrap_ci_delta_lo : a.bootstrap_ci_lo;
      var ciHi = a.bootstrap_ci_delta_hi != null ? a.bootstrap_ci_delta_hi : a.bootstrap_ci_hi;
      var sig = _significance(p);
      var sigCell = sig ? '<span class="badge">' + sig + "</span>" : "&mdash;";
      var n = (a.n_targets != null ? a.n_targets : a.n_ri_events != null ? a.n_ri_events : a.n_tornado_events) || 0;
      return (
        "<tr>" +
        "<td>" + _hazardDot(a.hazard) + _esc(a.analysis_id) + "</td>" +
        '<td class="mono">' + _esc(a.feature || "&mdash;") + "</td>" +
        '<td class="mono">' + n + "</td>" +
        '<td class="mono">' + _fmtFloat(delta, 3) + "</td>" +
        '<td class="mono">[' + _fmtFloat(ciLo, 3) + ", " + _fmtFloat(ciHi, 3) + "]</td>" +
        '<td class="mono">' + _fmtFloat(p, 3) + "</td>" +
        "<td>" + sigCell + "</td>" +
        "</tr>"
      );
    });

    wrapper.innerHTML =
      '<table class="data-table"><thead><tr>' +
      "<th>Analysis</th><th>Feature</th><th>n</th><th>&Delta; mean</th>" +
      "<th>95% CI</th><th>p</th><th>sig</th>" +
      "</tr></thead><tbody>" + rows.join("") + "</tbody></table>" +
      '<p class="muted" style="margin-top:8px;">' +
      "*** p&lt;0.001  ** p&lt;0.01  * p&lt;0.05" +
      "</p>";
  }

  function init() {
    fetch("/api/v1/laic/summary")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (envelope) {
        var data = envelope && envelope.data ? envelope.data : envelope;
        render(data);
      })
      .catch(function () {
        var wrapper = document.getElementById("laic-table-wrapper");
        if (wrapper) {
          wrapper.innerHTML = '<p class="muted">' +
            "No LAIC results published yet. The cross-modality analyses workflow runs " +
            "weekly on Monday 11:30 UTC. Trigger manually via GitHub Actions to seed." +
            "</p>";
        }
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
