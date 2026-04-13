/**
 * HazardPulse Hurricane Monitor — MapLibre GL global tropical cyclone map.
 *
 * Loads hurricane-storms.geojson and renders interactive markers with
 * popups showing storm name, category, wind, pressure, and RI probability.
 * Color-coded by Saffir-Simpson category for immediate visual assessment.
 */
(function () {
  "use strict";

  var CATEGORY_COLORS = {
    "Category 5": "#8B0000",
    "Category 4": "#DC143C",
    "Category 3": "#FF6600",
    "Category 2": "#FFB800",
    "Category 1": "#66BB6A",
    "Tropical Storm": "#42A5F5",
    "Tropical Depression": "#90CAF9",
    "Unknown": "#78909C",
  };

  var BASIN_LABELS = {
    AL: "Atlantic",
    EP: "East Pacific",
    CP: "Central Pacific",
    WP: "West Pacific",
    IO: "Indian Ocean",
    SH: "Southern Hemisphere",
  };

  function colorForCategory(cat) {
    return CATEGORY_COLORS[cat] || CATEGORY_COLORS["Unknown"];
  }

  function sizeForWind(vmax) {
    if (vmax == null) return 14;
    if (vmax >= 137) return 32;
    if (vmax >= 96) return 26;
    if (vmax >= 64) return 22;
    if (vmax >= 34) return 18;
    return 14;
  }

  function init() {
    var mapEl = document.getElementById("hurricane-map");
    if (!mapEl || typeof maplibregl === "undefined") return;

    var geojsonSrc = mapEl.getAttribute("data-geojson-src") || "/data/hurricane-storms.geojson";

    var map = new maplibregl.Map({
      container: "hurricane-map",
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>",
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
      },
      center: [10, 15],
      zoom: 1.8,
      minZoom: 1,
      maxZoom: 12,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    fetch(geojsonSrc)
      .then(function (r) { return r.json(); })
      .then(function (geojson) {
        var features = geojson.features || [];

        if (features.length === 0) {
          var emptyDiv = document.createElement("div");
          emptyDiv.className = "map-empty-overlay";
          emptyDiv.innerHTML =
            '<p class="muted" style="text-align:center;padding:12px 0;margin:0;">' +
            "No active tropical cyclones in any basin.</p>";
          mapEl.appendChild(emptyDiv);
          return;
        }

        features.forEach(function (feature) {
          var props = feature.properties || {};
          var coords = feature.geometry && feature.geometry.coordinates;
          if (!coords) return;

          var cat = props.category || "Unknown";
          var vmax = props.vmax_kt;
          var riProb = props.ri_probability || 0;
          var basin = BASIN_LABELS[props.basin] || props.basin || "";
          var size = sizeForWind(vmax);

          var el = document.createElement("div");
          el.style.width = size + "px";
          el.style.height = size + "px";
          el.style.borderRadius = "50%";
          el.style.backgroundColor = colorForCategory(cat);
          el.style.border = "2px solid rgba(255,255,255,0.6)";
          el.style.boxShadow = "0 0 6px rgba(0,0,0,0.3)";
          el.style.cursor = "pointer";
          el.setAttribute("role", "button");
          el.setAttribute("tabindex", "0");
          el.setAttribute(
            "aria-label",
            _esc(props.storm_name || props.storm_id) + " " + _esc(cat)
          );

          var popupHtml =
            '<div style="font-family:var(--font-mono,monospace);font-size:13px;line-height:1.5;">' +
            "<strong>" + _esc(props.storm_name || props.storm_id) + "</strong>" +
            ' <span style="opacity:0.6;">(' + _esc(basin) + ")</span><br>" +
            '<span style="color:' + _esc(colorForCategory(cat)) + ';">' + _esc(cat) + "</span><br>" +
            "Wind: <strong>" + _esc(vmax != null ? vmax + " kt" : "--") + "</strong><br>" +
            "Pressure: " + _esc(props.mslp_hpa != null ? props.mslp_hpa + " hPa" : "--") + "<br>" +
            "RI 24h: <strong>" + (riProb * 100).toFixed(1) + "%</strong>" +
            "</div>";

          var popup = new maplibregl.Popup({ offset: 12, maxWidth: "240px" }).setHTML(popupHtml);

          new maplibregl.Marker({ element: el })
            .setLngLat(coords)
            .setPopup(popup)
            .addTo(map);
        });

        // Fit bounds to all storms with padding
        if (features.length > 1) {
          var bounds = new maplibregl.LngLatBounds();
          features.forEach(function (f) {
            bounds.extend(f.geometry.coordinates);
          });
          map.fitBounds(bounds, { padding: 60, maxZoom: 6 });
        } else if (features.length === 1) {
          map.flyTo({
            center: features[0].geometry.coordinates,
            zoom: 5,
          });
        }
      })
      .catch(function () {
        mapEl.innerHTML =
          '<div style="text-align:center;padding:48px 16px;">' +
          '<p class="muted">Hurricane map data unavailable.</p></div>';
      });
  }

  function _esc(s) {
    if (!s) return "";
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
