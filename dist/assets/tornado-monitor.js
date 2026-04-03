(function () {
  function renderFallback(mapEl) {
    mapEl.innerHTML =
      '<div class="card" style="text-align:center;padding:40px;">' +
      "<h3>Map temporarily unavailable</h3>" +
      '<p class="muted">Storm data is still available in the list below.</p>' +
      "</div>";
  }

  function init() {
    var mapEl = document.getElementById("tornado-map");
    if (!mapEl || typeof maplibregl === "undefined") return;

    var geojsonSrc = mapEl.getAttribute("data-geojson-src") || "/data/tornado-storms.geojson";
    var map = new maplibregl.Map({
      container: "tornado-map",
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "&copy; OpenStreetMap contributors",
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
      },
      center: [-95, 38],
      zoom: 4,
      maxZoom: 12,
    });

    map.on("error", function () {
      renderFallback(mapEl);
    });

    fetch(geojsonSrc)
      .then(function (response) {
        if (!response.ok) throw new Error("geojson unavailable");
        return response.json();
      })
      .then(function (geojson) {
        (geojson.features || []).forEach(function (feature) {
          var props = feature.properties || {};
          var probability = props.probability || 0;
          var coords = feature.geometry && feature.geometry.coordinates;
          if (!coords) return;

          var marker = document.createElement("div");
          marker.style.width = 12 + probability * 30 + "px";
          marker.style.height = 12 + probability * 30 + "px";
          marker.style.borderRadius = "50%";
          marker.style.backgroundColor =
            probability > 0.3 ? "#EF4444" : probability > 0.15 ? "#F59E0B" : "#14B8A6";
          marker.style.border = "2px solid rgba(255,255,255,0.3)";
          marker.style.cursor = "pointer";
          marker.setAttribute("role", "button");
          marker.setAttribute(
            "aria-label",
            "Storm " +
              props.storm_id +
              ", " +
              (probability * 100).toFixed(0) +
              "% tornado probability"
          );
          marker.setAttribute("tabindex", "0");

          var popup = new maplibregl.Popup({ offset: 15 }).setHTML(
            "<strong>Storm " +
              props.storm_id +
              "</strong><br>" +
              'Probability: <span class="mono">' +
              (probability * 100).toFixed(1) +
              "%</span><br>" +
              'CAPE: <span class="mono">' +
              (props.mucape || 0) +
              "</span> J/kg<br>" +
              'SRH: <span class="mono">' +
              (props.srh01 || 0) +
              "</span> m&sup2;/s&sup2;"
          );

          new maplibregl.Marker({ element: marker })
            .setLngLat(coords)
            .setPopup(popup)
            .addTo(map);
        });
      })
      .catch(function () {
        renderFallback(mapEl);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
