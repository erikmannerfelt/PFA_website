
function add_overview_map(map_id, fit_bounds = true, radar_key = null) {

  let overview_map = L.map(map_id, {
    maxZoom: 17,
    minZoom: 3,
  });

  // If it's called from index.html, the location should be parsed from the map_id. If digitize.html, then from the radar_key
  let location_key = "";
  if (radar_key != null) {
    let parts = radar_key.split("-");
    location_key = `${parts[0]}_${parts[2]}`;
  } else {
    location_key = map_id.replace("map-", "").replace(" ", "_");
  };

  fetch(`/location_info/${location_key}.json`)
    .then((response) => response.json())
    .then(function (location_info) {
      for (let radar_key2 of location_info["radar_keys"]) {
        if (radar_key == radar_key2) {
          continue;
        };
        fetch(`/radargram_meta/${radar_key2}.json`).then((response) => response.json()).then(function (other_meta) {

          other_meta["track"].forEach(function (track_json, _) {
            L.geoJSON(track_json, {
              color: "#ccc",
              opacity: 0.5,
            })
              .bindPopup(function (_) {
                return `<a href=/digitize/${other_meta.radar_key} target="_blank">${other_meta.radar_key} </a>`
              })
              .addTo(overview_map);
          });
        });

      if (fit_bounds) {
        let overview_bounds = [
          [location_info["bounds"]["minlat"], location_info["bounds"]["minlon"]],
          [location_info["bounds"]["maxlat"], location_info["bounds"]["maxlon"]],
        ];
        overview_map.fitBounds(overview_bounds);
      }
      };
  });

  let esri = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      bounds: [
        [-90, -180],
        [90, 180],
      ],
      noWrap: true,
    }
  ).addTo(overview_map);

  let auf_layers = {"Austfonna 2024": "austfonna_20240903", "Austfonna 2024 (dark)": "austfonna_20240903_snow"};
  let overlays = {};

  for (key in auf_layers) {
    let lyr = L.tileLayer(
      `https://static.livingiceproject.com/maptiles/${auf_layers[key]}` + "/{z}/{x}/{y}.webp",
      {
        bounds: [
          [79.2, 20.7],
          [80.27, 27.5],
        ],
        minZoom: 4,
        maxZoom: 14,
      }
    )
    overlays[key] = lyr;
    if (key == "Austfonna 2024") {
      lyr.addTo(overview_map);
    };
  };
  L.control.layers({"ESRI": esri}, overlays, {collapsed: true}).addTo(overview_map);

  return overview_map;

}
