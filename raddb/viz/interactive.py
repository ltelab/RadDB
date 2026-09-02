"""Interactive Area-of-Interest selection on an ipyleaflet map (Jupyter).

Draw a shape on the map and it is dispatched to the matching RadDB method:

    rectangle  ->  crop_by_bbox
    polygon    ->  crop_by_polygon
    marker     ->  crop_around_point       (distance from the widget)
    polyline   ->  extract_cross_section   (first & last vertex)

The map is WGS-84 (lon/lat), so the drawn coordinates are passed with ``crs=4326``
(reprojected to LV95 internally).  Each call returns a new data-carrying
``RadDB``.  The drawn AOI can be saved as GeoJSON and reloaded later via
``crop_by_polygon("aoi.geojson")``.

This is an **optional** UI layer — it needs ``ipyleaflet`` + ``ipywidgets`` and a
Jupyter frontend.  The hand-typed / shapefile crop methods are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ============================================================================
# GeoJSON feature -> crop dispatch  (pure, unit-testable — no widgets)
# ============================================================================


def _is_axis_aligned_box(ring: list) -> bool:
    """True if a polygon ring is an axis-aligned rectangle (a drawn 'rectangle')."""
    pts = ring[:-1] if ring and ring[0] == ring[-1] else ring
    if len(pts) != 4:
        return False
    lons = {round(p[0], 9) for p in pts}
    lats = {round(p[1], 9) for p in pts}
    return len(lons) == 2 and len(lats) == 2


def _crop_from_feature(db, feature, distance_m: float = 15_000.0):
    """Dispatch a drawn GeoJSON feature (lon/lat) to the matching ``db`` method.

    Parameters
    ----------
    db : RadDB
        A data-carrying RadDB (from :meth:`RadDB.open`).
    feature : dict
        A GeoJSON Feature or bare geometry (from ipyleaflet's draw control).
    distance_m : float
        Distance (meters) used when the drawn shape is a marker (Point).

    Returns
    -------
    (kind, rdf) : (str, RadDB)
        ``kind`` is one of ``"bbox" | "polygon" | "point" | "cross_section"``.
    """
    import shapely.geometry

    geom = feature.get("geometry", feature)
    gtype = geom["type"]

    if gtype == "Point":
        lon, lat = geom["coordinates"][:2]
        return "point", db.crop_around_point((lon, lat), distance_m, crs=4326)

    if gtype == "LineString":
        coords = geom["coordinates"]
        p1, p2 = tuple(coords[0][:2]), tuple(coords[-1][:2])
        return "cross_section", db.extract_cross_section(p1, p2, crs=4326)

    if gtype == "Polygon":
        ring = geom["coordinates"][0]
        if _is_axis_aligned_box(ring):
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            bounds = (min(lons), min(lats), max(lons), max(lats))
            return "bbox", db.crop_by_bbox(bounds=bounds, crs=4326)
        return "polygon", db.crop_by_polygon(shapely.geometry.shape(geom), crs=4326)

    raise ValueError(f"unsupported drawn geometry type: {gtype!r}")


def _feature_collection(feature) -> dict:
    """Wrap a drawn feature as a GeoJSON FeatureCollection (WGS-84) for saving."""
    if feature.get("type") != "Feature":  # bare geometry
        feature = {"type": "Feature", "properties": {}, "geometry": feature}
    return {"type": "FeatureCollection", "features": [feature]}


# ============================================================================
# The ipyleaflet widget
# ============================================================================


class AOISelector:
    """Interactive map to draw an AOI and crop a DataFrame with it.

    Displayed automatically in Jupyter (returns from :meth:`RadDB.interactive_crop`).
    After drawing a shape and clicking **Apply crop**, the result is available on:

    - :attr:`result` — the cropped ``RadDB``,
    - :attr:`kind` — which crop ran (``bbox``/``polygon``/``point``/``cross_section``),
    - :attr:`feature` — the drawn GeoJSON feature (lon/lat).

    Parameters
    ----------
    db : RadDB
        A data-carrying RadDB (from :meth:`RadDB.open`).
    radars : list of str, optional
        Radars to mark on the map.  Default: the radars in ``db``.
    center : (lat, lon), optional
        Initial map center.  Default: mean of the radar sites, else Switzerland.
    zoom : int
    point_radius_m : float
        Default distance (meters) for the marker (point) crop.
    """

    def __init__(self, db, radars=None, center=None, zoom=8, point_radius_m=15_000.0):
        try:
            import ipywidgets as W
            from ipyleaflet import AwesomeIcon, DrawControl, Map, Marker
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "interactive_crop needs ipyleaflet + ipywidgets. " "Install with: pip install ipyleaflet ipywidgets",
            ) from exc

        self.db = db
        self.feature = None
        self.result = None
        self.kind = None

        radars = radars if radars is not None else db.radars()
        sites = self._radar_sites(radars)
        if center is None:
            center = (
                (sum(s[0] for s in sites.values()) / len(sites), sum(s[1] for s in sites.values()) / len(sites))
                if sites
                else (46.82, 8.23)
            )

        self.map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
        for r, (lat, lon) in sites.items():
            self.map.add(
                Marker(
                    location=(lat, lon),
                    title=f"radar {r}",
                    draggable=False,
                    icon=AwesomeIcon(name="broadcast-tower", marker_color="red"),
                ),
            )

        self.draw = DrawControl(
            rectangle={"shapeOptions": {"color": "#e31a1c", "weight": 2, "fillOpacity": 0.05}},
            polygon={"shapeOptions": {"color": "#e31a1c", "weight": 2, "fillOpacity": 0.05}},
            polyline={"shapeOptions": {"color": "#e31a1c", "weight": 3}},
            marker={"shapeOptions": {}},
            circle={},
            circlemarker={},
        )
        self.draw.on_draw(self._on_draw)
        self.map.add(self.draw)

        # --- controls ---
        self.radius = W.FloatText(
            value=float(point_radius_m),
            description="if marker → radius [m]",
            style={"description_width": "initial"},
            layout=W.Layout(width="320px"),
        )
        self.apply_btn = W.Button(description="Apply crop", button_style="primary", icon="scissors")
        self.save_path = W.Text(value="aoi.geojson", description="save as", layout=W.Layout(width="240px"))
        self.save_btn = W.Button(description="Save GeoJSON", icon="save")
        self.out = W.Output()
        self.apply_btn.on_click(self._apply)
        self.save_btn.on_click(self._save)

        instructions = W.HTML(
            "<b>Draw an AOI</b> with the toolbar (top-left): "
            "▭ rectangle → <code>crop_by_bbox</code>, ⬠ polygon → <code>crop_by_polygon</code>, "
            "📍 marker → <code>crop_around_point</code> (uses the radius box below), "
            "/ line → <code>extract_cross_section</code>. Then <b>Apply crop</b>.",
        )
        self._widget = W.VBox(
            [
                instructions,
                self.map,
                W.HBox([self.radius, self.apply_btn, self.save_path, self.save_btn]),
                self.out,
            ],
        )

    # -- radar site coords (lat, lon) --
    def _radar_sites(self, radars):
        sites = {}
        for r in radars:
            try:
                info = self.db.get_radar_info(r)
                sites[r] = (float(info["latitude"]), float(info["longitude"]))
            except Exception:  # - a missing radar shouldn't break the map
                continue
        return sites

    # -- draw callback: keep the last drawn feature --
    def _on_draw(self, _target, action, geo_json):
        if action in ("created", "edited"):
            self.feature = geo_json

    def _apply(self, _btn=None):
        with self.out:
            self.out.clear_output()
            if self.feature is None:
                print("Draw a shape on the map first (rectangle / polygon / marker / line).")
                return
            try:
                self.kind, self.result = _crop_from_feature(
                    self.db,
                    self.feature,
                    distance_m=self.radius.value,
                )
            except Exception as exc:  # - surface errors in the widget
                print(f"crop failed: {exc}")
                return
            r = self.result
            try:
                radars = r.radars() if len(r) else []
            except Exception:
                radars = []
            # `sweep` is a LUT column, never stored per gate, so reading it off
            # r.columns() always missed and printed "?".  It is decoded from the
            # gate_id instead, which every row carries.
            sweeps = 0
            if len(r):
                from raddb.lut import decode_gate_ids

                sweeps = int(np.unique(decode_gate_ids(r.data["gate_id"].to_numpy())[0]).size)
            print(f"{self.kind} -> {len(r):,} gates | radars {radars} | {sweeps} sweeps")
            print("result is available as  <selector>.result  (a cropped RadDB).")

    def _save(self, _btn=None):
        with self.out:
            if self.feature is None:
                print("Nothing to save — draw a shape first.")
                return
            path = Path(self.save_path.value)
            path.write_text(json.dumps(_feature_collection(self.feature), indent=1))
            print(f"saved AOI -> {path}  (reload with db.crop_by_polygon('{path}') for polygons)")

    def display(self):
        """Render the widget in the notebook and return ``self``."""
        from IPython.display import display

        display(self._widget)
        return self

    def _ipython_display_(self):
        from IPython.display import display

        display(self._widget)
