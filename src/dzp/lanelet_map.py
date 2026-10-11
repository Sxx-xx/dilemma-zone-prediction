"""Minimal Lanelet2 (.osm) reader: nodes projected to the trajectory x/y frame, stop lines.

SinD maps store node coordinates as lat/lon around an origin of (0, 0). The official
SinD tooling (SIND-Vis-tool/utils/map_vis_without_lanelet.py, LL2XYProjector) projects
them with UTM (zone derived from the origin longitude) and subtracts the origin. We do the
same so map coordinates line up with the trajectory x/y columns.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pyproj


class LL2XYProjector:
    def __init__(self, lat_origin: float = 0.0, lon_origin: float = 0.0):
        zone = math.floor((lon_origin + 180.0) / 6) + 1
        self.p = pyproj.Proj(proj="utm", ellps="WGS84", zone=zone, datum="WGS84")
        self.x0, self.y0 = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        x, y = self.p(lon, lat)
        return np.asarray(x) - self.x0, np.asarray(y) - self.y0


@dataclass
class StopLine:
    way_id: str
    xy: np.ndarray  # (n, 2) polyline in trajectory frame [m]

    @property
    def p0(self) -> np.ndarray:
        return self.xy[0]

    @property
    def p1(self) -> np.ndarray:
        return self.xy[-1]

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class LaneletMap:
    nodes: dict[str, np.ndarray]
    ways: dict[str, dict]  # way_id -> {"nodes": [...], "tags": {...}}
    relations: dict[str, dict]

    def ways_of_type(self, type_: str) -> list[str]:
        return [w for w, d in self.ways.items() if d["tags"].get("type") == type_]

    def way_xy(self, way_id: str) -> np.ndarray:
        return np.array([self.nodes[n] for n in self.ways[way_id]["nodes"]])

    def stop_lines(self) -> list[StopLine]:
        return [StopLine(w, self.way_xy(w)) for w in self.ways_of_type("stop_line")]

    def all_ways_xy(self) -> list[tuple[str, np.ndarray]]:
        return [(d["tags"].get("type", ""), self.way_xy(w)) for w, d in self.ways.items()]


def load_lanelet_map(path: str, lat_origin: float = 0.0, lon_origin: float = 0.0) -> LaneletMap:
    root = ET.parse(path).getroot()
    proj = LL2XYProjector(lat_origin, lon_origin)
    ids, lats, lons = [], [], []
    for n in root.findall("node"):
        ids.append(n.get("id"))
        lats.append(float(n.get("lat")))
        lons.append(float(n.get("lon")))
    xs, ys = proj.latlon2xy(np.array(lats), np.array(lons))
    nodes = {i: np.array([x, y]) for i, x, y in zip(ids, xs, ys)}
    ways = {}
    for w in root.findall("way"):
        ways[w.get("id")] = {
            "nodes": [nd.get("ref") for nd in w.findall("nd")],
            "tags": {t.get("k"): t.get("v") for t in w.findall("tag")},
        }
    relations = {}
    for r in root.findall("relation"):
        relations[r.get("id")] = {
            "members": [(m.get("type"), m.get("ref"), m.get("role")) for m in r.findall("member")],
            "tags": {t.get("k"): t.get("v") for t in r.findall("tag")},
        }
    return LaneletMap(nodes, ways, relations)
