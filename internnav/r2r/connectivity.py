"""Loader for the Matterport3D / R2R connectivity graphs."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from internnav.r2r.utils import relative_heading_elevation


@dataclass
class GraphState:
    """Discrete agent state on the navigation graph."""

    scan: str
    viewpoint: str
    heading: float = 0.0  # radians, clockwise from +y
    elevation: float = 0.0  # radians, up positive


@dataclass
class Candidate:
    """A navigable neighbor of the current viewpoint."""

    viewpoint: str
    position: np.ndarray
    rel_heading: float
    rel_elevation: float
    distance: float


@dataclass
class ConnectivityGraph:
    scan: str
    graph: nx.Graph
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    _shortest_distances: Optional[dict] = None
    _shortest_paths: Optional[dict] = None

    def position(self, viewpoint: str) -> np.ndarray:
        return self.positions[viewpoint]

    def neighbors(self, viewpoint: str) -> List[str]:
        return list(self.graph.neighbors(viewpoint))

    def candidates(self, state: GraphState) -> List[Candidate]:
        """Navigable neighbors with headings relative to the agent state."""
        result = []
        from_pos = self.positions[state.viewpoint]
        for neighbor in self.graph.neighbors(state.viewpoint):
            to_pos = self.positions[neighbor]
            rel_heading, rel_elevation, distance = relative_heading_elevation(
                from_pos, to_pos, state.heading, state.elevation
            )
            result.append(
                Candidate(
                    viewpoint=neighbor,
                    position=to_pos,
                    rel_heading=rel_heading,
                    rel_elevation=rel_elevation,
                    distance=distance,
                )
            )
        return result

    def distance(self, viewpoint_a: str, viewpoint_b: str) -> float:
        """Geodesic distance along the graph."""
        if self._shortest_distances is None:
            self._shortest_distances = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
        return self._shortest_distances[viewpoint_a][viewpoint_b]

    def shortest_path(self, viewpoint_a: str, viewpoint_b: str) -> List[str]:
        if self._shortest_paths is None:
            self._shortest_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight='weight'))
        return self._shortest_paths[viewpoint_a][viewpoint_b]


def load_connectivity(connectivity_dir: str, scan: str) -> ConnectivityGraph:
    """Load one scan's connectivity graph from ``{scan}_connectivity.json``.

    An edge exists between two viewpoints when both are included and the
    connection is unobstructed in either direction (following MatterSim).
    """
    path = os.path.join(connectivity_dir, f'{scan}_connectivity.json')
    with open(path) as f:
        data = json.load(f)

    graph = nx.Graph()
    positions: Dict[str, np.ndarray] = {}
    included = []
    for item in data:
        if not item['included']:
            included.append(False)
            continue
        included.append(True)
        pose = item['pose']
        # Row-major 4x4 pose; translation at indices 3, 7, 11 (camera position).
        position = np.array([pose[3], pose[7], pose[11]], dtype=np.float64)
        positions[item['image_id']] = position
        graph.add_node(item['image_id'])

    for i, item in enumerate(data):
        if not included[i]:
            continue
        for j, unobstructed in enumerate(item['unobstructed']):
            if not unobstructed or j >= len(data) or not included[j]:
                continue
            a, b = item['image_id'], data[j]['image_id']
            if graph.has_edge(a, b):
                continue
            weight = float(np.linalg.norm(positions[a] - positions[b]))
            graph.add_edge(a, b, weight=weight)

    return ConnectivityGraph(scan=scan, graph=graph, positions=positions)


class ConnectivityCache:
    """Lazily loads and caches connectivity graphs per scan."""

    def __init__(self, connectivity_dir: str):
        self.connectivity_dir = connectivity_dir
        self._cache: Dict[str, ConnectivityGraph] = {}

    def get(self, scan: str) -> ConnectivityGraph:
        if scan not in self._cache:
            self._cache[scan] = load_connectivity(self.connectivity_dir, scan)
        return self._cache[scan]


def interpolate_positions(from_pos: np.ndarray, to_pos: np.ndarray, step: float = 0.25) -> List[np.ndarray]:
    """Intermediate positions between two viewpoints at ``step`` spacing.

    Excludes the start position and includes the end position.
    """
    from_pos = np.asarray(from_pos, dtype=np.float64)
    to_pos = np.asarray(to_pos, dtype=np.float64)
    distance = float(np.linalg.norm(to_pos - from_pos))
    if distance < 1e-6:
        return [to_pos]
    num = max(1, int(math.ceil(distance / step)))
    return [from_pos + (to_pos - from_pos) * (i / num) for i in range(1, num + 1)]
