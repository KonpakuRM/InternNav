"""Native R2R visual providers: MatterSim rendering and a prerendered cache.

These providers produce the official R2R imagery used for the main-table
results. Depth is not part of the official R2R observation, so both default to
RGB-only (the executor then uses the pure-angular scoring branch, lambda=0).
"""

import math
import os

import numpy as np

from internnav.r2r.connectivity import GraphState
from internnav.r2r.providers.base import Observation, VisualProvider
from internnav.r2r.utils import camera_pose_from_state

try:
    import MatterSim

    MATTERSIM_AVAILABLE = True
except ImportError:
    MATTERSIM_AVAILABLE = False


def build_intrinsic(width: int, height: int, hfov: float) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array([[fx, 0.0, cx, 0.0], [0.0, fx, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


class MatterSimProvider(VisualProvider):
    """Renders egocentric RGB with the official Matterport3DSimulator."""

    supports_depth = False
    supports_interpolation = False

    def __init__(
        self,
        scan_data_dir: str,
        connectivity_dir: str,
        width: int = 640,
        height: int = 480,
        vfov: float = 60.0,
    ):
        if not MATTERSIM_AVAILABLE:
            raise ImportError('MatterSim is required for MatterSimProvider')
        self.width = width
        self.height = height
        self.vfov = math.radians(vfov)
        hfov = 2 * math.degrees(math.atan(math.tan(self.vfov / 2) * width / height))
        self.intrinsic = build_intrinsic(width, height, hfov)

        self.sim = MatterSim.Simulator()
        self.sim.setDatasetPath(scan_data_dir)
        self.sim.setNavGraphPath(connectivity_dir)
        self.sim.setCameraResolution(width, height)
        self.sim.setCameraVFOV(self.vfov)
        self.sim.setDiscretizedViewingAngles(False)
        self.sim.setRenderingEnabled(True)
        self.sim.initialize()

    def observe(self, state: GraphState) -> Observation:
        self.sim.newEpisode([state.scan], [state.viewpoint], [state.heading], [state.elevation])
        sim_state = self.sim.getState()[0]
        rgb = np.array(sim_state.rgb, copy=True)[..., ::-1]  # BGR -> RGB
        location = sim_state.location
        position = np.array([location.x, location.y, location.z])
        cam_to_world = camera_pose_from_state(position, state.heading, state.elevation)
        return Observation(rgb=rgb, depth=None, intrinsic=self.intrinsic, cam_to_world=cam_to_world)

    def close(self):
        self.sim.close()


class PrerenderedProvider(VisualProvider):
    """Serves cached RGB frames when MatterSim is unavailable at runtime.

    Cache layout: ``{cache_dir}/{scan}/{viewpoint}/{heading_idx:02d}_{elev_idx}.png``
    with ``num_headings`` discretized headings (clockwise from +y) and three
    elevation levels (0: down 30 deg, 1: level, 2: up 30 deg). Requested states are
    snapped to the nearest cached view. Use ``scripts/eval/prerender_r2r.py``
    to generate the cache.
    """

    supports_depth = False
    supports_interpolation = False

    def __init__(
        self,
        cache_dir: str,
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        num_headings: int = 12,
    ):
        self.cache_dir = cache_dir
        self.num_headings = num_headings
        self.intrinsic = build_intrinsic(width, height, hfov)

    def _snap(self, heading: float, elevation: float):
        heading_step = 2 * math.pi / self.num_headings
        heading_idx = int(round((heading % (2 * math.pi)) / heading_step)) % self.num_headings
        elev_idx = int(np.clip(round(elevation / math.radians(30.0)) + 1, 0, 2))
        return heading_idx, elev_idx

    def snapped_state(self, state: GraphState) -> GraphState:
        heading_idx, elev_idx = self._snap(state.heading, state.elevation)
        return GraphState(
            scan=state.scan,
            viewpoint=state.viewpoint,
            heading=heading_idx * 2 * math.pi / self.num_headings,
            elevation=(elev_idx - 1) * math.radians(30.0),
        )

    def _frame_path(self, state: GraphState) -> str:
        heading_idx, elev_idx = self._snap(state.heading, state.elevation)
        return os.path.join(self.cache_dir, state.scan, state.viewpoint, f'{heading_idx:02d}_{elev_idx}.png')

    def observe(self, state: GraphState) -> Observation:
        from PIL import Image

        snapped = self.snapped_state(state)
        rgb = np.asarray(Image.open(self._frame_path(state)).convert('RGB'))
        cam_to_world = camera_pose_from_state(np.zeros(3), snapped.heading, snapped.elevation)
        return Observation(rgb=rgb, depth=None, intrinsic=self.intrinsic, cam_to_world=cam_to_world)
