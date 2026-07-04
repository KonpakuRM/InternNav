"""Habitat-based visual provider for the R2R navigation graph.

Renders egocentric RGB-D at arbitrary (position, heading, elevation) states on
the MP3D mesh with the same camera specification as the VLN-CE pipeline, so
System-2 observations stay close to its training distribution
(distribution-aligned variant, a.k.a. R2R-Habitat-Graph).
"""

import math
import os
from typing import List

import numpy as np

from internnav.r2r.connectivity import (
    ConnectivityCache,
    GraphState,
    interpolate_positions,
)
from internnav.r2r.providers.base import Observation, VisualProvider
from internnav.r2r.utils import camera_pose_from_state

try:
    import habitat_sim

    HABITAT_SIM_AVAILABLE = True
except ImportError:
    HABITAT_SIM_AVAILABLE = False


def mp3d_to_habitat(position: np.ndarray) -> np.ndarray:
    """Convert an MP3D (z-up) point to Habitat (y-up) coordinates."""
    x, y, z = position
    return np.array([x, z, -y], dtype=np.float32)


class HabitatGraphProvider(VisualProvider):
    supports_depth = True
    supports_interpolation = True

    def __init__(
        self,
        scene_data_dir: str,
        connectivity_dir: str,
        width: int = 640,
        height: int = 480,
        hfov: float = 79.0,
        camera_height: float = 0.0,
        gpu_device_id: int = 0,
    ):
        if not HABITAT_SIM_AVAILABLE:
            raise ImportError('habitat_sim is required for HabitatGraphProvider')
        self.scene_data_dir = scene_data_dir
        self.connectivity = ConnectivityCache(connectivity_dir)
        self.width = width
        self.height = height
        self.hfov = hfov
        # MP3D connectivity poses are camera positions, so the default camera
        # height offset is zero.
        self.camera_height = camera_height
        self.gpu_device_id = gpu_device_id
        self._sim = None
        self._current_scan = None

    def _scene_path(self, scan: str) -> str:
        return os.path.join(self.scene_data_dir, scan, f'{scan}.glb')

    def _make_sim(self, scan: str):
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self._scene_path(scan)
        backend_cfg.gpu_device_id = self.gpu_device_id
        backend_cfg.enable_physics = False

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = 'rgb'
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.height, self.width]
        rgb_spec.hfov = self.hfov
        rgb_spec.position = [0.0, 0.0, 0.0]

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = 'depth'
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [self.height, self.width]
        depth_spec.hfov = self.hfov
        depth_spec.position = [0.0, 0.0, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    def _ensure_scene(self, scan: str):
        if self._current_scan == scan and self._sim is not None:
            return
        if self._sim is not None:
            self._sim.close()
        self._sim = self._make_sim(scan)
        self._current_scan = scan

    @property
    def intrinsic(self) -> np.ndarray:
        fx = (self.width / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        cx = (self.width - 1.0) / 2.0
        cy = (self.height - 1.0) / 2.0
        return np.array([[fx, 0.0, cx, 0.0], [0.0, fx, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def _render(self, scan: str, position_mp3d: np.ndarray, heading: float, elevation: float) -> Observation:
        self._ensure_scene(scan)
        agent = self._sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = mp3d_to_habitat(position_mp3d + np.array([0.0, 0.0, self.camera_height]))
        # MP3D heading 0 faces +y (Habitat -z, i.e. identity rotation); positive
        # heading is clockwise, i.e. negative rotation about Habitat +y.
        rotation = habitat_sim.utils.common.quat_from_angle_axis(
            -heading, np.array([0.0, 1.0, 0.0])
        ) * habitat_sim.utils.common.quat_from_angle_axis(elevation, np.array([1.0, 0.0, 0.0]))
        agent_state.rotation = rotation
        for sensor_state in agent_state.sensor_states.values():
            sensor_state.position = agent_state.position
            sensor_state.rotation = agent_state.rotation
        agent.set_state(agent_state, infer_sensor_states=True)

        observations = self._sim.get_sensor_observations()
        rgb = np.asarray(observations['rgb'])[..., :3].astype(np.uint8)
        depth = np.asarray(observations['depth']).astype(np.float32)
        cam_to_world = camera_pose_from_state(position_mp3d, heading, elevation, self.camera_height)
        return Observation(rgb=rgb, depth=depth, intrinsic=self.intrinsic, cam_to_world=cam_to_world)

    def observe(self, state: GraphState) -> Observation:
        position = self.connectivity.get(state.scan).position(state.viewpoint)
        return self._render(state.scan, position, state.heading, state.elevation)

    def interpolate_observations(
        self, from_state: GraphState, to_state: GraphState, step: float = 0.25
    ) -> List[Observation]:
        graph = self.connectivity.get(from_state.scan)
        from_pos = graph.position(from_state.viewpoint)
        to_pos = graph.position(to_state.viewpoint)
        delta = to_pos - from_pos
        travel_heading = math.atan2(delta[0], delta[1])
        positions = interpolate_positions(from_pos, to_pos, step)
        return [self._render(from_state.scan, p, travel_heading, 0.0) for p in positions[:-1]]

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None
