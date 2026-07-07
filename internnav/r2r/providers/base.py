"""Abstract visual provider interface for the R2R graph executor.

Two implementations are planned:
- ``HabitatGraphProvider``: renders the MP3D mesh with Habitat (RGB-D,
  supports intermediate-frame interpolation) — distribution-aligned variant.
- ``MatterSimProvider`` / prerendered cache: native R2R imagery for official
  results (RGB only by default, no interpolation).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from internnav.r2r.connectivity import GraphState


@dataclass
class Observation:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray]  # (H, W) or (H, W, 1) meters, None if unavailable
    intrinsic: np.ndarray  # 4x4 pinhole intrinsic matrix
    cam_to_world: np.ndarray  # 4x4 camera-to-world pose (MP3D frame)


class VisualProvider(ABC):
    supports_depth: bool = False
    supports_interpolation: bool = False

    @abstractmethod
    def observe(self, state: GraphState) -> Observation:
        """Render the egocentric observation for a graph state."""

    def interpolate_observations(
        self, from_state: GraphState, to_state: GraphState, step: float = 0.25
    ) -> List[Observation]:
        """Observations along the straight line between two viewpoints.

        Used to align the S2 history granularity with its 0.25m training
        distribution. Only meaningful for mesh-based providers.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support interpolation')

    def close(self):
        pass
