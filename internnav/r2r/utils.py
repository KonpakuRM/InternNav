"""Geometry utilities for the R2R graph executor.

Conventions:
- Pixel coordinates are (row, col), matching ``S2Output.output_pixel``.
- Camera frame is a standard pinhole camera: +x right, +y down, +z forward.
- World frame follows Matterport3D: +z up, heading measured clockwise from +y.
"""

import math
from typing import Optional, Tuple

import numpy as np


def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def pixel_to_bearing_elevation(row: float, col: float, intrinsic: np.ndarray) -> Tuple[float, float]:
    """Convert a pixel to (bearing, elevation) relative to the camera axis.

    Bearing is positive to the right, elevation is positive upward.
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    x = (col - cx) / fx
    y = (row - cy) / fy
    bearing = math.atan2(x, 1.0)
    elevation = -math.atan2(y, math.sqrt(x * x + 1.0))
    return bearing, elevation


def pixel_to_camera_point(row: float, col: float, depth: float, intrinsic: np.ndarray) -> np.ndarray:
    """Unproject a pixel with z-depth into a 3D point in the camera frame."""
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    x = (col - cx) / fx * depth
    y = (row - cy) / fy * depth
    return np.array([x, y, depth], dtype=np.float64)


def camera_to_world(point_cam: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """Transform a camera-frame point into the world frame with a 4x4 pose."""
    p = np.ones(4, dtype=np.float64)
    p[:3] = point_cam
    return (cam_to_world @ p)[:3]


def camera_pose_from_state(
    position: np.ndarray, heading: float, elevation: float, camera_height: float = 0.0
) -> np.ndarray:
    """Build a camera-to-world 4x4 pose from an MP3D agent state.

    The camera sits at ``position + [0, 0, camera_height]`` looking along the
    direction given by heading (clockwise from +y) and elevation (up positive).
    Camera frame: +x right, +y down, +z forward.
    """
    sh, ch = math.sin(heading), math.cos(heading)
    se, ce = math.sin(elevation), math.cos(elevation)
    # Forward direction in world coordinates.
    forward = np.array([sh * ce, ch * ce, se])
    # Right direction (horizontal, perpendicular to forward).
    right = np.array([ch, -sh, 0.0])
    # Camera +y is down.
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = np.asarray(position, dtype=np.float64) + np.array([0.0, 0.0, camera_height])
    return pose


def relative_heading_elevation(
    from_position: np.ndarray,
    to_position: np.ndarray,
    heading: float,
    elevation: float = 0.0,
) -> Tuple[float, float, float]:
    """Relative (heading, elevation, distance) of a target from an agent state."""
    delta = np.asarray(to_position, dtype=np.float64) - np.asarray(from_position, dtype=np.float64)
    horizontal = math.sqrt(delta[0] ** 2 + delta[1] ** 2)
    target_heading = math.atan2(delta[0], delta[1])
    rel_heading = normalize_angle(target_heading - heading)
    rel_elevation = normalize_angle(math.atan2(delta[2], horizontal) - elevation)
    distance = float(np.linalg.norm(delta))
    return rel_heading, rel_elevation, distance


def robust_pixel_depth(
    depth_map: np.ndarray,
    row: int,
    col: int,
    kernel: int = 5,
    max_depth: float = 5.0,
    var_threshold: float = 0.25,
) -> Optional[float]:
    """Median depth in a ``kernel x kernel`` window with reliability checks.

    Returns None when the depth is unreliable: invalid/zero values, farther
    than ``max_depth`` or high local variance (edges, door frames, mirrors).
    """
    if depth_map is None:
        return None
    depth_map = np.asarray(depth_map)
    if depth_map.ndim == 3:
        depth_map = depth_map[..., 0]
    h, w = depth_map.shape
    row = int(np.clip(row, 0, h - 1))
    col = int(np.clip(col, 0, w - 1))
    half = kernel // 2
    window = depth_map[max(0, row - half) : row + half + 1, max(0, col - half) : col + half + 1]
    valid = window[np.isfinite(window) & (window > 1e-3)]
    if valid.size == 0:
        return None
    depth = float(np.median(valid))
    if depth > max_depth:
        return None
    if float(np.var(valid)) > var_threshold:
        return None
    return depth
