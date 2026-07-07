"""Thin wrapper around the frozen System-2 planner for graph execution.

The adapter only consumes ``output_action`` and ``output_pixel`` from
``S2Output`` (never ``output_latent``), so it is compatible with both
System2-only and DualVLN checkpoints. Pixel goals keep the repository
convention of (row, col).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Discrete action codes emitted by System 2 (see InternVLAN1Net.actions2idx).
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_LOOK_UP = 4
ACTION_LOOK_DOWN = 5

VIEW_ADJUST_ACTIONS = (ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_LOOK_UP, ACTION_LOOK_DOWN)


@dataclass
class S2Plan:
    """Normalized System-2 decision for the graph executor."""

    kind: str  # 'stop' | 'pixel' | 'view_adjust' | 'forward'
    pixel: Optional[np.ndarray] = None  # (row, col)
    view_actions: List[int] = field(default_factory=list)
    raw_text: str = ''


class R2RPlannerAdapter:
    """Adapts an InternVLA-N1 policy (System 2) to graph-executor plans.

    ``policy`` is duck-typed and must provide ``s2_step``, ``step_no_infer``
    and ``reset`` (``InternVLAN1Net`` satisfies this interface).
    """

    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        self.policy.reset()

    def observe_no_infer(self, rgb: np.ndarray, depth: Optional[np.ndarray], pose: np.ndarray):
        """Feed an intermediate frame into the S2 history without inference."""
        self.policy.step_no_infer(rgb, depth, pose)

    def plan(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose: np.ndarray,
        instruction: str,
        intrinsic: np.ndarray,
        look_down: bool = False,
    ) -> S2Plan:
        output = self.policy.s2_step(rgb, depth, pose, instruction, intrinsic, look_down)
        raw_text = getattr(self.policy, 'llm_output', '')

        if output.output_pixel is not None:
            return S2Plan(kind='pixel', pixel=np.asarray(output.output_pixel), raw_text=raw_text)

        actions = list(output.output_action) if output.output_action is not None else []
        if not actions or actions[0] == ACTION_STOP:
            return S2Plan(kind='stop', raw_text=raw_text)
        if actions[0] == ACTION_FORWARD:
            return S2Plan(kind='forward', raw_text=raw_text)
        view_actions = [a for a in actions if a in VIEW_ADJUST_ACTIONS]
        if view_actions:
            return S2Plan(kind='view_adjust', view_actions=view_actions, raw_text=raw_text)
        return S2Plan(kind='stop', raw_text=raw_text)
