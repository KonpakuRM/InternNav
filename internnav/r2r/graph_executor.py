"""Graph executor mapping System-2 outputs to native R2R viewpoint actions.

Decision logic (per step):
- STOP: end the episode.
- View adjustment (turn/look): update the virtual camera pose only, bounded by
  ``max_view_adjustments`` per node; no trajectory movement.
- FORWARD (``forward_heading_fallback``): only executed when a candidate lies
  within ``forward_fallback_angle`` of the current heading, otherwise the agent
  turns and re-queries System 2.
- Pixel goal (row, col): candidates are scored with direction consistency as
  the primary term plus an optional normalized distance-consistency term
  (weight ``lambda_dist``) when reliable depth is available:

      score(n) = |d_bearing| + w_elev * |d_elevation| + lambda * D(n) + penalties

  When the unprojected goal is farther than every candidate, the goal is kept
  as a pending multi-hop subgoal consumed for up to ``max_hops_per_pixel_goal``
  hops before System 2 is re-queried.

Every step emits a structured JSONL record for debugging and failure
attribution (perception / matching / stop).
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from internnav.r2r.connectivity import Candidate, ConnectivityCache, GraphState
from internnav.r2r.planner_adapter import (
    ACTION_LOOK_DOWN,
    ACTION_LOOK_UP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    R2RPlannerAdapter,
    S2Plan,
)
from internnav.r2r.providers.base import Observation, VisualProvider
from internnav.r2r.utils import (
    camera_to_world,
    normalize_angle,
    pixel_to_bearing_elevation,
    pixel_to_camera_point,
    robust_pixel_depth,
)

TURN_ANGLE = math.radians(15.0)


@dataclass
class ExecutorConfig:
    max_graph_steps: int = 20
    max_view_adjustments: int = 4
    # Scoring. lambda_dist=0.0 is the official R2R (RGB-only) branch.
    lambda_dist: float = 0.0
    w_elevation: float = 0.3
    revisit_penalty: float = 1.0
    # Depth reliability.
    depth_kernel: int = 5
    depth_max: float = 5.0
    depth_var_threshold: float = 0.25
    # Fallbacks.
    forward_fallback_angle: float = math.radians(20.0)
    reask_angle: float = math.radians(45.0)
    max_hops_per_pixel_goal: int = 3
    multi_hop_margin: float = 0.5
    # Stop guard: force stop after two consecutive near-agent pixel goals.
    stop_guard_distance: float = 0.5
    # History alignment (HabitatGraphProvider only; must stay False for
    # official R2R runs).
    interpolate_history: bool = False
    interpolate_step: float = 0.25
    log_dir: Optional[str] = None


@dataclass
class EpisodeResult:
    trajectory: List[List] = field(default_factory=list)  # [viewpoint, heading, elevation]
    stop_called: bool = False
    steps: int = 0


class R2RGraphExecutor:
    def __init__(
        self,
        adapter: R2RPlannerAdapter,
        provider: VisualProvider,
        connectivity: ConnectivityCache,
        config: Optional[ExecutorConfig] = None,
    ):
        self.adapter = adapter
        self.provider = provider
        self.connectivity = connectivity
        self.config = config or ExecutorConfig()
        if self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)

    # ------------------------------------------------------------------ scoring
    def score_candidates(
        self,
        candidates: List[Candidate],
        pixel_bearing: float,
        pixel_elevation: float,
        goal_distance: Optional[float],
        visit_counts: dict,
    ) -> List[dict]:
        cfg = self.config
        scored = []
        for cand in candidates:
            bearing_err = abs(normalize_angle(cand.rel_heading - pixel_bearing))
            elev_err = abs(normalize_angle(cand.rel_elevation - pixel_elevation))
            angular = bearing_err + cfg.w_elevation * elev_err
            dist_term = 0.0
            if goal_distance is not None and cfg.lambda_dist > 0 and goal_distance > 1e-3:
                dist_term = abs(cand.distance - goal_distance) / goal_distance
            revisit = cfg.revisit_penalty * visit_counts.get(cand.viewpoint, 0)
            score = angular + cfg.lambda_dist * dist_term + revisit
            scored.append(
                {
                    'viewpoint': cand.viewpoint,
                    'bearing_err': bearing_err,
                    'elev_err': elev_err,
                    'dist_term': dist_term,
                    'revisit': revisit,
                    'score': score,
                    'candidate': cand,
                }
            )
        return sorted(scored, key=lambda item: item['score'])

    # ------------------------------------------------------------------ helpers
    def _unproject_goal(self, plan: S2Plan, obs: Observation) -> Optional[np.ndarray]:
        cfg = self.config
        if obs.depth is None or cfg.lambda_dist <= 0:
            return None
        row, col = int(plan.pixel[0]), int(plan.pixel[1])
        depth = robust_pixel_depth(
            obs.depth, row, col, kernel=cfg.depth_kernel, max_depth=cfg.depth_max, var_threshold=cfg.depth_var_threshold
        )
        if depth is None:
            return None
        point_cam = pixel_to_camera_point(row, col, depth, obs.intrinsic)
        return camera_to_world(point_cam, obs.cam_to_world)

    def _goto(self, state: GraphState, candidate: Candidate, result: EpisodeResult, visit_counts: dict) -> GraphState:
        cfg = self.config
        graph = self.connectivity.get(state.scan)
        from_pos = graph.position(state.viewpoint)
        delta = candidate.position - from_pos
        new_heading = math.atan2(delta[0], delta[1])
        new_state = GraphState(scan=state.scan, viewpoint=candidate.viewpoint, heading=new_heading, elevation=0.0)
        if cfg.interpolate_history and self.provider.supports_interpolation:
            for frame in self.provider.interpolate_observations(state, new_state, cfg.interpolate_step):
                self.adapter.observe_no_infer(frame.rgb, frame.depth, frame.cam_to_world)
        visit_counts[candidate.viewpoint] = visit_counts.get(candidate.viewpoint, 0) + 1
        result.trajectory.append([new_state.viewpoint, new_state.heading, new_state.elevation])
        return new_state

    def _log(self, log_file, record: dict):
        if log_file is not None:
            log_file.write(json.dumps(record, default=_json_default) + '\n')
            log_file.flush()

    # ------------------------------------------------------------------ episode
    def run_episode(self, instruction: str, start_state: GraphState, episode_id: str = 'episode') -> EpisodeResult:
        cfg = self.config
        self.adapter.reset()
        state = GraphState(**vars(start_state))
        graph = self.connectivity.get(state.scan)
        result = EpisodeResult(trajectory=[[state.viewpoint, state.heading, state.elevation]])
        visit_counts = {state.viewpoint: 1}

        log_file = None
        if cfg.log_dir:
            log_file = open(os.path.join(cfg.log_dir, f'{episode_id}.jsonl'), 'w')

        view_adjustments = 0
        near_goal_streak = 0
        pending_goal: Optional[np.ndarray] = None
        pending_hops = 0

        try:
            while result.steps < cfg.max_graph_steps:
                obs = self.provider.observe(state)
                candidates = graph.candidates(state)
                if not candidates:
                    break

                record = {
                    'step': result.steps,
                    'node': state.viewpoint,
                    'heading': state.heading,
                    'elevation': state.elevation,
                    'view_adjustments_used': view_adjustments,
                    'reask_triggered': False,
                }

                # Multi-hop consumption of a pending far subgoal (no S2 call).
                if pending_goal is not None and pending_hops > 0:
                    from_pos = graph.position(state.viewpoint)
                    best = min(candidates, key=lambda c: float(np.linalg.norm(c.position - pending_goal)))
                    self.adapter.observe_no_infer(obs.rgb, obs.depth, obs.cam_to_world)
                    pending_hops -= 1
                    reached = float(np.linalg.norm(best.position - pending_goal)) < cfg.multi_hop_margin
                    goal_dist_before = float(np.linalg.norm(pending_goal - from_pos))
                    record.update({'s2_type': 'pending_goal', 'chosen': best.viewpoint, 'pending_hops': pending_hops})
                    self._log(log_file, record)
                    state = self._goto(state, best, result, visit_counts)
                    result.steps += 1
                    view_adjustments = 0
                    goal_dist_after = float(np.linalg.norm(pending_goal - graph.position(state.viewpoint)))
                    if reached or pending_hops <= 0 or goal_dist_after >= goal_dist_before:
                        pending_goal, pending_hops = None, 0
                    continue

                plan = self.adapter.plan(
                    obs.rgb, obs.depth, obs.cam_to_world, instruction, obs.intrinsic, look_down=state.elevation < 0
                )
                record.update({'s2_type': plan.kind, 's2_text': plan.raw_text})

                if plan.kind == 'stop':
                    result.stop_called = True
                    self._log(log_file, record)
                    break

                if plan.kind == 'view_adjust':
                    if view_adjustments >= cfg.max_view_adjustments:
                        # Budget exhausted: fall back to the forward-most candidate.
                        best = min(candidates, key=lambda c: abs(c.rel_heading))
                        record.update({'chosen': best.viewpoint, 'fallback': 'view_budget_exhausted'})
                        self._log(log_file, record)
                        state = self._goto(state, best, result, visit_counts)
                        result.steps += 1
                        view_adjustments = 0
                        continue
                    action = plan.view_actions[0]
                    if action == ACTION_TURN_LEFT:
                        state.heading = normalize_angle(state.heading - TURN_ANGLE)
                    elif action == ACTION_TURN_RIGHT:
                        state.heading = normalize_angle(state.heading + TURN_ANGLE)
                    elif action == ACTION_LOOK_DOWN:
                        state.elevation = normalize_angle(state.elevation - TURN_ANGLE)
                    elif action == ACTION_LOOK_UP:
                        state.elevation = normalize_angle(state.elevation + TURN_ANGLE)
                    view_adjustments += 1
                    record.update({'view_action': action})
                    self._log(log_file, record)
                    continue

                if plan.kind == 'forward':
                    best = min(candidates, key=lambda c: abs(c.rel_heading))
                    if abs(best.rel_heading) <= cfg.forward_fallback_angle:
                        record.update({'chosen': best.viewpoint, 'fallback': 'forward_heading_fallback'})
                        self._log(log_file, record)
                        state = self._goto(state, best, result, visit_counts)
                        result.steps += 1
                        view_adjustments = 0
                    else:
                        # No candidate ahead: turn toward the nearest one and re-query.
                        state.heading = normalize_angle(
                            state.heading + np.clip(best.rel_heading, -TURN_ANGLE, TURN_ANGLE)
                        )
                        view_adjustments += 1
                        record['reask_triggered'] = True
                        self._log(log_file, record)
                    continue

                # ---- pixel goal ----
                row, col = float(plan.pixel[0]), float(plan.pixel[1])
                # Pixel ray relative to the current camera axis; candidates'
                # rel_heading/rel_elevation share the same reference frame.
                pixel_bearing, pixel_elevation = pixel_to_bearing_elevation(row, col, obs.intrinsic)
                record['pixel'] = [row, col]

                goal_world = self._unproject_goal(plan, obs)
                goal_distance = None
                if goal_world is not None:
                    from_pos = graph.position(state.viewpoint)
                    goal_distance = float(np.linalg.norm(goal_world - from_pos))
                    record.update({'depth_valid': True, 'unprojected_goal': goal_world.tolist()})
                else:
                    record['depth_valid'] = False

                # Stop guard: repeated near-agent pixel goals imply arrival.
                if goal_distance is not None and goal_distance < cfg.stop_guard_distance:
                    near_goal_streak += 1
                    if near_goal_streak >= 2:
                        result.stop_called = True
                        record['fallback'] = 'stop_guard'
                        self._log(log_file, record)
                        break
                else:
                    near_goal_streak = 0

                scored = self.score_candidates(candidates, pixel_bearing, pixel_elevation, goal_distance, visit_counts)
                record['candidates'] = [{k: v for k, v in item.items() if k != 'candidate'} for item in scored]
                best = scored[0]

                if best['bearing_err'] > cfg.reask_angle and view_adjustments < cfg.max_view_adjustments:
                    # Turn toward the pixel ray and re-query System 2.
                    state.heading = normalize_angle(state.heading + np.clip(pixel_bearing, -TURN_ANGLE, TURN_ANGLE))
                    view_adjustments += 1
                    record['reask_triggered'] = True
                    self._log(log_file, record)
                    continue

                candidate = best['candidate']
                record['chosen'] = candidate.viewpoint
                # Far subgoal: keep it as a pending multi-hop goal.
                if (
                    goal_distance is not None
                    and goal_distance > max(c.distance for c in candidates) + cfg.multi_hop_margin
                    and cfg.max_hops_per_pixel_goal > 1
                ):
                    pending_goal = goal_world
                    pending_hops = cfg.max_hops_per_pixel_goal - 1
                    record['pending_hops'] = pending_hops
                self._log(log_file, record)
                state = self._goto(state, candidate, result, visit_counts)
                result.steps += 1
                view_adjustments = 0
                near_goal_streak = 0
        finally:
            if log_file is not None:
                log_file.close()

        return result


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
