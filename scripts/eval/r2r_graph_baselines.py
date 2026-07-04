"""Reference baselines for the R2R graph executor on real R2R data.

Runs directly on the connectivity graph without any rendering or model:
- oracle: follows the geodesic shortest path to the goal (upper bound; this is
  the "S2 + oracle nearest-next-viewpoint" bound, not the Habitat variant).
- random: uniformly random neighbor for a fixed number of hops.
- forward_only: always the candidate closest to the current heading.

Validates the loader/metrics end-to-end (oracle must reach SR = 1.0) and
provides the upper/lower reference rows for the paper tables.

Usage:
    python scripts/eval/r2r_graph_baselines.py --baseline oracle \
        --annotation_path data/r2r/R2R_val_seen.json \
        --connectivity_dir data/connectivity --output_path logs/r2r_baselines
"""

import argparse
import json
import math
import os
import random
from typing import Dict

import numpy as np

from internnav.r2r.connectivity import ConnectivityCache, GraphState
from internnav.r2r.evaluation import R2RGraphEvaluator, load_r2r_episodes
from internnav.r2r.graph_executor import EpisodeResult


class BaselineRunner:
    """Minimal executor substitute exposing run_episode for the evaluator."""

    def __init__(self, baseline: str, connectivity: ConnectivityCache, max_steps: int = 20, seed: int = 0):
        self.baseline = baseline
        self.connectivity = connectivity
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.goal_by_instr: Dict[str, str] = {}

    def run_episode(self, instruction: str, start_state: GraphState, episode_id: str = 'episode') -> EpisodeResult:
        graph = self.connectivity.get(start_state.scan)
        state = GraphState(**vars(start_state))
        result = EpisodeResult(trajectory=[[state.viewpoint, state.heading, state.elevation]])

        if self.baseline == 'oracle':
            goal = self.goal_by_instr[episode_id]
            path = graph.shortest_path(state.viewpoint, goal)
            for viewpoint in path[1:]:
                delta = graph.position(viewpoint) - graph.position(result.trajectory[-1][0])
                heading = math.atan2(delta[0], delta[1])
                result.trajectory.append([viewpoint, heading, 0.0])
                result.steps += 1
            result.stop_called = True
            return result

        for _ in range(self.max_steps):
            candidates = graph.candidates(state)
            if not candidates:
                break
            if self.baseline == 'random':
                cand = self.rng.choice(candidates)
            elif self.baseline == 'forward_only':
                cand = min(candidates, key=lambda c: abs(c.rel_heading))
            else:
                raise ValueError(f'Unknown baseline: {self.baseline}')
            delta = cand.position - graph.position(state.viewpoint)
            heading = math.atan2(delta[0], delta[1])
            state = GraphState(scan=state.scan, viewpoint=cand.viewpoint, heading=heading)
            result.trajectory.append([state.viewpoint, state.heading, 0.0])
            result.steps += 1
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=['oracle', 'random', 'forward_only'], default='oracle')
    parser.add_argument('--annotation_path', required=True)
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    connectivity = ConnectivityCache(args.connectivity_dir)
    runner = BaselineRunner(args.baseline, connectivity, max_steps=args.max_steps, seed=args.seed)
    output_path = os.path.join(args.output_path, args.baseline)
    evaluator = R2RGraphEvaluator(
        executor=runner,
        connectivity=connectivity,
        annotation_path=args.annotation_path,
        output_path=output_path,
        max_episodes=args.max_episodes,
    )
    for episode in load_r2r_episodes(args.annotation_path):
        runner.goal_by_instr[episode['instr_id']] = episode['path'][-1]

    summary = evaluator.eval()
    print(
        json.dumps(
            {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in summary.items()},
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
