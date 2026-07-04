"""Evaluator for native (discrete) R2R with the graph executor.

Runs ``R2RGraphExecutor`` on the standard R2R annotation files and reports
NE / SR / OSR / SPL / nDTW / SDTW computed with geodesic distances on the
connectivity graph. Results are written both as a local JSONL and in the
official R2R submission format (``instr_id`` + viewpoint trajectory).
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from internnav.r2r.connectivity import ConnectivityCache, GraphState
from internnav.r2r.graph_executor import EpisodeResult, R2RGraphExecutor

SUCCESS_DISTANCE = 3.0


def load_r2r_episodes(annotation_path: str) -> List[dict]:
    """Load R2R annotations (R2R_{split}.json), one entry per instruction."""
    with open(annotation_path) as f:
        data = json.load(f)
    episodes = []
    for item in data:
        for i, instruction in enumerate(item['instructions']):
            episodes.append(
                {
                    'instr_id': f"{item['path_id']}_{i}",
                    'scan': item['scan'],
                    'path': item['path'],
                    'heading': item['heading'],
                    'instruction': instruction,
                    'gt_length': item.get('distance'),
                }
            )
    return episodes


def dtw(distance_matrix: np.ndarray) -> float:
    """Classic dynamic-time-warping cost between two sequences."""
    n, m = distance_matrix.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = distance_matrix[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m])


@dataclass
class R2RGraphMetrics:
    ne: float = 0.0
    sr: float = 0.0
    osr: float = 0.0
    spl: float = 0.0
    ndtw: float = 0.0
    sdtw: float = 0.0
    count: int = 0


class R2RGraphEvaluator:
    def __init__(
        self,
        executor: R2RGraphExecutor,
        connectivity: ConnectivityCache,
        annotation_path: str,
        output_path: str,
        max_episodes: Optional[int] = None,
    ):
        self.executor = executor
        self.connectivity = connectivity
        self.episodes = load_r2r_episodes(annotation_path)
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    # ---------------------------------------------------------------- metrics
    def episode_metrics(self, episode: dict, result: EpisodeResult) -> Dict[str, float]:
        graph = self.connectivity.get(episode['scan'])
        gt_path = episode['path']
        goal = gt_path[-1]
        pred_path = [entry[0] for entry in result.trajectory]

        ne = graph.distance(pred_path[-1], goal)
        sr = float(ne < SUCCESS_DISTANCE)
        osr = float(any(graph.distance(vp, goal) < SUCCESS_DISTANCE for vp in pred_path))

        gt_length = sum(graph.distance(a, b) for a, b in zip(gt_path[:-1], gt_path[1:]))
        pred_length = sum(graph.distance(a, b) for a, b in zip(pred_path[:-1], pred_path[1:]))
        spl = sr * gt_length / max(pred_length, gt_length) if gt_length > 0 else sr

        dist_matrix = np.array([[graph.distance(p, g) for g in gt_path] for p in pred_path])
        ndtw = math.exp(-dtw(dist_matrix) / (len(gt_path) * SUCCESS_DISTANCE))
        sdtw = sr * ndtw
        return {'ne': ne, 'sr': sr, 'osr': osr, 'spl': spl, 'ndtw': ndtw, 'sdtw': sdtw}

    # ------------------------------------------------------------------- eval
    def eval(self) -> Dict[str, float]:
        results_file = os.path.join(self.output_path, 'results.jsonl')
        submission: List[dict] = []
        totals: Dict[str, float] = {}
        count = 0

        with open(results_file, 'w') as f:
            for episode in self.episodes:
                start = GraphState(scan=episode['scan'], viewpoint=episode['path'][0], heading=episode['heading'])
                result = self.executor.run_episode(episode['instruction'], start, episode_id=episode['instr_id'])
                metrics = self.episode_metrics(episode, result)

                record = {
                    'instr_id': episode['instr_id'],
                    'scan': episode['scan'],
                    'trajectory': result.trajectory,
                    'stop_called': result.stop_called,
                    'steps': result.steps,
                    **metrics,
                }
                f.write(json.dumps(record) + '\n')
                submission.append({'instr_id': episode['instr_id'], 'trajectory': result.trajectory})

                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1

        with open(os.path.join(self.output_path, 'submission.json'), 'w') as f:
            json.dump(submission, f)

        summary = {key: value / max(count, 1) for key, value in totals.items()}
        summary['count'] = count
        with open(os.path.join(self.output_path, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        return summary


@dataclass
class FailureAttribution:
    """Aggregates step logs into perception / matching / stop failure buckets."""

    perception: int = 0
    matching: int = 0
    stop: int = 0
    details: List[dict] = field(default_factory=list)


def attribute_failures(log_dir: str, results_file: str, connectivity: ConnectivityCache) -> FailureAttribution:
    """Coarse failure attribution from executor logs for failed episodes.

    - stop: agent passed within success distance of the goal but stopped elsewhere
      (or never stopped).
    - matching / perception: currently distinguished by whether the chosen
      candidate disagreed with the pixel-ray direction (matching) or the pixel
      goal itself pointed away from the ground-truth path (perception, requires
      manual inspection of logged frames).
    """
    attribution = FailureAttribution()
    with open(results_file) as f:
        for line in f:
            record = json.loads(line)
            if record['sr'] > 0:
                continue
            if record['osr'] > 0:
                attribution.stop += 1
                attribution.details.append({'instr_id': record['instr_id'], 'type': 'stop'})
            else:
                attribution.matching += 1
                attribution.details.append({'instr_id': record['instr_id'], 'type': 'matching_or_perception'})
    return attribution
