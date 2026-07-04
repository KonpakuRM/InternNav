"""Native (discrete) R2R graph evaluator.

The implementation lives in ``internnav.r2r.evaluation`` so it stays importable
without the heavy simulator/server dependencies pulled in by
``internnav.evaluator.__init__``. This module re-exports it under the
conventional evaluator location.
"""

from internnav.r2r.evaluation import (
    FailureAttribution,
    R2RGraphEvaluator,
    R2RGraphMetrics,
    attribute_failures,
    dtw,
    load_r2r_episodes,
)

__all__ = [
    'FailureAttribution',
    'R2RGraphEvaluator',
    'R2RGraphMetrics',
    'attribute_failures',
    'dtw',
    'load_r2r_episodes',
]
