import json
import math

import numpy as np
import pytest

pytest.importorskip('networkx')

from internnav.r2r.connectivity import (
    ConnectivityCache,
    GraphState,
    interpolate_positions,
    load_connectivity,
)
from internnav.r2r.graph_executor import ExecutorConfig, R2RGraphExecutor
from internnav.r2r.planner_adapter import R2RPlannerAdapter
from internnav.r2r.providers.base import Observation, VisualProvider
from internnav.r2r.utils import (
    camera_pose_from_state,
    camera_to_world,
    normalize_angle,
    pixel_to_bearing_elevation,
    pixel_to_camera_point,
    relative_heading_elevation,
    robust_pixel_depth,
)

WIDTH, HEIGHT, HFOV = 640, 480, 90.0


def make_intrinsic(width=WIDTH, height=HEIGHT, hfov=HFOV):
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array([[fx, 0, cx, 0], [0, fx, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


# --------------------------------------------------------------------- geometry
def test_pixel_center_has_zero_bearing_elevation():
    intrinsic = make_intrinsic()
    bearing, elevation = pixel_to_bearing_elevation((HEIGHT - 1) / 2.0, (WIDTH - 1) / 2.0, intrinsic)
    assert abs(bearing) < 1e-9
    assert abs(elevation) < 1e-9


def test_pixel_right_edge_bearing_matches_half_hfov():
    intrinsic = make_intrinsic()
    bearing, _ = pixel_to_bearing_elevation((HEIGHT - 1) / 2.0, WIDTH - 1.0, intrinsic)
    # cx/cy use the repo's (size - 1) / 2 convention, hence the half-pixel slack.
    assert bearing == pytest.approx(math.radians(HFOV / 2.0), abs=2e-3)
    bearing_left, _ = pixel_to_bearing_elevation((HEIGHT - 1) / 2.0, 0.0, intrinsic)
    assert bearing_left == pytest.approx(-math.radians(HFOV / 2.0), abs=2e-3)


def test_pixel_above_center_has_positive_elevation():
    intrinsic = make_intrinsic()
    _, elevation = pixel_to_bearing_elevation(0.0, (WIDTH - 1) / 2.0, intrinsic)
    assert elevation > 0


def test_unprojection_round_trip():
    intrinsic = make_intrinsic()
    point = pixel_to_camera_point(100.0, 400.0, 2.5, intrinsic)
    assert point[2] == pytest.approx(2.5)
    # Reproject.
    col = point[0] / point[2] * intrinsic[0, 0] + intrinsic[0, 2]
    row = point[1] / point[2] * intrinsic[1, 1] + intrinsic[1, 2]
    assert row == pytest.approx(100.0)
    assert col == pytest.approx(400.0)


def test_camera_to_world_identity_heading():
    # Agent at origin, heading 0 (facing +y), level: camera forward (+z) is world +y.
    pose = camera_pose_from_state(np.zeros(3), heading=0.0, elevation=0.0)
    world = camera_to_world(np.array([0.0, 0.0, 2.0]), pose)
    np.testing.assert_allclose(world, [0.0, 2.0, 0.0], atol=1e-9)
    # Camera +x (right) maps to world +x.
    world = camera_to_world(np.array([1.0, 0.0, 0.0]), pose)
    np.testing.assert_allclose(world, [1.0, 0.0, 0.0], atol=1e-9)
    # Camera +y (down) maps to world -z.
    world = camera_to_world(np.array([0.0, 1.0, 0.0]), pose)
    np.testing.assert_allclose(world, [0.0, 0.0, -1.0], atol=1e-9)


def test_camera_to_world_rotated_heading():
    # Heading pi/2 (facing +x): camera forward maps to world +x.
    pose = camera_pose_from_state(np.zeros(3), heading=math.pi / 2, elevation=0.0)
    world = camera_to_world(np.array([0.0, 0.0, 3.0]), pose)
    np.testing.assert_allclose(world, [3.0, 0.0, 0.0], atol=1e-9)


def test_relative_heading_elevation():
    rel_h, rel_e, dist = relative_heading_elevation(np.zeros(3), np.array([1.0, 1.0, 0.0]), heading=0.0)
    assert rel_h == pytest.approx(math.pi / 4)
    assert rel_e == pytest.approx(0.0)
    assert dist == pytest.approx(math.sqrt(2))
    # Positive elevation for a higher target.
    _, rel_e, _ = relative_heading_elevation(np.zeros(3), np.array([0.0, 1.0, 1.0]), heading=0.0)
    assert rel_e == pytest.approx(math.pi / 4)


def test_robust_pixel_depth_rejects_unreliable():
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    assert robust_pixel_depth(depth, 5, 5) == pytest.approx(2.0)
    # Too far.
    assert robust_pixel_depth(np.full((10, 10), 8.0, dtype=np.float32), 5, 5, max_depth=5.0) is None
    # High variance (edge).
    edge = np.full((10, 10), 2.0, dtype=np.float32)
    edge[:, 5:] = 5.0
    assert robust_pixel_depth(edge, 5, 5, var_threshold=0.25) is None
    # Invalid zeros.
    assert robust_pixel_depth(np.zeros((10, 10), dtype=np.float32), 5, 5) is None


# ----------------------------------------------------------------- connectivity
def make_connectivity(tmp_path, scan='testscan'):
    """Linear graph a -- b -- c, 2m apart along +y, with a side node d off b."""

    def node(image_id, x, y, z, unobstructed):
        pose = [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1]
        return {
            'image_id': image_id,
            'pose': pose,
            'included': True,
            'unobstructed': unobstructed,
            'visible': unobstructed,
            'height': 1.5,
        }

    data = [
        node('a', 0.0, 0.0, 1.5, [False, True, False, False]),
        node('b', 0.0, 2.0, 1.5, [True, False, True, True]),
        node('c', 0.0, 4.0, 1.5, [False, True, False, False]),
        node('d', 2.0, 2.0, 1.5, [False, True, False, False]),
    ]
    path = tmp_path / f'{scan}_connectivity.json'
    path.write_text(json.dumps(data))
    return str(tmp_path)


def test_load_connectivity(tmp_path):
    conn_dir = make_connectivity(tmp_path)
    graph = load_connectivity(conn_dir, 'testscan')
    assert set(graph.neighbors('b')) == {'a', 'c', 'd'}
    assert graph.distance('a', 'c') == pytest.approx(4.0)
    assert graph.shortest_path('a', 'c') == ['a', 'b', 'c']


def test_candidates_relative_heading(tmp_path):
    conn_dir = make_connectivity(tmp_path)
    graph = load_connectivity(conn_dir, 'testscan')
    state = GraphState(scan='testscan', viewpoint='b', heading=0.0)
    candidates = {c.viewpoint: c for c in graph.candidates(state)}
    assert candidates['c'].rel_heading == pytest.approx(0.0)  # straight ahead (+y)
    assert candidates['a'].rel_heading == pytest.approx(math.pi)  # behind
    assert candidates['d'].rel_heading == pytest.approx(math.pi / 2)  # right (+x)


def test_interpolate_positions():
    points = interpolate_positions(np.zeros(3), np.array([0.0, 2.0, 0.0]), step=0.25)
    assert len(points) == 8
    np.testing.assert_allclose(points[-1], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(points[0], [0.0, 0.25, 0.0])


# ---------------------------------------------------------------- fake policies
class FakePolicy:
    """Scripted S2: yields a fixed sequence of S2Output-like objects."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.llm_output = ''
        self.no_infer_calls = 0

    def reset(self):
        pass

    def step_no_infer(self, rgb, depth, pose):
        self.no_infer_calls += 1

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        return self.outputs.pop(0)


class FakeS2Output:
    def __init__(self, pixel=None, action=None):
        self.output_pixel = pixel
        self.output_action = action
        self.output_latent = None


class FakeProvider(VisualProvider):
    supports_depth = True
    supports_interpolation = True

    def __init__(self, connectivity, depth_value=2.0):
        self.connectivity = connectivity
        self.depth_value = depth_value
        self.intrinsic = make_intrinsic()

    def observe(self, state: GraphState) -> Observation:
        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        depth = np.full((HEIGHT, WIDTH), self.depth_value, dtype=np.float32)
        position = self.connectivity.get(state.scan).position(state.viewpoint)
        cam_to_world = camera_pose_from_state(position, state.heading, state.elevation)
        return Observation(rgb=rgb, depth=depth, intrinsic=self.intrinsic, cam_to_world=cam_to_world)

    def interpolate_observations(self, from_state, to_state, step=0.25):
        graph = self.connectivity.get(from_state.scan)
        num = int(
            math.ceil(np.linalg.norm(graph.position(to_state.viewpoint) - graph.position(from_state.viewpoint)) / step)
        )
        return [self.observe(from_state) for _ in range(max(num - 1, 0))]


# ------------------------------------------------------------------- adapter
def test_adapter_plan_kinds():
    center_pixel = np.array([(HEIGHT - 1) // 2, (WIDTH - 1) // 2])
    policy = FakePolicy(
        [
            FakeS2Output(pixel=center_pixel),
            FakeS2Output(action=[0]),
            FakeS2Output(action=[1]),
            FakeS2Output(action=[2, 2]),
            FakeS2Output(action=[5]),
        ]
    )
    adapter = R2RPlannerAdapter(policy)
    intrinsic = make_intrinsic()
    dummy = dict(rgb=None, depth=None, pose=np.eye(4), instruction='go', intrinsic=intrinsic)

    plan = adapter.plan(dummy['rgb'], dummy['depth'], dummy['pose'], dummy['instruction'], intrinsic)
    assert plan.kind == 'pixel' and plan.pixel[0] == center_pixel[0]
    assert adapter.plan(None, None, np.eye(4), 'go', intrinsic).kind == 'stop'
    assert adapter.plan(None, None, np.eye(4), 'go', intrinsic).kind == 'forward'
    plan = adapter.plan(None, None, np.eye(4), 'go', intrinsic)
    assert plan.kind == 'view_adjust' and plan.view_actions == [2, 2]
    assert adapter.plan(None, None, np.eye(4), 'go', intrinsic).kind == 'view_adjust'


# ------------------------------------------------------------------- scoring
def build_executor(tmp_path, policy, config=None):
    conn_dir = make_connectivity(tmp_path)
    connectivity = ConnectivityCache(conn_dir)
    provider = FakeProvider(connectivity)
    adapter = R2RPlannerAdapter(policy)
    executor = R2RGraphExecutor(adapter, provider, connectivity, config or ExecutorConfig())
    return executor, connectivity


def test_score_candidates_angular_and_revisit(tmp_path):
    executor, connectivity = build_executor(tmp_path, FakePolicy([]))
    state = GraphState(scan='testscan', viewpoint='b', heading=0.0)
    candidates = connectivity.get('testscan').candidates(state)

    # Pixel ray straight ahead: candidate c (rel_heading 0) must win.
    scored = executor.score_candidates(candidates, 0.0, 0.0, None, {})
    assert scored[0]['viewpoint'] == 'c'

    # Revisit penalty flips the choice toward the unvisited side node.
    scored = executor.score_candidates(candidates, 0.0, 0.0, None, {'c': 2})
    assert scored[0]['viewpoint'] != 'c'

    # Pixel ray to the right: d wins.
    scored = executor.score_candidates(candidates, math.pi / 2, 0.0, None, {})
    assert scored[0]['viewpoint'] == 'd'


def test_score_candidates_distance_consistency(tmp_path):
    config = ExecutorConfig(lambda_dist=0.3)
    executor, connectivity = build_executor(tmp_path, FakePolicy([]), config)
    state = GraphState(scan='testscan', viewpoint='a', heading=0.0)
    candidates = connectivity.get('testscan').candidates(state)
    # Only neighbor is b at 2m; distance term is zero when goal distance matches.
    scored = executor.score_candidates(candidates, 0.0, 0.0, 2.0, {})
    assert scored[0]['dist_term'] == pytest.approx(0.0)
    scored = executor.score_candidates(candidates, 0.0, 0.0, 4.0, {})
    assert scored[0]['dist_term'] == pytest.approx(0.5)


# ---------------------------------------------------------------- smoke tests
def center_pixel():
    return np.array([(HEIGHT - 1) // 2, (WIDTH - 1) // 2])


def test_episode_pixel_goals_then_stop(tmp_path):
    # a -> b -> c via center pixel goals (straight ahead), then STOP.
    policy = FakePolicy(
        [
            FakeS2Output(pixel=center_pixel()),
            FakeS2Output(pixel=center_pixel()),
            FakeS2Output(action=[0]),
        ]
    )
    config = ExecutorConfig(log_dir=str(tmp_path / 'logs'))
    executor, _ = build_executor(tmp_path, policy, config)
    start = GraphState(scan='testscan', viewpoint='a', heading=0.0)
    result = executor.run_episode('walk straight', start, episode_id='ep0')

    assert result.stop_called
    assert [entry[0] for entry in result.trajectory] == ['a', 'b', 'c']
    # Structured log exists and is parseable.
    log_lines = (tmp_path / 'logs' / 'ep0.jsonl').read_text().strip().splitlines()
    records = [json.loads(line) for line in log_lines]
    assert records[0]['s2_type'] == 'pixel'
    assert records[-1]['s2_type'] == 'stop'
    assert 'candidates' in records[0] and records[0]['chosen'] == 'b'


def test_episode_view_adjust_budget(tmp_path):
    # S2 keeps turning; after the budget is exhausted the executor falls back
    # to the forward-most candidate.
    policy = FakePolicy([FakeS2Output(action=[2])] * 5 + [FakeS2Output(action=[0])])
    config = ExecutorConfig(max_view_adjustments=4)
    executor, _ = build_executor(tmp_path, policy, config)
    start = GraphState(scan='testscan', viewpoint='a', heading=0.0)
    result = executor.run_episode('turn around', start)
    assert [entry[0] for entry in result.trajectory][:2] == ['a', 'b']


def test_episode_forward_fallback(tmp_path):
    # FORWARD with a candidate straight ahead moves; then STOP.
    policy = FakePolicy([FakeS2Output(action=[1]), FakeS2Output(action=[0])])
    executor, _ = build_executor(tmp_path, policy)
    start = GraphState(scan='testscan', viewpoint='a', heading=0.0)
    result = executor.run_episode('go forward', start)
    assert [entry[0] for entry in result.trajectory] == ['a', 'b']
    assert result.stop_called


def test_episode_interpolated_history(tmp_path):
    policy = FakePolicy([FakeS2Output(pixel=center_pixel()), FakeS2Output(action=[0])])
    config = ExecutorConfig(interpolate_history=True)
    executor, _ = build_executor(tmp_path, policy, config)
    start = GraphState(scan='testscan', viewpoint='a', heading=0.0)
    executor.run_episode('walk', start)
    # 2m hop at 0.25m spacing -> 7 intermediate frames fed via step_no_infer.
    assert policy.no_infer_calls == 7


def test_episode_max_steps_bound(tmp_path):
    # S2 always outputs a pixel goal; the episode must terminate at max steps.
    policy = FakePolicy([FakeS2Output(pixel=center_pixel()) for _ in range(50)])
    config = ExecutorConfig(max_graph_steps=5, revisit_penalty=0.0)
    executor, _ = build_executor(tmp_path, policy, config)
    start = GraphState(scan='testscan', viewpoint='a', heading=0.0)
    result = executor.run_episode('loop forever', start)
    assert result.steps == 5
    assert not result.stop_called


# ------------------------------------------------------------------ evaluator
def test_evaluator_metrics_and_submission(tmp_path):
    from internnav.r2r.evaluation import R2RGraphEvaluator, load_r2r_episodes

    make_connectivity(tmp_path)
    annotation = [
        {
            'path_id': 1,
            'scan': 'testscan',
            'heading': 0.0,
            'path': ['a', 'b', 'c'],
            'distance': 4.0,
            'instructions': ['walk straight to c'],
        }
    ]
    ann_path = tmp_path / 'R2R_tiny.json'
    ann_path.write_text(json.dumps(annotation))
    episodes = load_r2r_episodes(str(ann_path))
    assert episodes[0]['instr_id'] == '1_0'

    policy = FakePolicy(
        [FakeS2Output(pixel=center_pixel()), FakeS2Output(pixel=center_pixel()), FakeS2Output(action=[0])]
    )
    executor, connectivity = build_executor(tmp_path, policy)
    evaluator = R2RGraphEvaluator(
        executor=executor,
        connectivity=connectivity,
        annotation_path=str(ann_path),
        output_path=str(tmp_path / 'out'),
    )
    summary = evaluator.eval()
    assert summary['sr'] == pytest.approx(1.0)
    assert summary['ne'] == pytest.approx(0.0)
    assert summary['spl'] == pytest.approx(1.0)
    assert summary['ndtw'] == pytest.approx(1.0)

    submission = json.loads((tmp_path / 'out' / 'submission.json').read_text())
    assert submission[0]['instr_id'] == '1_0'
    assert [entry[0] for entry in submission[0]['trajectory']] == ['a', 'b', 'c']


def test_normalize_angle():
    assert normalize_angle(3 * math.pi) == pytest.approx(math.pi)
    assert normalize_angle(-3 * math.pi) == pytest.approx(math.pi)
    assert normalize_angle(0.5) == pytest.approx(0.5)
