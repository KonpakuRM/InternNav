"""Entry point for native (discrete) R2R evaluation with the graph executor.

Usage:
    python scripts/eval/eval_r2r_graph.py --config scripts/eval/configs/r2r_graph_internvla_n1_cfg.py
"""

import argparse
import importlib.util
import math


def load_cfg(path: str) -> dict:
    spec = importlib.util.spec_from_file_location('r2r_graph_cfg_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.r2r_graph_cfg


def build_provider(cfg: dict):
    provider_cfg = cfg['provider']
    provider_type = provider_cfg['type']
    if provider_type == 'habitat_graph':
        from internnav.r2r.providers.habitat_graph import HabitatGraphProvider

        return HabitatGraphProvider(
            scene_data_dir=provider_cfg['scene_data_dir'],
            connectivity_dir=cfg['data']['connectivity_dir'],
            width=provider_cfg['width'],
            height=provider_cfg['height'],
            hfov=provider_cfg['hfov'],
        )
    if provider_type == 'mattersim':
        from internnav.r2r.providers.mattersim import MatterSimProvider

        return MatterSimProvider(
            scan_data_dir=provider_cfg['scan_data_dir'],
            connectivity_dir=cfg['data']['connectivity_dir'],
            width=provider_cfg['width'],
            height=provider_cfg['height'],
        )
    if provider_type == 'prerendered':
        from internnav.r2r.providers.mattersim import PrerenderedProvider

        return PrerenderedProvider(
            cache_dir=provider_cfg['pre_render_dir'],
            width=provider_cfg['width'],
            height=provider_cfg['height'],
            hfov=provider_cfg['hfov'],
        )
    raise ValueError(f'Unknown provider type: {provider_type}')


def build_policy(cfg: dict):
    from internnav.model import get_config, get_policy

    model_cfg = cfg['model']
    policy_cls = get_policy(model_cfg['policy_name'])
    config_cls = get_config(model_cfg['policy_name'])
    policy = policy_cls(config=config_cls(model_cfg={'model': model_cfg}))
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='scripts/eval/configs/r2r_graph_internvla_n1_cfg.py')
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    from internnav.r2r.connectivity import ConnectivityCache
    from internnav.r2r.evaluation import R2RGraphEvaluator
    from internnav.r2r.graph_executor import ExecutorConfig, R2RGraphExecutor
    from internnav.r2r.planner_adapter import R2RPlannerAdapter

    executor_cfg = cfg['executor']
    config = ExecutorConfig(
        max_graph_steps=executor_cfg['max_graph_steps'],
        max_view_adjustments=executor_cfg['max_view_adjustments'],
        lambda_dist=executor_cfg['lambda_dist'],
        interpolate_history=executor_cfg['interpolate_history'],
        forward_fallback_angle=math.radians(executor_cfg['forward_fallback_angle_deg']),
        reask_angle=math.radians(executor_cfg['reask_angle_deg']),
        max_hops_per_pixel_goal=executor_cfg['max_hops_per_pixel_goal'],
        log_dir=cfg['log_dir'],
    )

    connectivity = ConnectivityCache(cfg['data']['connectivity_dir'])
    provider = build_provider(cfg)
    adapter = R2RPlannerAdapter(build_policy(cfg))
    executor = R2RGraphExecutor(adapter, provider, connectivity, config)

    evaluator = R2RGraphEvaluator(
        executor=executor,
        connectivity=connectivity,
        annotation_path=cfg['data']['annotation_path'],
        output_path=cfg['output_path'],
        max_episodes=cfg['data']['max_episodes'],
    )
    summary = evaluator.eval()
    print(summary)
    provider.close()


if __name__ == '__main__':
    main()
