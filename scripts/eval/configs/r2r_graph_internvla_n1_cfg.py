"""Config for evaluating InternVLA-N1 System 2 on native (discrete) R2R
with the graph executor. Used by scripts/eval/eval_r2r_graph.py.
"""

r2r_graph_cfg = {
    # ---- model (System 2, frozen; output_latent is never consumed) ----
    "model": {
        "policy_name": "InternVLAN1_Policy",
        "model_path": "checkpoints/InternVLA-N1-System2",
        "device": "cuda:0",
        "num_frames": 32,
        "num_history": 8,
        "num_future_steps": 4,
        "continuous_traj": True,
        "resize_w": 384,
        "resize_h": 384,
    },
    # ---- data ----
    "data": {
        "annotation_path": "data/r2r/R2R_val_unseen.json",
        "connectivity_dir": "data/connectivity",
        "max_episodes": None,
    },
    # ---- visual provider ----
    # 'habitat_graph': distribution-aligned diagnostic variant (R2R-Habitat-Graph,
    #                  RGB-D, supports history interpolation).
    # 'mattersim':     official R2R main results (RGB only, no interpolation).
    # 'prerendered':   official imagery served from a cache built by
    #                  scripts/eval/prerender_r2r.py when MatterSim is not
    #                  available at runtime.
    "provider": {
        "type": "habitat_graph",
        "scene_data_dir": "data/scene_datasets/mp3d",  # habitat_graph
        "scan_data_dir": "data/v1/scans",  # mattersim
        "pre_render_dir": "data/r2r_prerender",  # prerendered
        "width": 640,
        "height": 480,
        "hfov": 79.0,
    },
    # ---- executor ----
    "executor": {
        "max_graph_steps": 20,
        "max_view_adjustments": 4,
        # Official R2R runs must keep lambda_dist=0.0 (pure angular scoring)
        # and interpolate_history=False. The depth term and interpolation are
        # ablations for the habitat_graph diagnostic variant only.
        "lambda_dist": 0.0,
        "interpolate_history": False,
        "forward_fallback_angle_deg": 20.0,
        "reask_angle_deg": 45.0,
        "max_hops_per_pixel_goal": 3,
    },
    # ---- output ----
    "output_path": "./logs/r2r_graph/val_unseen",
    "log_dir": "./logs/r2r_graph/val_unseen/step_logs",
}
