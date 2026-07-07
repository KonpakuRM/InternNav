"""Prerender official R2R imagery into a cache for PrerenderedProvider.

Renders ``num_headings x 3`` discretized views (12 headings, elevations
-30/0/+30 deg) per viewpoint with MatterSim, so evaluation machines without a
MatterSim runtime can still use native R2R imagery.

Usage:
    python scripts/eval/prerender_r2r.py --scan_data_dir data/v1/scans \
        --connectivity_dir data/connectivity --output_dir data/r2r_prerender
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_data_dir', required=True)
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--vfov', type=float, default=60.0)
    parser.add_argument('--num_headings', type=int, default=12)
    args = parser.parse_args()

    import MatterSim

    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.scan_data_dir)
    sim.setNavGraphPath(args.connectivity_dir)
    sim.setCameraResolution(args.width, args.height)
    sim.setCameraVFOV(math.radians(args.vfov))
    sim.setDiscretizedViewingAngles(False)
    sim.setRenderingEnabled(True)
    sim.initialize()

    with open(os.path.join(args.connectivity_dir, 'scans.txt')) as f:
        scans = [line.strip() for line in f if line.strip()]

    heading_step = 2 * math.pi / args.num_headings
    elevations = [-math.radians(30.0), 0.0, math.radians(30.0)]

    for scan in scans:
        with open(os.path.join(args.connectivity_dir, f'{scan}_connectivity.json')) as f:
            nodes = [item['image_id'] for item in json.load(f) if item['included']]
        for viewpoint in nodes:
            out_dir = os.path.join(args.output_dir, scan, viewpoint)
            os.makedirs(out_dir, exist_ok=True)
            for heading_idx in range(args.num_headings):
                for elev_idx, elevation in enumerate(elevations):
                    out_path = os.path.join(out_dir, f'{heading_idx:02d}_{elev_idx}.png')
                    if os.path.exists(out_path):
                        continue
                    sim.newEpisode([scan], [viewpoint], [heading_idx * heading_step], [elevation])
                    state = sim.getState()[0]
                    rgb = np.array(state.rgb, copy=True)[..., ::-1]
                    Image.fromarray(rgb).save(out_path)
        print(f'Done: {scan}')


if __name__ == '__main__':
    main()
