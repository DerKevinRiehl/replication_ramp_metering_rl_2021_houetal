"""
Helper module for collecting per-step vehicle speeds in spatial bins along
the freeway mainline. Used during evaluation (baselines + RL) to later
reproduce Figure 6 from Hou et al. (2021).

Overview:

  1) env.py owns one SpaceSpeedLogger Object per episode (only when the
     `collect_space_speed` flag is set to true).
  2) Each step the env calls record_step(), which iterates over the
     mainline edges (seg1..seg8) and accumulates sum-of-speeds + count
     per spatial bin into (1 dimensional) arrays.
  3) At episode end the env calls save(...), which writes one .npz file
     containing the accumulated bin arrays. The plotting script
     (scripts/plot_figure6.py) later aggregates these files across
     episodes per controller.
"""

import numpy as np
import traci
from pathlib import Path


#start coordinates of each mainline edge along the freeway (global longitude of a vehicle is then EDGE_X_OFFSETS[edge] + lane_position.)
EDGE_X_OFFSETS = {
    "seg1": 0.0,
    "seg2": 100.0,
    "seg3": 200.0,
    "seg4": 300.0,
    "seg5": 400.0,
    "seg6": 492.06,
    "seg7": 654.00,
    "seg8": 750.00,
}

MAINLINE_EDGES = list(EDGE_X_OFFSETS.keys())

# Merging area extent (seg6). Stored here so the plot script can draw the red rectangle without hard-coding geometry.
MERGE_X_START = 492.06
MERGE_X_END   = 654.68


class SpaceSpeedLogger:

    def __init__(self, bin_width_m=5.0, x_max_m=850.0):
        self.bin_width_m = bin_width_m
        self.x_max_m = x_max_m

        # number of spatial bins along the mainline (seg1..seg8 ≈ 850 m)
        self.n_bins = int(np.ceil(x_max_m / bin_width_m))
        self.bin_edges = np.arange(self.n_bins + 1, dtype=np.float32) * bin_width_m

        self.sum_v = np.zeros(self.n_bins, dtype=np.float64)
        self.cnt   = np.zeros(self.n_bins, dtype=np.int64)

    def record_step(self):
        
        #Snapshot every vehicle on the mainline and add its speed to its bin.

        for edge in MAINLINE_EDGES:
            offset = EDGE_X_OFFSETS[edge]
            for vid in traci.edge.getLastStepVehicleIDs(edge):
                x = offset + traci.vehicle.getLanePosition(vid)
                if x < 0.0 or x >= self.x_max_m:
                    continue
                b = int(x // self.bin_width_m)
                self.sum_v[b] += traci.vehicle.getSpeed(vid)
                self.cnt[b]   += 1


    def save(self, out_path):
        
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            sum_v=self.sum_v.astype(np.float32),
            cnt=self.cnt.astype(np.int64),
            bin_edges=self.bin_edges,
            merge_x_start=np.float32(MERGE_X_START),
            merge_x_end=np.float32(MERGE_X_END),
        )
