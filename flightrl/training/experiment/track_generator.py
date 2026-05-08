"""Gate-primitive track generator for Condition C — IROS21 §III.D.

The paper's design:
  T = [G_1, ..., G_{j+1}]  where  G_{j+1} = f(G_j, Δp, ΔR)
Each gate is placed in the *previous gate's local frame* via a relative
position offset Δp and a relative rotation ΔR. A single scalar complexity
c ∈ [0, 1] linearly scales the magnitude of Δp's lateral/vertical components
and ΔR's per-step yaw change. At c=0 the track collapses to a straight line
(linear_8.yaml). At c=1 the per-gate offsets reach the configured
maxima. The forward step between gates is held constant — only the
direction-changing components scale with complexity.

Pitch and roll perturbations are deliberately omitted: the C++
applyPerturbedGates only takes a yaw per gate (the existing experiment
pipeline keeps gates upright). All complexity comes from yaw + lateral +
vertical offsets, which is consistent with the paper's described setup.
"""

from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


# ── Defaults tuned by visual inspection (see visualize_random_tracks.py). ──
# Forward distance preserved across complexities so the track length stays
# bounded; lateral/vertical/yaw bounds are what scale with c.
DEFAULT_NUM_GATES   = 8
DEFAULT_FORWARD_M   = 6.0       # forward step between gates (constant)
DEFAULT_LATERAL_M   = 4.0       # ± lateral offset at c=1
DEFAULT_VERTICAL_M  = 1.5       # ± vertical offset at c=1
DEFAULT_YAW_DEG     = 35.0      # ± per-gate yaw delta at c=1


class TrackGenerator:
    """Per-call generator. Reuse one instance across the training run."""

    def __init__(self,
                 num_gates: int = DEFAULT_NUM_GATES,
                 forward_m: float = DEFAULT_FORWARD_M,
                 lateral_max_m: float = DEFAULT_LATERAL_M,
                 vertical_max_m: float = DEFAULT_VERTICAL_M,
                 yaw_max_deg: float = DEFAULT_YAW_DEG,
                 start_pos=(0.0, 0.0, 5.0),
                 seed: int = 0):
        self.num_gates = int(num_gates)
        self.forward = float(forward_m)
        self.lateral_max = float(lateral_max_m)
        self.vertical_max = float(vertical_max_m)
        self.yaw_max = float(np.deg2rad(yaw_max_deg))
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_baseline_yaml(cls, yaml_path, **kwargs):
        """Match start_pos to a baseline YAML (e.g., linear_8.yaml)."""
        yaml = YAML()
        with open(yaml_path) as f:
            data = yaml.load(f)
        sp = list(data["start_pos"])
        return cls(start_pos=sp, **kwargs)

    def baseline(self):
        """Return the c=0 layout — 8 gates marching forward at z=start_z."""
        positions = np.zeros((self.num_gates, 3), dtype=np.float32)
        yaws = np.zeros(self.num_gates, dtype=np.float32)
        for i in range(self.num_gates):
            positions[i] = self.start_pos + np.array(
                [(i + 1) * self.forward, 0.0, 0.0], dtype=np.float32)
        return positions, yaws

    def sample(self, complexity: float):
        """Draw one track at a given complexity. Returns (positions[N,3],
        yaws_rad[N]).

        Construction (per gate primitive G_{i+1} = f(G_i, Δp, ΔR)):
          - Sample Δp_local = (forward, c·U(-lat_max, lat_max),
                               c·U(-vert_max, vert_max))   [previous frame]
          - Sample Δyaw     = c·U(-yaw_max, yaw_max)        [around z]
          - World offset    = Rz(yaw_prev) · Δp_local        [vector to next gate]
          - Next gate pose  = (pos_prev + offset, yaw_prev + Δyaw)
        At c=0 the track becomes the straight baseline; at c=1 each step
        can deviate up to (lateral_max, vertical_max, yaw_max).
        """
        c = float(np.clip(complexity, 0.0, 1.0))
        positions = np.zeros((self.num_gates, 3), dtype=np.float32)
        yaws = np.zeros(self.num_gates, dtype=np.float32)

        pos = self.start_pos.copy().astype(np.float32)
        yaw = 0.0  # world heading carried forward

        for i in range(self.num_gates):
            d_lat = c * self.rng.uniform(-self.lateral_max, self.lateral_max)
            d_vert = c * self.rng.uniform(-self.vertical_max, self.vertical_max)
            d_yaw = c * self.rng.uniform(-self.yaw_max, self.yaw_max)

            # Local Δp = (forward, d_lat, d_vert) expressed in current frame
            # Apply Rz(yaw) to lift to world.
            cy, sy = np.cos(yaw), np.sin(yaw)
            world_dx = cy * self.forward - sy * d_lat
            world_dy = sy * self.forward + cy * d_lat
            world_dz = d_vert

            pos = pos + np.array([world_dx, world_dy, world_dz],
                                 dtype=np.float32)
            # Advance the carried yaw AFTER the position step (paper-faithful:
            # the gate face rotates into the new heading, so the next forward
            # leg points along the new direction).
            yaw = yaw + d_yaw

            positions[i] = pos
            yaws[i] = yaw

        return positions, yaws.astype(np.float32)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    yaml_path = here.parents[2] / "flightlib" / "configs" / "tracks" / "linear_8.yaml"
    gen = TrackGenerator.from_baseline_yaml(yaml_path, seed=0)

    p0, y0 = gen.sample(0.0)
    bp, by = gen.baseline()
    assert np.allclose(p0, bp), "c=0 should match baseline (positions)"
    assert np.allclose(y0, by), "c=0 should match baseline (yaws)"
    print(f"OK c=0 matches baseline ({gen.num_gates} gates).")

    p_hi, y_hi = gen.sample(1.0)
    print(f"c=1 first 3 gates:\n{p_hi[:3]}\nyaws[deg]={np.rad2deg(y_hi[:3])}")
