"""
NRI Springs benchmark — THE standard benchmark for interaction graph discovery.

From "Neural Relational Inference" (Kipf et al., ICML 2018).
N particles connected by springs with different spring constants.
The model must discover which particles are connected (graph structure)
and predict their dynamics.

Compositional split: hold out specific (spring_constant, particle_type) combinations.

This benchmark is used by: NRI, dNRI, EGNN, VCDN, fNRI, ACD, and many more.
"""

import math
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
import numpy as np


class NRISpringsDataset(Dataset):
    """NRI Springs benchmark.

    N particles in 2D, some connected by springs with varying constants.
    Connected particles exert spring forces; unconnected ones don't interact.
    The graph structure (which particles are connected) is the GT causal graph.

    State per particle: [x, y, vx, vy] = 4 dims
    """

    def __init__(self, num_videos: int = 5000, num_frames: int = 49,
                 num_particles: int = 5, num_frame_samples: int = 16,
                 spring_types: int = 3,
                 box_size: float = 5.0, dt: float = 0.001,
                 steps_per_frame: int = 100,
                 seed: int = 42, interaction_strength: float = 0.1):
        """
        Args:
            num_videos: number of simulations
            num_frames: total simulation frames
            num_particles: particles per simulation
            num_frame_samples: frames to return per sample (subsampled)
            spring_types: number of different spring constants (0=no spring)
            box_size: simulation box half-size
            dt: simulation timestep
            steps_per_frame: integration steps between recorded frames
            seed: random seed
            interaction_strength: base spring constant
        """
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.num_particles = num_particles
        self.num_frame_samples = num_frame_samples
        self.spring_types = spring_types
        self.box_size = box_size
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        self.interaction_strength = interaction_strength

        # Pre-generate all simulations
        rng = np.random.RandomState(seed)
        self.data = []

        for vid in range(num_videos):
            sample = self._simulate_one(rng)
            self.data.append(sample)

        print(f"NRI Springs: generated {num_videos} simulations, "
              f"{num_particles} particles, {spring_types} spring types")

    def _simulate_one(self, rng):
        N = self.num_particles
        T = self.num_frames

        # Random initial positions and velocities
        positions = rng.randn(N, 2) * 0.5
        velocities = rng.randn(N, 2) * 0.5

        # Random spring connections: edges[i,j] ∈ {0, 1, ..., spring_types}
        # 0 = no connection, 1..spring_types = different spring constants
        edges = np.zeros((N, N), dtype=np.int64)
        for i in range(N):
            for j in range(i + 1, N):
                edge_type = rng.randint(0, self.spring_types + 1)  # 0 to spring_types
                edges[i, j] = edge_type
                edges[j, i] = edge_type

        # Spring constants for each type
        spring_constants = [0.0] + [
            self.interaction_strength * (k + 1) for k in range(self.spring_types)
        ]

        # Simulate
        all_positions = []
        all_velocities = []

        for t in range(T):
            all_positions.append(positions.copy())
            all_velocities.append(velocities.copy())

            for _ in range(self.steps_per_frame):
                # Compute forces
                forces = np.zeros((N, 2))

                for i in range(N):
                    for j in range(N):
                        if i == j or edges[i, j] == 0:
                            continue
                        k = spring_constants[edges[i, j]]
                        diff = positions[j] - positions[i]
                        dist = np.sqrt(np.sum(diff**2)) + 1e-8
                        # Spring force: F = k * (dist - rest_length) * direction
                        rest_length = 1.0
                        force_mag = k * (dist - rest_length)
                        forces[i] += force_mag * diff / dist

                # Update velocities and positions (Euler integration)
                velocities += forces * self.dt
                positions += velocities * self.dt

                # Damping
                velocities *= 0.999

                # Box boundaries (soft walls)
                for i in range(N):
                    for d in range(2):
                        if positions[i, d] > self.box_size:
                            positions[i, d] = self.box_size
                            velocities[i, d] *= -0.8
                        elif positions[i, d] < -self.box_size:
                            positions[i, d] = -self.box_size
                            velocities[i, d] *= -0.8

        all_positions = np.array(all_positions)  # [T, N, 2]
        all_velocities = np.array(all_velocities)  # [T, N, 2]

        # Subsample frames
        indices = np.linspace(0, T - 1, self.num_frame_samples, dtype=int)
        positions_sub = all_positions[indices]
        velocities_sub = all_velocities[indices]

        # Normalize positions to [0, 1]
        pos_min = all_positions.min()
        pos_max = all_positions.max()
        pos_range = max(pos_max - pos_min, 1e-6)
        positions_norm = (positions_sub - pos_min) / pos_range
        velocities_norm = velocities_sub / (pos_range + 1e-6)

        return {
            "positions": positions_norm,      # [T_sub, N, 2]
            "velocities": velocities_norm,    # [T_sub, N, 2]
            "edges": edges,                   # [N, N] int, 0=no edge
            "edge_exists": (edges > 0).astype(np.float32),  # [N, N] binary
        }

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        d = self.data[idx]
        N = self.num_particles
        T = self.num_frame_samples

        # GT state: [x, y, vx, vy] = 4 dims
        gt_states = torch.zeros(T, N, 4)
        gt_states[:, :, :2] = torch.tensor(d["positions"], dtype=torch.float32)
        gt_states[:, :, 2:] = torch.tensor(d["velocities"], dtype=torch.float32)

        # Edge adjacency (binary: connected or not)
        collision_adj = torch.tensor(d["edge_exists"], dtype=torch.float32)
        collision_adj = collision_adj.unsqueeze(0).expand(T, -1, -1)  # [T, N, N]

        # Typed edges for evaluation
        interaction_adj = torch.tensor(d["edges"], dtype=torch.long)
        interaction_adj = interaction_adj.unsqueeze(0).expand(T, -1, -1)  # [T, N, N]

        positions = torch.tensor(d["positions"], dtype=torch.float32)  # [T, N, 2]

        return {
            "gt_states": gt_states,         # [T, N, 4]
            "collision_adj": collision_adj,  # [T, N, N] binary spring connections
            "interaction_adj": interaction_adj,  # [T, N, N] typed (spring constant)
            "positions": positions,
            "video_id": f"springs_{idx:05d}",
            "objects": {"num_objects": N, "properties": [
                {"color": f"p{i}", "shape": "particle", "material": "default"}
                for i in range(N)
            ]},
            "events": [],
            "num_objects": N,
            "video": torch.zeros(T, 3, 64, 64),  # placeholder
        }


def springs_collate_fn(batch):
    return {
        "gt_states": torch.stack([b["gt_states"] for b in batch]),
        "collision_adj": torch.stack([b["collision_adj"] for b in batch]),
        "interaction_adj": torch.stack([b["interaction_adj"] for b in batch]),
        "positions": torch.stack([b["positions"] for b in batch]),
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
        "num_objects": [b["num_objects"] for b in batch],
        "video": torch.stack([b["video"] for b in batch]),
    }


class NBodyChargedDataset(Dataset):
    """N-Body Charged Particles benchmark.

    From Interaction Networks (Battaglia et al., NeurIPS 2016).
    N charged particles interact via Coulomb forces.
    Same-sign charges repel, opposite attract.

    State per particle: [x, y, vx, vy, charge] = 5 dims
    """

    def __init__(self, num_videos: int = 5000, num_frames: int = 16,
                 num_particles: int = 5, dt: float = 0.001,
                 steps_per_frame: int = 100, seed: int = 42):
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.num_particles = num_particles

        rng = np.random.RandomState(seed)
        self.data = []

        for _ in range(num_videos):
            self.data.append(self._simulate_one(rng, num_particles, num_frames,
                                                 dt, steps_per_frame))

        print(f"N-Body Charged: generated {num_videos} simulations, {num_particles} particles")

    def _simulate_one(self, rng, N, T, dt, steps_per_frame):
        positions = rng.randn(N, 2) * 0.5
        velocities = rng.randn(N, 2) * 0.3
        charges = rng.choice([-1.0, 1.0], size=N)
        masses = rng.uniform(0.5, 2.0, size=N)

        all_states = []

        for t in range(T):
            state = np.zeros((N, 5))
            state[:, :2] = positions
            state[:, 2:4] = velocities
            state[:, 4] = charges
            all_states.append(state.copy())

            for _ in range(steps_per_frame):
                forces = np.zeros((N, 2))
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        diff = positions[j] - positions[i]
                        dist = np.sqrt(np.sum(diff**2)) + 0.1
                        # Coulomb: F = k * q1 * q2 / r^2 (repel same, attract opposite)
                        f_mag = -0.1 * charges[i] * charges[j] / (dist**2)
                        forces[i] += f_mag * diff / dist

                velocities += forces * dt / masses[:, None]
                positions += velocities * dt
                velocities *= 0.999

                # Soft walls
                mask = np.abs(positions) > 5.0
                velocities[mask] *= -0.8
                positions = np.clip(positions, -5.0, 5.0)

        all_states = np.array(all_states)  # [T, N, 5]

        # Normalize
        pos_vals = all_states[:, :, :2]
        pmin, pmax = pos_vals.min(), pos_vals.max()
        pr = max(pmax - pmin, 1e-6)
        all_states[:, :, :2] = (all_states[:, :, :2] - pmin) / pr
        all_states[:, :, 2:4] = all_states[:, :, 2:4] / (pr + 1e-6)

        # Interaction matrix: all pairs interact (gravity-like), strength varies
        interact = np.ones((N, N), dtype=np.float32)
        np.fill_diagonal(interact, 0)

        return {"states": all_states, "interact": interact, "charges": charges}

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        d = self.data[idx]
        N = self.num_particles
        T = self.num_frames
        states = torch.tensor(d["states"], dtype=torch.float32)
        interact = torch.tensor(d["interact"], dtype=torch.float32)

        return {
            "gt_states": states,  # [T, N, 5]
            "collision_adj": interact.unsqueeze(0).expand(T, -1, -1),
            "positions": states[:, :, :2],
            "video_id": f"nbody_{idx:05d}",
            "objects": {"num_objects": N, "properties": [
                {"color": "pos" if d["charges"][i] > 0 else "neg",
                 "shape": "particle", "material": "default"}
                for i in range(N)
            ]},
            "events": [],
            "num_objects": N,
            "video": torch.zeros(T, 3, 64, 64),
        }


def nbody_collate_fn(batch):
    return {
        "gt_states": torch.stack([b["gt_states"] for b in batch]),
        "collision_adj": torch.stack([b["collision_adj"] for b in batch]),
        "positions": torch.stack([b["positions"] for b in batch]),
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
        "num_objects": [b["num_objects"] for b in batch],
        "video": torch.stack([b["video"] for b in batch]),
    }
