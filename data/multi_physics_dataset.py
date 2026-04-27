"""
Multi-Physics synthetic environment (Environment 2).

More complex than the basic collision environment:
- Objects have DIFFERENT MASSES (affects collision dynamics)
- Two interaction types: ELASTIC COLLISION + GRAVITATIONAL ATTRACTION
- Objects can have charge: same-charge repel, opposite-charge attract
- Wall bouncing with friction

This tests whether CausalComp can discover AND differentiate
multiple interaction types, not just collision vs no-collision.
"""

import math
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

# Object properties
COLORS = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.0, 1.0),  # magenta
]
COLOR_NAMES = ["red", "green", "blue", "yellow", "cyan", "magenta"]
MASSES = [1.0, 2.0, 4.0]  # light, medium, heavy
MASS_NAMES = ["light", "medium", "heavy"]
CHARGES = [-1, 0, 1]  # negative, neutral, positive
CHARGE_NAMES = ["neg", "neutral", "pos"]


class MultiPhysicsDataset(Dataset):
    """Multi-physics environment with collisions + attractions + repulsions.

    Object types = (color, mass, charge) = 6 × 3 × 3 = 54 types
    Interaction types:
      0: no interaction (too far apart)
      1: elastic collision (contact)
      2: gravitational attraction (always, but weak)
      3: charge attraction (opposite charges)
      4: charge repulsion (same charges)
    """

    def __init__(self, num_videos: int = 5000, num_frames: int = 16,
                 resolution: int = 64, num_objects_range=(3, 5),
                 gravity_strength: float = 0.02,
                 charge_strength: float = 0.05,
                 max_velocity: float = 2.0,
                 seed: int = 42, comp_split: Optional[Dict] = None):
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_objects_range = num_objects_range
        self.gravity_strength = gravity_strength
        self.charge_strength = charge_strength
        self.max_velocity = max_velocity
        self.seed = seed

        rng = random.Random(seed)
        self.video_seeds = [rng.randint(0, 2**31) for _ in range(num_videos)]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.video_seeds[idx])
        R = self.resolution

        num_obj = rng.randint(*self.num_objects_range)
        max_obj = self.num_objects_range[1]

        # Generate objects
        objects = []
        for i in range(num_obj):
            radius = rng.choice([4, 6, 8])
            color_idx = i % len(COLORS)
            mass = rng.choice(MASSES)
            charge = rng.choice(CHARGES)
            x = rng.uniform(radius + 2, R - radius - 2)
            y = rng.uniform(radius + 2, R - radius - 2)
            vx = rng.uniform(-self.max_velocity, self.max_velocity)
            vy = rng.uniform(-self.max_velocity, self.max_velocity)
            objects.append({
                "x": x, "y": y, "vx": vx, "vy": vy,
                "radius": radius, "color": COLORS[color_idx],
                "color_name": COLOR_NAMES[color_idx],
                "mass": mass, "mass_name": MASS_NAMES[MASSES.index(mass)],
                "charge": charge, "charge_name": CHARGE_NAMES[CHARGES.index(charge)],
            })

        # Simulate
        frames = []
        gt_states = []
        events = []
        interaction_types = []  # per-frame interaction type matrix

        for t in range(self.num_frames):
            # Record state: [x, y, vx, vy, r, g, b, radius, mass, charge] = 10 dims
            state_t = []
            for o in objects:
                state_t.append([
                    o["x"] / R, o["y"] / R,
                    o["vx"] / self.max_velocity,
                    o["vy"] / self.max_velocity,
                    o["color"][0], o["color"][1], o["color"][2],
                    o["radius"] / 10.0,
                    o["mass"] / 4.0,  # normalize
                    o["charge"] / 1.0,
                ])
            gt_states.append(state_t)

            # Render
            frame = self._render_frame(objects, R)
            frames.append(frame)

            # Physics step with multiple interaction types
            step_events, step_types = self._physics_step(objects, R)
            events.extend([{"type": e[0], "frame": t, "objects": e[1]} for e in step_events])
            interaction_types.append(step_types)

        video = torch.stack(frames)

        # Build GT state tensor
        gt_states_tensor = torch.zeros(self.num_frames, max_obj, 10)
        for t in range(self.num_frames):
            for i, state in enumerate(gt_states[t]):
                if i < max_obj:
                    gt_states_tensor[t, i] = torch.tensor(state)

        # Build interaction type matrix [T, max_obj, max_obj]
        # 0=none, 1=collision, 2=gravity, 3=charge_attract, 4=charge_repel
        interaction_adj = torch.zeros(self.num_frames, max_obj, max_obj, dtype=torch.long)
        for t, types in enumerate(interaction_types):
            for (i, j), itype in types.items():
                if i < max_obj and j < max_obj:
                    interaction_adj[t, i, j] = itype
                    interaction_adj[t, j, i] = itype

        # Collision adjacency (binary, for edge supervision)
        collision_adj = torch.zeros(self.num_frames, max_obj, max_obj)
        for ev in events:
            if ev["type"] == "collision":
                i, j = ev["objects"]
                if i < max_obj and j < max_obj:
                    collision_adj[ev["frame"], i, j] = 1.0
                    collision_adj[ev["frame"], j, i] = 1.0

        # Any interaction adjacency (collision + close-range force)
        any_interact_adj = (interaction_adj > 0).float()

        obj_props = [{
            "color": o["color_name"],
            "shape": "circle",
            "material": o["mass_name"],
            "charge": o["charge_name"],
        } for o in objects]

        positions = torch.zeros(self.num_frames, max_obj, 2)
        for t in range(self.num_frames):
            for i in range(min(num_obj, max_obj)):
                positions[t, i] = gt_states_tensor[t, i, :2]

        return {
            "video": video,
            "video_id": f"multiphys_{idx:05d}",
            "objects": {"num_objects": num_obj, "properties": obj_props},
            "events": events,
            "gt_states": gt_states_tensor,
            "collision_adj": collision_adj,
            "interaction_adj": interaction_adj,  # typed interactions
            "any_interact_adj": any_interact_adj,
            "positions": positions,
            "num_objects": num_obj,
        }

    def _render_frame(self, objects, R):
        frame = torch.zeros(3, R, R)
        yy, xx = torch.meshgrid(
            torch.arange(R, dtype=torch.float32),
            torch.arange(R, dtype=torch.float32),
            indexing="ij",
        )
        for obj in objects:
            dist = torch.sqrt((xx - obj["x"])**2 + (yy - obj["y"])**2)
            mask = (dist < obj["radius"]).float()
            # Color intensity based on charge
            intensity = 0.7 if obj["charge"] == 0 else 1.0
            for c in range(3):
                frame[c] = torch.maximum(frame[c], mask * obj["color"][c] * intensity)
        return frame

    def _physics_step(self, objects, R):
        """Multi-physics step. Returns events and interaction type matrix."""
        events = []
        interactions = {}  # (i,j) → type

        n = len(objects)

        # Apply forces (gravity + charge)
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = objects[i], objects[j]
                dx = oj["x"] - oi["x"]
                dy = oj["y"] - oi["y"]
                dist = math.sqrt(dx**2 + dy**2) + 1e-6
                nx, ny = dx / dist, dy / dist

                # Gravitational attraction (always present, proportional to masses)
                f_grav = self.gravity_strength * oi["mass"] * oj["mass"] / (dist**2 + 10)
                oi["vx"] += f_grav * nx / oi["mass"]
                oi["vy"] += f_grav * ny / oi["mass"]
                oj["vx"] -= f_grav * nx / oj["mass"]
                oj["vy"] -= f_grav * ny / oj["mass"]

                if dist < 20:  # close enough to register as gravitational interaction
                    interactions[(i, j)] = 2  # gravity

                # Charge interaction
                if oi["charge"] != 0 and oj["charge"] != 0:
                    # Same charge → repel, opposite → attract
                    f_charge = -self.charge_strength * oi["charge"] * oj["charge"] / (dist**2 + 5)
                    oi["vx"] += f_charge * nx / oi["mass"]
                    oi["vy"] += f_charge * ny / oi["mass"]
                    oj["vx"] -= f_charge * nx / oj["mass"]
                    oj["vy"] -= f_charge * ny / oj["mass"]

                    if dist < 25:
                        if oi["charge"] * oj["charge"] < 0:
                            interactions[(i, j)] = 3  # charge attract
                        else:
                            interactions[(i, j)] = 4  # charge repel

        # Move objects
        for obj in objects:
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]
            # Damping
            obj["vx"] *= 0.99
            obj["vy"] *= 0.99

        # Wall collisions
        for obj in objects:
            r = obj["radius"]
            if obj["x"] - r < 0:
                obj["x"] = r; obj["vx"] = abs(obj["vx"]) * 0.9
            elif obj["x"] + r > R:
                obj["x"] = R - r; obj["vx"] = -abs(obj["vx"]) * 0.9
            if obj["y"] - r < 0:
                obj["y"] = r; obj["vy"] = abs(obj["vy"]) * 0.9
            elif obj["y"] + r > R:
                obj["y"] = R - r; obj["vy"] = -abs(obj["vy"]) * 0.9

        # Object-object collisions (mass-dependent)
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = objects[i], objects[j]
                dx = oi["x"] - oj["x"]
                dy = oi["y"] - oj["y"]
                dist = math.sqrt(dx**2 + dy**2)
                min_dist = oi["radius"] + oj["radius"]

                if dist < min_dist and dist > 0:
                    events.append(("collision", [i, j]))
                    interactions[(i, j)] = 1  # collision overrides other types

                    nx_c, ny_c = dx / dist, dy / dist
                    dvx = oi["vx"] - oj["vx"]
                    dvy = oi["vy"] - oj["vy"]
                    dvn = dvx * nx_c + dvy * ny_c

                    if dvn < 0:
                        # Mass-dependent elastic collision
                        m1, m2 = oi["mass"], oj["mass"]
                        oi["vx"] -= (2 * m2 / (m1 + m2)) * dvn * nx_c
                        oi["vy"] -= (2 * m2 / (m1 + m2)) * dvn * ny_c
                        oj["vx"] += (2 * m1 / (m1 + m2)) * dvn * nx_c
                        oj["vy"] += (2 * m1 / (m1 + m2)) * dvn * ny_c

                    overlap = min_dist - dist
                    oi["x"] += overlap * 0.5 * nx_c
                    oi["y"] += overlap * 0.5 * ny_c
                    oj["x"] -= overlap * 0.5 * nx_c
                    oj["y"] -= overlap * 0.5 * ny_c

        return events, interactions


def multi_physics_collate_fn(batch):
    videos = torch.stack([b["video"] for b in batch])
    result = {
        "video": videos,
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
    }
    if "gt_states" in batch[0]:
        result["gt_states"] = torch.stack([b["gt_states"] for b in batch])
        result["collision_adj"] = torch.stack([b["collision_adj"] for b in batch])
        result["interaction_adj"] = torch.stack([b["interaction_adj"] for b in batch])
        result["positions"] = torch.stack([b["positions"] for b in batch])
        result["num_objects"] = [b["num_objects"] for b in batch]
    return result
