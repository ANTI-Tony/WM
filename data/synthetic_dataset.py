"""
Synthetic dataset: colored circles moving and colliding on black background.
Mimics CLEVRER-style physics without needing to download anything.

Each video has 3-6 circles with random colors, sizes, positions, and velocities.
Simple elastic collision physics.
"""

import math
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


# 6 distinct colors (RGB, normalized to [0,1])
COLORS = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.0, 1.0),  # magenta
]

# 3 size categories
SIZES = [4, 6, 8]  # radius in pixels


class SyntheticPhysicsDataset(Dataset):
    """Generates videos of colored circles moving and colliding.

    Each video is procedurally generated on-the-fly:
    - 3-6 objects with random color, size, position, velocity
    - Simple 2D elastic collisions + wall bouncing
    - Ground-truth causal events (which objects collided)
    """

    def __init__(self, num_videos: int = 5000, num_frames: int = 16,
                 resolution: int = 64, num_objects_range=(3, 6),
                 max_velocity: float = 3.0, seed: int = 42,
                 comp_split: Optional[Dict] = None):
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_objects_range = num_objects_range
        self.max_velocity = max_velocity
        self.seed = seed
        self.comp_split = comp_split

        # Pre-generate seeds for reproducibility
        rng = random.Random(seed)
        self.video_seeds = [rng.randint(0, 2**31) for _ in range(num_videos)]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.video_seeds[idx])
        R = self.resolution

        # Random number of objects
        num_obj = rng.randint(*self.num_objects_range)

        # Generate objects: position, velocity, color, radius
        objects = []
        for i in range(num_obj):
            radius = rng.choice(SIZES)
            color = COLORS[i % len(COLORS)]
            # Random position (avoid edges)
            x = rng.uniform(radius + 2, R - radius - 2)
            y = rng.uniform(radius + 2, R - radius - 2)
            # Random velocity
            vx = rng.uniform(-self.max_velocity, self.max_velocity)
            vy = rng.uniform(-self.max_velocity, self.max_velocity)
            objects.append({
                "x": x, "y": y, "vx": vx, "vy": vy,
                "radius": radius, "color": color,
                "color_name": ["red", "green", "blue", "yellow", "cyan", "magenta"][i % 6],
                "size_name": ["small", "medium", "large"][SIZES.index(radius)],
            })

        # Simulate physics and render frames
        frames = []
        events = []
        positions = []  # [T, num_obj, 2] object (x,y) per frame

        for t in range(self.num_frames):
            # Record positions before rendering
            pos_t = [[o["x"] / R, o["y"] / R] for o in objects]  # normalize to [0,1]
            positions.append(pos_t)

            # Render current frame
            frame = self._render_frame(objects, R)
            frames.append(frame)

            # Physics step
            collisions = self._physics_step(objects, R)
            for c in collisions:
                events.append({"type": "collision", "frame": t, "objects": c})

        video = torch.stack(frames)  # [T, 3, H, W]

        # Build per-frame collision adjacency matrix [T, max_obj, max_obj]
        max_obj = self.num_objects_range[1]
        collision_adj = torch.zeros(self.num_frames, max_obj, max_obj)
        for ev in events:
            t = ev["frame"]
            i, j = ev["objects"]
            if i < max_obj and j < max_obj:
                collision_adj[t, i, j] = 1.0
                collision_adj[t, j, i] = 1.0

        # Pad positions to max_obj
        positions_tensor = torch.zeros(self.num_frames, max_obj, 2)
        for t in range(self.num_frames):
            for i, pos in enumerate(positions[t]):
                if i < max_obj:
                    positions_tensor[t, i] = torch.tensor(pos)

        # Object properties for evaluation
        obj_props = [{
            "color": o["color_name"],
            "shape": "circle",
            "material": o["size_name"],
        } for o in objects]

        return {
            "video": video,
            "video_id": f"synthetic_{idx:05d}",
            "objects": {"num_objects": num_obj, "properties": obj_props},
            "events": events,
            "positions": positions_tensor,    # [T, max_obj, 2]
            "collision_adj": collision_adj,   # [T, max_obj, max_obj]
        }

    def _render_frame(self, objects: list, R: int) -> torch.Tensor:
        """Render all objects to a frame. Returns [3, H, W]."""
        frame = torch.zeros(3, R, R)

        # Create coordinate grids
        yy, xx = torch.meshgrid(
            torch.arange(R, dtype=torch.float32),
            torch.arange(R, dtype=torch.float32),
            indexing="ij",
        )

        for obj in objects:
            # Distance from object center
            dist = torch.sqrt((xx - obj["x"])**2 + (yy - obj["y"])**2)
            mask = (dist < obj["radius"]).float()

            # Anti-aliasing at edges
            edge = ((dist >= obj["radius"] - 1) & (dist < obj["radius"])).float()
            alpha = mask - edge * 0.5

            # Paint color
            for c in range(3):
                frame[c] = torch.maximum(frame[c], alpha * obj["color"][c])

        return frame

    def _physics_step(self, objects: list, R: int) -> list:
        """Advance physics by one step. Returns list of collision pairs."""
        collisions = []

        # Move objects
        for obj in objects:
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

        # Wall collisions
        for obj in objects:
            r = obj["radius"]
            if obj["x"] - r < 0:
                obj["x"] = r
                obj["vx"] = abs(obj["vx"])
            elif obj["x"] + r > R:
                obj["x"] = R - r
                obj["vx"] = -abs(obj["vx"])
            if obj["y"] - r < 0:
                obj["y"] = r
                obj["vy"] = abs(obj["vy"])
            elif obj["y"] + r > R:
                obj["y"] = R - r
                obj["vy"] = -abs(obj["vy"])

        # Object-object collisions (elastic)
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                oi, oj = objects[i], objects[j]
                dx = oi["x"] - oj["x"]
                dy = oi["y"] - oj["y"]
                dist = math.sqrt(dx**2 + dy**2)
                min_dist = oi["radius"] + oj["radius"]

                if dist < min_dist and dist > 0:
                    # Collision detected
                    collisions.append([i, j])

                    # Normal vector
                    nx, ny = dx / dist, dy / dist

                    # Relative velocity along normal
                    dvx = oi["vx"] - oj["vx"]
                    dvy = oi["vy"] - oj["vy"]
                    dvn = dvx * nx + dvy * ny

                    if dvn < 0:  # approaching
                        # Equal mass elastic collision
                        oi["vx"] -= dvn * nx
                        oi["vy"] -= dvn * ny
                        oj["vx"] += dvn * nx
                        oj["vy"] += dvn * ny

                    # Separate overlapping objects
                    overlap = min_dist - dist
                    oi["x"] += overlap * 0.5 * nx
                    oi["y"] += overlap * 0.5 * ny
                    oj["x"] -= overlap * 0.5 * nx
                    oj["y"] -= overlap * 0.5 * ny

        return collisions


def synthetic_collate_fn(batch: List[dict]) -> dict:
    """Collate function for synthetic dataset."""
    videos = torch.stack([b["video"] for b in batch])
    result = {
        "video": videos,
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
    }
    # Stack GT supervision signals if available
    if "collision_adj" in batch[0]:
        result["collision_adj"] = torch.stack([b["collision_adj"] for b in batch])
        result["positions"] = torch.stack([b["positions"] for b in batch])
    return result
