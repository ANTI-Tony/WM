"""
PHYRE benchmark adapter for CausalComp.

PHYRE (Physical Reasoning) is a 2D physics benchmark from Meta.
Install: pip install phyre

We use PHYRE's simulation API to generate GT object states
(positions, velocities, radii) for training the GT-mode CausalComp.

PHYRE provides compositional generalization via cross-template evaluation:
- Within-template: same puzzle type, different parameters
- Cross-template: entirely different puzzle configurations

Reference: https://phyre.ai/
"""

import torch
from torch.utils.data import Dataset

try:
    import phyre
    PHYRE_AVAILABLE = True
except ImportError:
    PHYRE_AVAILABLE = False


class PHYREDataset(Dataset):
    """PHYRE benchmark dataset for GT-state world model training.

    Each sample is a sequence of object states from a PHYRE simulation.
    Objects are balls/bars with positions, velocities, and properties.
    """

    def __init__(self, tier: str = "ball", num_frames: int = 16,
                 frame_skip: int = 4, split: str = "train",
                 template_ids: list = None, max_videos: int = 2000,
                 seed: int = 42):
        """
        Args:
            tier: "ball" (1 ball) or "two_balls" (2 balls)
            num_frames: frames per clip
            frame_skip: subsample simulation steps
            split: "train" or "test"
            template_ids: list of template IDs to use (for comp split)
            max_videos: max number of simulations
            seed: random seed
        """
        assert PHYRE_AVAILABLE, "Install phyre: pip install phyre"

        self.num_frames = num_frames
        self.frame_skip = frame_skip

        # Setup PHYRE simulator
        eval_setup = f"ball_{split}_template" if tier == "ball" else f"two_balls_{split}_template"
        _, _, self.test_tasks = phyre.get_fold(eval_setup, 0)

        if template_ids is not None:
            self.test_tasks = [t for t in self.test_tasks
                               if t.split(":")[0] in template_ids]

        self.simulator = phyre.initialize_simulator(self.test_tasks[:max_videos], tier)

        # Pre-generate actions
        import numpy as np
        rng = np.random.RandomState(seed)
        self.actions = []
        for i in range(min(max_videos, len(self.test_tasks))):
            action = rng.uniform(0, 1, size=3)  # [x, y, radius]
            self.actions.append(action)

        # Run simulations and cache results
        self.samples = []
        for i, action in enumerate(self.actions):
            result = self.simulator.simulate_action(
                i, action, need_featurized_objects=True
            )
            if result.status.is_not_invalid():
                states = self._extract_states(result)
                if states is not None:
                    self.samples.append({
                        "gt_states": states,
                        "task_id": self.test_tasks[i],
                        "template": self.test_tasks[i].split(":")[0],
                        "solved": result.status.is_solved(),
                    })

        print(f"PHYRE: {len(self.samples)} valid simulations from {len(self.actions)} attempts")

    def _extract_states(self, result):
        """Extract object state sequence from PHYRE simulation result."""
        featurized = result.featurized_objects
        if featurized is None:
            return None

        # featurized.features: [T, num_obj, feature_dim]
        # Features: x, y, angle, diameter, shape_type, color_r, color_g, color_b, ...
        features = featurized.features
        T_total = features.shape[0]
        num_obj = features.shape[1]

        if T_total < self.num_frames * self.frame_skip:
            return None

        # Subsample frames
        indices = list(range(0, self.num_frames * self.frame_skip, self.frame_skip))
        features = features[indices]  # [num_frames, num_obj, feat_dim]

        # Extract: [x, y, vx, vy, diameter, r, g, b]
        states = torch.zeros(self.num_frames, num_obj, 8)
        for t in range(self.num_frames):
            for o in range(num_obj):
                f = features[t, o]
                states[t, o, 0] = f[0]  # x
                states[t, o, 1] = f[1]  # y
                # Estimate velocity from position difference
                if t > 0:
                    states[t, o, 2] = features[t, o, 0] - features[t-1, o, 0]  # vx
                    states[t, o, 3] = features[t, o, 1] - features[t-1, o, 1]  # vy
                states[t, o, 4] = f[3]  # diameter
                states[t, o, 5] = f[5]  # r
                states[t, o, 6] = f[6]  # g
                states[t, o, 7] = f[7]  # b

        return states

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        max_obj = s["gt_states"].shape[1]

        # Build collision adjacency (approximate: objects within distance threshold)
        collision_adj = torch.zeros(self.num_frames, max_obj, max_obj)
        for t in range(self.num_frames):
            for i in range(max_obj):
                for j in range(i+1, max_obj):
                    xi, yi = s["gt_states"][t, i, 0], s["gt_states"][t, i, 1]
                    xj, yj = s["gt_states"][t, j, 0], s["gt_states"][t, j, 1]
                    ri = s["gt_states"][t, i, 4] / 2
                    rj = s["gt_states"][t, j, 4] / 2
                    dist = ((xi-xj)**2 + (yi-yj)**2).sqrt()
                    if dist < ri + rj + 0.02:  # small margin
                        collision_adj[t, i, j] = 1.0
                        collision_adj[t, j, i] = 1.0

        return {
            "gt_states": s["gt_states"],
            "collision_adj": collision_adj,
            "video_id": s["task_id"],
            "template": s["template"],
            "objects": {"num_objects": s["gt_states"].shape[1], "properties": []},
            "events": [],
        }


def phyre_collate_fn(batch):
    """Collate PHYRE samples (may have different num_objects)."""
    # Pad to max num_objects in batch
    max_obj = max(b["gt_states"].shape[1] for b in batch)
    T = batch[0]["gt_states"].shape[0]

    gt_list, col_list = [], []
    for b in batch:
        n = b["gt_states"].shape[1]
        gt_padded = torch.zeros(T, max_obj, 8)
        gt_padded[:, :n] = b["gt_states"]
        gt_list.append(gt_padded)

        col_padded = torch.zeros(T, max_obj, max_obj)
        col_padded[:, :n, :n] = b["collision_adj"]
        col_list.append(col_padded)

    return {
        "gt_states": torch.stack(gt_list),
        "collision_adj": torch.stack(col_list),
        "video_ids": [b["video_id"] for b in batch],
        "templates": [b["template"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
    }
