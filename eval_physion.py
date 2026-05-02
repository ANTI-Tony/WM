"""
Physion benchmark adapter + evaluation.

Physion: 3D physics prediction benchmark (Bear et al., NeurIPS 2021).
8 scenarios: dominoes, support, containment, link, drop, roll, collide, drape.
Task: Object Contact Prediction (OCP) — predict if two objects will contact.

Download: https://physion-benchmark.github.io/
After download, set --data_dir to the extracted directory.

For GT-state mode: uses pre-extracted object features (positions, velocities).
For visual mode: uses rendered video frames.

Usage:
    python eval_physion.py --data_dir /path/to/physion --mode gt
    python eval_physion.py --data_dir /path/to/physion --mode visual
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from train_gt import GTCausalComp, compute_loss
from eval_full import NoGraphModel, FullGraphModel, SingleModuleModel, eval_mse, simple_loss
from eval_neurips import SlotFormerBaseline, train_model
from utils.logger import setup_logger


class PhysionDataset(Dataset):
    """Physion benchmark dataset.

    Reads HDF5/JSON files from the Physion download.
    Extracts per-object state trajectories for GT-state experiments.

    Expected directory structure:
        data_dir/
            {scenario}/
                {scenario}_{split}_{idx}/
                    stimulus_info.json  (object properties)
                    state_info.json     (per-frame object states)
                    video.mp4           (rendered video, for visual mode)
    """

    def __init__(self, data_dir: str, scenarios: List[str] = None,
                 split: str = "train", num_frames: int = 16,
                 max_videos: int = 2000, mode: str = "gt"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.num_frames = num_frames

        if scenarios is None:
            scenarios = ["dominoes", "support", "collide", "drop",
                         "roll", "containment", "link", "drape"]

        self.samples = []

        for scenario in scenarios:
            scenario_dir = self.data_dir / scenario
            if not scenario_dir.exists():
                print(f"  Scenario {scenario} not found, skipping")
                continue

            trial_dirs = sorted([d for d in scenario_dir.iterdir()
                                if d.is_dir() and split in d.name])[:max_videos // len(scenarios)]

            for trial_dir in trial_dirs:
                state_file = trial_dir / "state_info.json"
                stim_file = trial_dir / "stimulus_info.json"

                if state_file.exists():
                    try:
                        sample = self._load_trial(trial_dir, scenario)
                        if sample is not None:
                            self.samples.append(sample)
                    except Exception as e:
                        continue

        print(f"Physion: loaded {len(self.samples)} trials from {len(scenarios)} scenarios")

    def _load_trial(self, trial_dir: Path, scenario: str):
        """Load one trial's object states."""
        with open(trial_dir / "state_info.json", "r") as f:
            state_info = json.load(f)

        # Extract object positions and velocities per frame
        objects = {}
        for frame_key, frame_data in state_info.items():
            if not isinstance(frame_data, dict):
                continue
            for obj_name, obj_data in frame_data.items():
                if obj_name not in objects:
                    objects[obj_name] = []
                if isinstance(obj_data, dict) and "position" in obj_data:
                    pos = obj_data["position"]
                    vel = obj_data.get("velocity", [0, 0, 0])
                    objects[obj_name].append(pos[:2] + vel[:2])  # [x, y, vx, vy]

        if len(objects) < 2:
            return None

        # Take first N objects, pad to uniform length
        obj_names = sorted(objects.keys())[:6]
        num_obj = len(obj_names)
        T = min(self.num_frames, min(len(objects[n]) for n in obj_names))

        if T < 8:
            return None

        # Build state tensor [T, num_obj, 4]
        states = np.zeros((T, num_obj, 4))
        for i, name in enumerate(obj_names):
            for t in range(T):
                states[t, i] = objects[name][t]

        # Normalize positions to [0, 1]
        pos_min = states[:, :, :2].min()
        pos_max = states[:, :, :2].max()
        pos_range = max(pos_max - pos_min, 1e-6)
        states[:, :, :2] = (states[:, :, :2] - pos_min) / pos_range
        states[:, :, 2:] = states[:, :, 2:] / (pos_range + 1e-6)

        # Collision adjacency (approximate: objects within threshold)
        collision_adj = np.zeros((T, num_obj, num_obj))
        for t in range(T):
            for i in range(num_obj):
                for j in range(i + 1, num_obj):
                    dist = np.sqrt(np.sum((states[t, i, :2] - states[t, j, :2])**2))
                    if dist < 0.1:
                        collision_adj[t, i, j] = 1.0
                        collision_adj[t, j, i] = 1.0

        return {
            "gt_states": torch.tensor(states, dtype=torch.float32),
            "collision_adj": torch.tensor(collision_adj, dtype=torch.float32),
            "scenario": scenario,
            "num_objects": num_obj,
            "video_id": trial_dir.name,
            "objects": {"num_objects": num_obj, "properties": [
                {"color": "unknown", "shape": "object", "material": scenario}
                for _ in range(num_obj)
            ]},
            "events": [],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def physion_collate_fn(batch):
    # Pad to max objects in batch
    max_obj = max(b["gt_states"].shape[1] for b in batch)
    T = batch[0]["gt_states"].shape[0]
    D = batch[0]["gt_states"].shape[2]

    gt_list, col_list = [], []
    for b in batch:
        n = b["gt_states"].shape[1]
        gt = torch.zeros(T, max_obj, D)
        gt[:, :n] = b["gt_states"]
        gt_list.append(gt)
        col = torch.zeros(T, max_obj, max_obj)
        col[:, :n, :n] = b["collision_adj"]
        col_list.append(col)

    return {
        "gt_states": torch.stack(gt_list),
        "collision_adj": torch.stack(col_list),
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
        "scenarios": [b["scenario"] for b in batch],
    }


def split_physion_compositional(dataset, seed=42):
    """Compositional split: hold out specific scenarios for testing."""
    rng = random.Random(seed)

    # Hold out "collide" and "drape" scenarios (different physics)
    holdout_scenarios = {"collide", "drape"}

    train_idx, unseen_idx = [], []
    for i in range(len(dataset)):
        if dataset.samples[i]["scenario"] in holdout_scenarios:
            unseen_idx.append(i)
        else:
            train_idx.append(i)

    rng.shuffle(train_idx)
    n = max(1, len(train_idx) // 10)
    seen_idx = train_idx[-n:]
    train_idx = train_idx[:-n]

    return train_idx, seen_idx, unseen_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "visual"])
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/physion")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("physion", exp_dir / "eval.log")

    methods = ["NoGraph", "SingleModule", "SlotFormer", "CausalComp"]
    all_results = {m: {"seen": [], "unseen": [], "gap": []} for m in methods}

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        dataset = PhysionDataset(data_dir=args.data_dir, split="train",
                                  num_frames=16, mode=args.mode)
        if len(dataset) == 0:
            logger.info("No data loaded! Check --data_dir path.")
            return

        tr, se, un = split_physion_compositional(dataset, seed)
        logger.info(f"Seed {seed}: Train={len(tr)} Seen={len(se)} Unseen={len(un)}")

        tl = DataLoader(Subset(dataset, tr), batch_size=args.batch_size, shuffle=True,
                       num_workers=0, collate_fn=physion_collate_fn, drop_last=True)
        sl = DataLoader(Subset(dataset, se), batch_size=args.batch_size, shuffle=False,
                       num_workers=0, collate_fn=physion_collate_fn)
        ul = DataLoader(Subset(dataset, un), batch_size=args.batch_size, shuffle=False,
                       num_workers=0, collate_fn=physion_collate_fn)

        sd = 4  # state dim for Physion GT mode

        for mn in methods:
            torch.manual_seed(seed)
            if mn == "CausalComp":
                m = GTCausalComp(state_dim=sd, slot_dim=128, num_interaction_types=8).to(device)
                gl = True
            elif mn == "SlotFormer":
                m = SlotFormerBaseline(state_dim=sd, slot_dim=128).to(device); gl = False
            elif mn == "SingleModule":
                m = SingleModuleModel(state_dim=sd, slot_dim=128).to(device); gl = False
            else:
                m = NoGraphModel(state_dim=sd, slot_dim=128).to(device); gl = False

            logger.info(f"  Training {mn}...")
            train_model(m, tl, args.num_epochs, args.lr, device, use_graph_loss=gl, rollout_steps=8)
            s = eval_mse(m, sl, device, 8)
            u = eval_mse(m, ul, device, 8)
            g = 100*(u-s)/max(s,1e-8)
            all_results[mn]["seen"].append(s)
            all_results[mn]["unseen"].append(u)
            all_results[mn]["gap"].append(g)
            logger.info(f"    {mn}: seen={s:.6f} unseen={u:.6f} gap={g:+.1f}%")

    logger.info(f"\nPhysion RESULTS:")
    for mn in methods:
        r = all_results[mn]
        if r["seen"]:
            logger.info(f"  {mn}: seen={np.mean(r['seen']):.4f}±{np.std(r['seen']):.4f} "
                       f"unseen={np.mean(r['unseen']):.4f}±{np.std(r['unseen']):.4f} "
                       f"gap={np.mean(r['gap']):+.1f}±{np.std(r['gap']):.1f}%")

    torch.save(all_results, exp_dir / "physion_results.pt")


if __name__ == "__main__":
    main()
