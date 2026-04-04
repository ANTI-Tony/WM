"""
CLEVRER dataset loader for CausalComp.

Loads video frames and annotations from the CLEVRER dataset.
Supports compositional train/test splits (CLEVRER-Comp).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CLEVRERDataset(Dataset):
    """CLEVRER video dataset.

    Each sample is a short video clip of T frames with object annotations.

    Directory structure expected:
        data_dir/
            video_train/      or  video_validation/
                video_00000/
                    frame_00000.png
                    frame_00001.png
                    ...
            train.json
            validation.json
    """

    def __init__(self, data_dir: str, split: str = "train",
                 num_frames: int = 16, frame_skip: int = 4,
                 resolution: int = 128, comp_split: Optional[Dict] = None):
        """
        Args:
            data_dir: path to CLEVRER data
            split: "train" or "validation"
            num_frames: number of frames to sample per clip
            frame_skip: sample every N-th frame
            resolution: resize frames to this size
            comp_split: optional compositional split config
                        {"video_ids": [list of video IDs to include]}
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_skip = frame_skip

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),  # [0, 1], [C, H, W]
        ])

        # Load annotations
        ann_file = self.data_dir / f"{split}.json"
        if ann_file.exists():
            with open(ann_file, "r") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []
            print(f"Warning: {ann_file} not found. Running in frame-only mode.")

        # Determine video directory
        if split == "train":
            self.video_dir = self.data_dir / "video_train"
        else:
            self.video_dir = self.data_dir / "video_validation"

        # List available videos
        if self.video_dir.exists():
            self.video_ids = sorted([
                d.name for d in self.video_dir.iterdir() if d.is_dir()
            ])
        else:
            # Fallback: generate dummy IDs from annotations
            self.video_ids = [
                f"video_{ann['scene_index']:05d}"
                for ann in self.annotations
            ] if self.annotations else []

        # Apply compositional split filter
        if comp_split is not None and "video_ids" in comp_split:
            allowed = set(comp_split["video_ids"])
            self.video_ids = [v for v in self.video_ids if v in allowed]

        # Build annotation index
        self.ann_index = {}
        for ann in self.annotations:
            vid_name = f"video_{ann['scene_index']:05d}"
            self.ann_index[vid_name] = ann

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vid_name = self.video_ids[idx]
        vid_dir = self.video_dir / vid_name

        # Load frames
        frames = self._load_frames(vid_dir)

        # Load annotations if available
        ann = self.ann_index.get(vid_name, {})

        # Extract object info from annotations
        objects = self._extract_objects(ann)

        # Extract causal events (ground truth for evaluation)
        events = self._extract_events(ann)

        return {
            "video": frames,          # [T, 3, H, W]
            "video_id": vid_name,
            "objects": objects,        # dict of object properties
            "events": events,          # list of causal events
        }

    def _load_frames(self, vid_dir: Path) -> torch.Tensor:
        """Load and sample T frames from a video directory."""
        # List all frames
        frame_files = sorted(vid_dir.glob("frame_*.png"))
        if not frame_files:
            # Try alternative naming
            frame_files = sorted(vid_dir.glob("*.png"))

        if not frame_files:
            # Return dummy frames for debugging
            return torch.randn(self.num_frames, 3, 128, 128)

        total_frames = len(frame_files)

        # Sample frames with frame_skip
        max_start = total_frames - self.num_frames * self.frame_skip
        start = np.random.randint(0, max(1, max_start))
        indices = [
            min(start + i * self.frame_skip, total_frames - 1)
            for i in range(self.num_frames)
        ]

        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert("RGB")
            frames.append(self.transform(img))

        return torch.stack(frames)  # [T, 3, H, W]

    def _extract_objects(self, ann: dict) -> dict:
        """Extract object properties from CLEVRER annotations."""
        if not ann or "ground_truth" not in ann:
            return {"num_objects": 0, "properties": []}

        gt = ann["ground_truth"]
        objects = []
        if "objects" in gt:
            for obj in gt["objects"]:
                objects.append({
                    "color": obj.get("color", ""),
                    "shape": obj.get("shape", ""),
                    "material": obj.get("material", ""),
                })

        return {
            "num_objects": len(objects),
            "properties": objects,
        }

    def _extract_events(self, ann: dict) -> List[dict]:
        """Extract causal events (collisions, etc.) from annotations."""
        if not ann or "ground_truth" not in ann:
            return []

        gt = ann["ground_truth"]
        events = []
        if "collisions" in gt:
            for col in gt["collisions"]:
                events.append({
                    "type": "collision",
                    "frame": col.get("frame", 0),
                    "objects": col.get("objects", []),
                })
        return events


def clevrer_collate_fn(batch: List[dict]) -> dict:
    """Custom collate function for CLEVRER dataset."""
    videos = torch.stack([b["video"] for b in batch])
    return {
        "video": videos,                          # [B, T, 3, H, W]
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
    }
