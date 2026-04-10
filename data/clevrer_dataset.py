"""
CLEVRER dataset loader for CausalComp.

Loads video frames and annotations from the CLEVRER dataset.
Supports compositional train/test splits (CLEVRER-Comp).

Actual CLEVRER directory structure:
    data_dir/
        video_10000-11000/
            video_10000.mp4
            video_10001.mp4
            ...
        video_11000-12000/
            ...
        validation.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.io as vio


class CLEVRERDataset(Dataset):
    """CLEVRER video dataset.

    Each sample is a short video clip of T frames with object annotations.
    Reads .mp4 files directly via torchvision.io.
    """

    def __init__(self, data_dir: str, split: str = "train",
                 num_frames: int = 16, frame_skip: int = 4,
                 resolution: int = 128, comp_split: Optional[Dict] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.resolution = resolution

        # Image transform (applied per frame after reading)
        self.resize = transforms.Resize((resolution, resolution), antialias=True)

        # Load annotations
        ann_file = self.data_dir / f"{split}.json"
        if ann_file.exists() and ann_file.stat().st_size > 0:
            try:
                with open(ann_file, "r") as f:
                    self.annotations = json.load(f)
                print(f"Loaded {len(self.annotations)} annotations from {ann_file}")
            except json.JSONDecodeError:
                self.annotations = []
                print(f"Warning: {ann_file} is invalid JSON. Running in frame-only mode.")
        else:
            self.annotations = []
            print(f"Warning: {ann_file} not found or empty. Running in frame-only mode.")

        # Scan for all .mp4 files across range directories
        self.video_paths = self._scan_videos()
        print(f"Found {len(self.video_paths)} videos for split '{split}'")

        # Apply compositional split filter
        if comp_split is not None and "video_ids" in comp_split:
            allowed = set(comp_split["video_ids"])
            self.video_paths = [
                p for p in self.video_paths if p.stem in allowed
            ]

        # Build annotation index (keyed by scene_index)
        self.ann_index = {}
        for ann in self.annotations:
            scene_idx = ann.get("scene_index", None)
            if scene_idx is not None:
                self.ann_index[scene_idx] = ann

    def _scan_videos(self) -> List[Path]:
        """Scan data_dir for all .mp4 files in range subdirectories."""
        mp4_files = []

        # CLEVRER stores videos in range dirs: video_10000-11000/, video_0-1000/, etc.
        for subdir in sorted(self.data_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("video_"):
                mp4s = sorted(subdir.glob("video_*.mp4"))
                mp4_files.extend(mp4s)

        # Also check for mp4s directly in data_dir
        mp4_files.extend(sorted(self.data_dir.glob("video_*.mp4")))

        return mp4_files

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mp4_path = self.video_paths[idx]
        vid_name = mp4_path.stem  # e.g. "video_10000"

        # Extract scene index from filename
        scene_idx = int(vid_name.split("_")[1])

        # Load video frames
        frames = self._load_video(mp4_path)

        # Load annotations if available
        ann = self.ann_index.get(scene_idx, {})
        objects = self._extract_objects(ann)
        events = self._extract_events(ann)

        return {
            "video": frames,          # [T, 3, H, W]
            "video_id": vid_name,
            "objects": objects,
            "events": events,
        }

    def _load_video(self, mp4_path: Path) -> torch.Tensor:
        """Load and sample T frames from an mp4 file."""
        try:
            # Read video: returns (T, H, W, C) uint8
            video, _, info = vio.read_video(str(mp4_path), pts_unit="sec")
        except Exception as e:
            print(f"Warning: failed to read {mp4_path}: {e}")
            return torch.zeros(self.num_frames, 3, self.resolution, self.resolution)

        total_frames = video.shape[0]

        # Sample frames with frame_skip
        needed = self.num_frames * self.frame_skip
        max_start = max(0, total_frames - needed)
        start = np.random.randint(0, max(1, max_start + 1))
        indices = [
            min(start + i * self.frame_skip, total_frames - 1)
            for i in range(self.num_frames)
        ]

        # Select frames and convert: [T, H, W, C] uint8 → [T, C, H, W] float32
        frames = video[indices]                         # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float()    # [T, C, H, W]
        frames = frames / 255.0                         # normalize to [0, 1]

        # Resize
        frames = self.resize(frames)  # [T, C, H_new, W_new]

        return frames

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
