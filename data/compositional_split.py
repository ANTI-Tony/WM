"""
Compositional generalization split for synthetic physics data.

Design: Objects have (color, size) attributes. Interactions are collisions.
We hold out specific (color_A, size_A) × (color_B, size_B) collision pairs
from training and test on them.

Train: 60% of attribute-pair combinations
Test-seen: held-out samples of training combinations
Test-unseen: 40% of combinations NEVER seen during training

This follows the MCD (Maximum Compound Divergence) methodology from
Keysers et al., "Measuring Compositional Generalization" (ICLR 2020).
"""

from itertools import combinations
from typing import Dict, List, Tuple
import random

# Object types = (color, size)
COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta"]
SIZES = ["small", "medium", "large"]

# All possible object types
ALL_TYPES = [(c, s) for c in COLORS for s in SIZES]  # 18 types

# All possible unordered COLLISION PAIRS of object types
# This is what we split on: which type-pairs collide during training
ALL_PAIRS = list(combinations(range(len(ALL_TYPES)), 2))  # C(18,2) = 153 pairs


def create_compositional_split(
    seed: int = 42,
    train_fraction: float = 0.6,
) -> Dict:
    """Create train/test split based on collision pair types.

    Returns:
        split_info: dict with:
            - train_pairs: set of (type_i, type_j) indices seen during training
            - test_pairs: set of (type_i, type_j) indices held out
            - all_types: list of (color, size) tuples
            - type_to_idx: dict mapping (color, size) → index
    """
    rng = random.Random(seed)

    # Shuffle and split pairs
    pairs = list(ALL_PAIRS)
    rng.shuffle(pairs)

    n_train = int(len(pairs) * train_fraction)
    train_pairs = set(pairs[:n_train])
    test_pairs = set(pairs[n_train:])

    # Verify all individual types appear in BOTH splits
    # (atom distribution should match — MCD requirement)
    train_types = set()
    for i, j in train_pairs:
        train_types.add(i)
        train_types.add(j)

    test_types = set()
    for i, j in test_pairs:
        test_types.add(i)
        test_types.add(j)

    # If some types only appear in test, move a pair to train
    missing = test_types - train_types
    if missing:
        for m in missing:
            # Find a test pair containing m and move it to train
            for p in list(test_pairs):
                if m in p:
                    test_pairs.remove(p)
                    train_pairs.add(p)
                    break

    type_to_idx = {t: i for i, t in enumerate(ALL_TYPES)}

    print(f"Compositional split: {len(train_pairs)} train pairs, "
          f"{len(test_pairs)} test pairs out of {len(ALL_PAIRS)} total")
    print(f"  Train types covered: {len(train_types)}/{len(ALL_TYPES)}")

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "all_types": ALL_TYPES,
        "type_to_idx": type_to_idx,
    }


def get_object_type_idx(color: str, size: str) -> int:
    """Map (color, size) to type index."""
    for i, (c, s) in enumerate(ALL_TYPES):
        if c == color and s == size:
            return i
    return -1


def classify_video(objects: list, events: list, split_info: dict) -> str:
    """Classify a video as 'train', 'test_seen', or 'test_unseen'.

    A video belongs to 'test_unseen' if ANY collision in it involves
    a held-out type pair. Otherwise it's in the train pool.

    Args:
        objects: list of {"color": str, "size_name": str, ...}
        events: list of {"type": "collision", "objects": [i, j], ...}
        split_info: from create_compositional_split()
    Returns:
        "train" or "test_unseen"
    """
    test_pairs = split_info["test_pairs"]

    # Get type indices for each object
    obj_types = []
    for o in objects:
        idx = get_object_type_idx(o["color"], o.get("size_name", o.get("material", "")))
        obj_types.append(idx)

    # Check each collision
    for ev in events:
        if ev["type"] != "collision":
            continue
        i, j = ev["objects"]
        if i < len(obj_types) and j < len(obj_types):
            ti, tj = obj_types[i], obj_types[j]
            pair = (min(ti, tj), max(ti, tj))
            if pair in test_pairs:
                return "test_unseen"

    return "train"


if __name__ == "__main__":
    # Test the split
    split = create_compositional_split(seed=42, train_fraction=0.6)

    # Generate some videos and classify them
    from synthetic_dataset import SyntheticPhysicsDataset
    ds = SyntheticPhysicsDataset(num_videos=1000, num_frames=16, resolution=64, seed=123)

    counts = {"train": 0, "test_unseen": 0}
    for i in range(len(ds)):
        sample = ds[i]
        props = sample["objects"]["properties"]
        label = classify_video(props, sample["events"], split)
        counts[label] += 1

    total = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k}: {v} ({100*v/total:.1f}%)")
