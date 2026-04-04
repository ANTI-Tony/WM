#!/bin/bash
# RunPod one-click setup script.
# Usage: bash scripts/setup_runpod.sh
set -e

echo "=== Setting up CausalComp on RunPod ==="

# Install dependencies
pip install -q torch torchvision einops wandb h5py matplotlib tqdm scipy scikit-learn networkx

# Download CLEVRER (takes ~15 min on RunPod)
echo "=== Downloading CLEVRER dataset ==="
mkdir -p data/clevrer
cd data/clevrer

# Training videos (5 parts, ~1.3GB each)
for i in $(seq 0 4); do
    if [ ! -f "video_train_${i}.zip" ]; then
        echo "Downloading video_train_${i}.zip ..."
        wget -q --show-progress "http://data.csail.mit.edu/clevrer/videos/train/video_train_${i}.zip"
    fi
done

# Validation videos
if [ ! -f "video_validation.zip" ]; then
    echo "Downloading video_validation.zip ..."
    wget -q --show-progress "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"
fi

# Annotations
wget -q -O train.json "http://data.csail.mit.edu/clevrer/annotations/train.json" 2>/dev/null || true
wget -q -O validation.json "http://data.csail.mit.edu/clevrer/annotations/validation.json" 2>/dev/null || true

# Extract
echo "=== Extracting videos ==="
for f in *.zip; do
    echo "  Extracting $f ..."
    unzip -q -o "$f"
done

cd ../..

echo "=== Running smoke test ==="
python test_smoke.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start training:"
echo "  python train.py --exp_name v1_debug --num_epochs 5 --batch_size 8 --resolution 64"
echo ""
echo "Full training:"
echo "  python train.py --exp_name v1_full --wandb"
echo ""
