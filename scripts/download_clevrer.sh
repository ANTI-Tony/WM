#!/bin/bash
# Download CLEVRER dataset.
# Run this on RunPod or wherever you have storage.
#
# CLEVRER: http://clevrer.csail.mit.edu/
# Videos: ~7GB total (train + val + test)
# Annotations: ~100MB

set -e

DATA_DIR="${1:-./data/clevrer}"
mkdir -p "$DATA_DIR"

echo "=== Downloading CLEVRER videos ==="

# Training videos (5 parts)
for i in $(seq 0 4); do
    URL="http://data.csail.mit.edu/clevrer/videos/train/video_train_${i}.zip"
    echo "Downloading video_train_${i}.zip ..."
    wget -q --show-progress -P "$DATA_DIR" "$URL"
done

# Validation videos
wget -q --show-progress -P "$DATA_DIR" \
    "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"

echo "=== Downloading CLEVRER annotations ==="

# Train/val annotations
wget -q --show-progress -O "$DATA_DIR/train.json" \
    "http://data.csail.mit.edu/clevrer/annotations/train.json"
wget -q --show-progress -O "$DATA_DIR/validation.json" \
    "http://data.csail.mit.edu/clevrer/annotations/validation.json"

echo "=== Extracting ==="
cd "$DATA_DIR"
for f in *.zip; do
    echo "Extracting $f ..."
    unzip -q -o "$f"
done

echo "=== Done ==="
echo "Data directory: $DATA_DIR"
ls -lh "$DATA_DIR"
