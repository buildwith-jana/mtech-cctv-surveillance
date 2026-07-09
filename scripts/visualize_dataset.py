"""
Phase 5 - Dataset Visualization
=================================
Renders a grid of sample training images with their YOLO bounding boxes
drawn on top, and a bar chart of class distribution across splits.
Outputs are saved to outputs/metrics/ so they can be dropped straight
into the dissertation's "Dataset" section as figures.

Usage:
    python scripts/visualize_dataset.py --split train --num-samples 9
"""

import argparse
import random
from pathlib import Path
from collections import Counter

import cv2
import matplotlib.pyplot as plt

from utils import load_class_names, project_root

PROJECT_ROOT = project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
CLASS_NAMES = load_class_names(DATA_YAML)  # e.g. {0: "pistol"}; extends automatically when data.yaml grows


def load_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x, y, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.append((cls_id, x, y, w, h))
    return boxes


def draw_boxes(img, boxes):
    h_img, w_img = img.shape[:2]
    for cls_id, x, y, w, h in boxes:
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.putText(img, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    return img


def visualize_samples(split: str, num_samples: int, seed: int = 42):
    img_dir = DATA_DIR / split / "images"
    lbl_dir = DATA_DIR / split / "labels"

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not image_files:
        print(f"[ERROR] No images found in {img_dir}. Run download_dataset.py first.")
        return

    random.seed(seed)
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    cols = 3
    rows = (len(sample_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for ax, img_path in zip(axes, sample_files):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = load_yolo_labels(lbl_dir / f"{img_path.stem}.txt")
        img = draw_boxes(img, boxes)
        ax.imshow(img)
        ax.set_title(f"{img_path.name} ({len(boxes)} box{'es' if len(boxes)!=1 else ''})", fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample_files):]:
        ax.axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"sample_annotations_{split}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved sample grid to {out_path}")


def plot_class_distribution():
    counts_by_split = {}
    for split in ["train", "valid", "test"]:
        lbl_dir = DATA_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        counter = Counter()
        for lbl_path in lbl_dir.glob("*.txt"):
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    counter[int(parts[0])] += 1
        counts_by_split[split] = counter

    fig, ax = plt.subplots(figsize=(8, 5))
    splits = list(counts_by_split.keys())
    class_ids = sorted(CLASS_NAMES.keys())
    bar_width = 0.8 / max(len(class_ids), 1)

    for i, cls_id in enumerate(class_ids):
        values = [counts_by_split[s].get(cls_id, 0) for s in splits]
        positions = [j + i * bar_width for j in range(len(splits))]
        ax.bar(positions, values, width=bar_width, label=CLASS_NAMES[cls_id])

    ax.set_xticks([j + bar_width * (len(class_ids) - 1) / 2 for j in range(len(splits))])
    ax.set_xticklabels(splits)
    ax.set_ylabel("Instance count")
    ax.set_title("Class Distribution per Split")
    ax.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved class distribution chart to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples and class distribution")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--num-samples", type=int, default=9)
    args = parser.parse_args()

    visualize_samples(args.split, args.num_samples)
    plot_class_distribution()


if __name__ == "__main__":
    main()
