"""
Phase 4 - Dataset Verification
================================
Sanity-checks the downloaded dataset before we spend GPU hours training on it.

Checks performed:
    1. Every image has a matching .txt label file (and vice versa).
    2. Every label file is well-formed (5 floats/line, class_id, x, y, w, h).
    3. Bounding box coordinates are within [0, 1] (normalized YOLO format).
    4. Images are not corrupt / unreadable.
    5. Class distribution per split (train/valid/test) is reported.
    6. Warns on empty label files (background-only images) so you know
       they exist before training, since they are valid but worth tracking.

Usage:
    python scripts/verify_dataset.py
"""

import sys
from pathlib import Path
from collections import Counter

import cv2

from utils import load_class_names, project_root

PROJECT_ROOT = project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
SPLITS = ["train", "valid", "test"]


def verify_split(split: str):
    img_dir = DATA_DIR / split / "images"
    lbl_dir = DATA_DIR / split / "labels"

    report = {
        "split": split,
        "num_images": 0,
        "num_labels": 0,
        "missing_labels": [],
        "missing_images": [],
        "corrupt_images": [],
        "malformed_labels": [],
        "empty_labels": 0,
        "class_counts": Counter(),
        "bbox_out_of_range": [],
    }

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[WARN] Split '{split}' missing images/ or labels/ folder — skipping.")
        return report

    image_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    label_files = sorted([p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"])

    report["num_images"] = len(image_files)
    report["num_labels"] = len(label_files)

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    report["missing_labels"] = sorted(image_stems - label_stems)
    report["missing_images"] = sorted(label_stems - image_stems)

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            report["corrupt_images"].append(img_path.name)

    for lbl_path in label_files:
        lines = lbl_path.read_text().strip().splitlines()
        if len(lines) == 0:
            report["empty_labels"] += 1
            continue
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                report["malformed_labels"].append(f"{lbl_path.name}:line{i+1}")
                continue
            try:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                report["malformed_labels"].append(f"{lbl_path.name}:line{i+1}")
                continue

            report["class_counts"][cls_id] += 1

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                report["bbox_out_of_range"].append(f"{lbl_path.name}:line{i+1}")

    return report


def print_report(report):
    print(f"\n{'='*60}")
    print(f"SPLIT: {report['split'].upper()}")
    print(f"{'='*60}")
    print(f"  Images        : {report['num_images']}")
    print(f"  Labels        : {report['num_labels']}")
    print(f"  Empty labels  : {report['empty_labels']} (background-only images)")
    print(f"  Class counts  : {dict(report['class_counts'])}")

    issues = 0
    if report["missing_labels"]:
        issues += len(report["missing_labels"])
        print(f"  [ISSUE] {len(report['missing_labels'])} images missing labels, e.g. {report['missing_labels'][:5]}")
    if report["missing_images"]:
        issues += len(report["missing_images"])
        print(f"  [ISSUE] {len(report['missing_images'])} labels missing images, e.g. {report['missing_images'][:5]}")
    if report["corrupt_images"]:
        issues += len(report["corrupt_images"])
        print(f"  [ISSUE] {len(report['corrupt_images'])} corrupt/unreadable images, e.g. {report['corrupt_images'][:5]}")
    if report["malformed_labels"]:
        issues += len(report["malformed_labels"])
        print(f"  [ISSUE] {len(report['malformed_labels'])} malformed label lines, e.g. {report['malformed_labels'][:5]}")
    if report["bbox_out_of_range"]:
        issues += len(report["bbox_out_of_range"])
        print(f"  [ISSUE] {len(report['bbox_out_of_range'])} bbox coords out of [0,1] range, e.g. {report['bbox_out_of_range'][:5]}")

    if report.get("unknown_classes"):
        issues += len(report["unknown_classes"])
        print(f"  [ISSUE] class IDs found in labels but not declared in configs/data.yaml: {report['unknown_classes']}")

    if issues == 0:
        print("  STATUS: CLEAN ✓")
    else:
        print(f"  STATUS: {issues} ISSUE(S) FOUND ✗")

    return issues


def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] {DATA_DIR} does not exist. Run scripts/download_dataset.py first.")
        sys.exit(1)

    declared_classes = load_class_names(DATA_YAML)
    print(f"[INFO] Classes declared in configs/data.yaml: {declared_classes}")

    total_issues = 0
    for split in SPLITS:
        report = verify_split(split)
        report["unknown_classes"] = sorted(set(report["class_counts"]) - set(declared_classes))
        total_issues += print_report(report)

    print(f"\n{'='*60}")
    if total_issues == 0:
        print("ALL SPLITS CLEAN — dataset is ready for training.")
    else:
        print(f"TOTAL ISSUES ACROSS ALL SPLITS: {total_issues}")
        print("Review the issues above before proceeding to training.")
        print("Minor issues (a few missing/corrupt files) can usually be")
        print("deleted safely; widespread issues indicate a bad download.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
