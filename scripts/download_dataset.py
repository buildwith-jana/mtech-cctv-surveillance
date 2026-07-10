"""
Phase 3 - Dataset Download
===========================
Downloads the Roboflow "Pistols Dataset" in YOLO format using the
Roboflow Python SDK, and normalizes the folder layout so it matches
what configs/data.yaml expects.

Usage:
    python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

Get a free API key at: https://app.roboflow.com/settings/api
(Sign up -> Settings -> Roboflow API -> Private API Key)

Why this dataset (for dissertation justification):
    The Roboflow "Pistols Dataset" (workspace: joseph-nelson, project: pistols)
    is a publicly available, pre-annotated dataset of ~3000 images with
    bounding-box labels for handguns, sourced from varied real-world contexts
    (news images, CCTV-like angles, indoor/outdoor scenes). Using a
    pre-annotated public dataset removes annotation-quality as a confound
    and lets Module 1 focus on detection pipeline correctness and CCTV
    generalization, which is the actual research contribution.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def download_pistols_dataset(api_key: str, workspace: str, project: str, version: int):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    print(f"[INFO] Connecting to Roboflow workspace='{workspace}' project='{project}' v{version}")
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    proj = ws.project(project)
    dataset_version = proj.version(version)

    print("[INFO] Downloading dataset in YOLOv11 (Ultralytics) format ...")
    dataset = dataset_version.download("yolov11", location=str(DATA_DIR / "raw_download"))
    print(f"[INFO] Downloaded to: {dataset.location}")
    return Path(dataset.location)


def find_split_directories(download_path: Path) -> dict:
    """
    Recursively locate the train/valid/test split directories anywhere under
    download_path, regardless of nesting depth. Roboflow's export layout is
    not guaranteed to be flat -- it sometimes wraps the splits in an extra
    project-name folder (e.g. download_path/Pistols-1/train/images instead
    of download_path/train/images), and split naming can vary ("valid" vs
    "val"). We identify a split directory by finding any "images" folder and
    checking whether its parent directory name matches a known alias.

    Returns: {"train": Path|None, "valid": Path|None, "test": Path|None}
    where each Path (if found) is the split directory itself (the one that
    directly contains "images" and "labels" subfolders).
    """
    aliases = {
        "train": {"train", "training"},
        "valid": {"valid", "val", "validation"},
        "test": {"test", "testing"},
    }
    found = {"train": None, "valid": None, "test": None}

    for images_dir in download_path.rglob("images"):
        if not images_dir.is_dir():
            continue
        split_dir = images_dir.parent
        dir_name = split_dir.name.lower()
        for canonical, alias_set in aliases.items():
            if dir_name in alias_set and found[canonical] is None:
                found[canonical] = split_dir
                print(f"[INFO]   Located '{canonical}' split at: {split_dir}")

    return found


def normalize_layout(download_path: Path):
    """
    Moves whatever split directories were located (train/valid/test, each
    containing images/ and labels/) to a clean top-level structure:
        data/train/images, data/train/labels
        data/valid/images, data/valid/labels
        data/test/images,  data/test/labels

    Works regardless of how deeply Roboflow nested the splits under
    download_path, and regardless of "valid" vs "val" naming.
    """
    print("[INFO] Searching recursively for train/valid/test split directories ...")
    split_dirs = find_split_directories(download_path)

    for split, src in split_dirs.items():
        dst = DATA_DIR / split
        if src is not None:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"[INFO]   {src} -> {dst}")
        else:
            print(f"[WARN]   Split '{split}' not found anywhere under {download_path}, skipping.")

    # Clean up whatever wrapper folder(s) Roboflow created during extraction,
    # now that the split directories have been moved out of them.
    if download_path.exists():
        shutil.rmtree(download_path, ignore_errors=True)


def count_images():
    print("\n[INFO] Dataset summary:")
    total = 0
    for split in ["train", "valid", "test"]:
        img_dir = DATA_DIR / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*.*")))
            total += n
            print(f"    {split:6s}: {n} images")
        else:
            print(f"    {split:6s}: MISSING")
    print(f"    TOTAL : {total} images\n")
    if total == 0:
        print("[ERROR] No images found after download. Check your API key and dataset access.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download the Roboflow Pistols Dataset")
    parser.add_argument("--api-key", required=True, help="Your Roboflow private API key")
    parser.add_argument("--workspace", default="joseph-nelson", help="Roboflow workspace slug")
    parser.add_argument("--project", default="pistols", help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

   download_path = download_pistols_dataset(
    api_key=args.api_key,
    workspace=args.workspace,
    project=args.project,
    version=args.version,
)

print("DOWNLOAD PATH:", download_path)

import os
for root, dirs, files in os.walk(download_path):
    print(root)
    if len(files):
        print(" Files:", files[:5])

exit()
    print("[DONE] Dataset ready at:", DATA_DIR)


if __name__ == "__main__":
    main()
