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


def normalize_layout(download_path: Path):
    """
    Roboflow exports as:
        raw_download/train/images, raw_download/train/labels
        raw_download/valid/images, raw_download/valid/labels
        raw_download/test/images,  raw_download/test/labels
        raw_download/data.yaml

    We move these to a clean top-level structure:
        data/train/images, data/train/labels
        data/valid/images, data/valid/labels
        data/test/images,  data/test/labels
    """
    print("[INFO] Normalizing folder layout ...")
    for split in ["train", "valid", "test"]:
        src = download_path / split
        dst = DATA_DIR / split
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"[INFO]   {src} -> {dst}")
        else:
            print(f"[WARN]   Split '{split}' not found in download, skipping.")

    # Clean up the now-empty raw_download wrapper folder (Roboflow nests
    # things one level deeper, e.g. raw_download/Pistols-1/...)
    raw_root = DATA_DIR / "raw_download"
    if raw_root.exists():
        shutil.rmtree(raw_root, ignore_errors=True)


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
    normalize_layout(download_path)
    count_images()
    print("[DONE] Dataset ready at:", DATA_DIR)


if __name__ == "__main__":
    main()
