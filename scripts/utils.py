"""
Shared utilities for the weapon-detection pipeline.

Design principle: NOTHING in this project hardcodes a specific class name
(e.g. "pistol"). Every script reads class identities from configs/data.yaml
at runtime. Adding a new class (e.g. "knife") means editing ONLY:
    1. configs/data.yaml  (add the class + reserved index)
    2. the dataset used for training
No script in scripts/ needs to change.
"""

from pathlib import Path
import yaml


def load_class_names(data_yaml_path: Path) -> dict:
    """
    Load {class_id: class_name} from a YOLO data.yaml file.
    Supports both the dict form (0: pistol) and list form ([pistol, knife]).
    """
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names")
    if names is None:
        raise ValueError(f"'names' key missing in {data_yaml_path}")

    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    else:
        raise ValueError(f"Unrecognized 'names' format in {data_yaml_path}: {type(names)}")


def project_root() -> Path:
    """Absolute path to the project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent
