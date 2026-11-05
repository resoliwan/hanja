"""Repository paths shared by handwriting preparation, training and evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "resources/handwriting/datasets"
MODELS = ROOT / "resources/handwriting/models"
RESULTS = ROOT / "experiments/handwriting/results"
TOOLS = ROOT / "tools/handwriting"
APP_ASSETS = ROOT / "apps/dic/assets"


def source_path(recorded_path):
    """Resolve source metadata recorded before the dataset directory moved."""
    path = Path(recorded_path)
    if path.parts and path.parts[0] == "data":
        return DATASETS.joinpath(*path.parts[1:])
    return ROOT / path
