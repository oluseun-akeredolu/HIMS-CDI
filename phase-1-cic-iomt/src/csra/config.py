from pathlib import Path

PAPER3_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PAPER3_DIR / "data"
ARTIFACTS_DIR = PAPER3_DIR / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
