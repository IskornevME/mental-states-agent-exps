from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
ALFWORLD_DATA_DIR = DATA_DIR / "alfworld"
SCIWORLD_DATA_DIR = DATA_DIR / "sciworld"
SCIENCEWORLD_JAR = REPO_ROOT / "envs" / "scienceworld" / "scienceworld.jar"
