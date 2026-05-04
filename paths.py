from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
PROJECT_NAME = PROJECT_ROOT.name

_DROPBOX_DATA = Path.home() / "Dropbox" / "Work" / "research_projects" / "archive" / PROJECT_NAME

def _resolve(name: str) -> Path:
    cloud = _DROPBOX_DATA / name
    return cloud if cloud.exists() else PROJECT_ROOT / name

DATA_DIR = _resolve("data")
OUTPUTS_DIR = _resolve("outputs")
NOTES_DIR = _DROPBOX_DATA / "notes"
REFERENCES_DIR = _resolve("references")

