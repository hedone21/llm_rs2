"""Parser for component gate status data."""
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GATE_FILE = _PROJECT_ROOT / "results" / "data" / "component_gates.json"


def load_gate_status():
    """Load component gate status from JSON file."""
    if not GATE_FILE.exists():
        return None
    with open(GATE_FILE) as f:
        return json.load(f)
