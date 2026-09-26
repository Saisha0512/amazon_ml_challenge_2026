"""Run the complete local matching pipeline from the repository root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.pipeline import main

if __name__ == "__main__":
    main()
