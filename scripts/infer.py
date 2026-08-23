#!/usr/bin/env python3
"""WorldBridge4D inference entrypoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldbridge.evaluation.inference import main


if __name__ == "__main__":
    main()
