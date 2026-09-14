#!/usr/bin/env python3
"""WorldBridge4D fixed-budget 122-query validation evaluation entrypoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldbridge.evaluation.benchmark import main


if __name__ == "__main__":
    main()
