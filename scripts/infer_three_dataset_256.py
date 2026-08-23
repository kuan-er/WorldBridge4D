#!/usr/bin/env python3
"""Thin CLI for WorldBridge4D inference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.evaluation.inference import main

if __name__ == "__main__":
    main()
