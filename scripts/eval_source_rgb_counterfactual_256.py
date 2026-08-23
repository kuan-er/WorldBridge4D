#!/usr/bin/env python3
"""Thin CLI for source-RGB counterfactual evaluation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.evaluation.counterfactual import main

if __name__ == "__main__":
    main()
