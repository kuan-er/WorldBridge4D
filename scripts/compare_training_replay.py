#!/usr/bin/env python3
"""Compare two full training checkpoints recursively and exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.trainer.compatibility import compare_training_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = compare_training_checkpoints(args.reference, args.candidate)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("TRAINING_REPLAY_EXACT_MATCH_OK", flush=True)


if __name__ == "__main__":
    main()
