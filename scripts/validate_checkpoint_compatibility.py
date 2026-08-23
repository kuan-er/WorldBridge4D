#!/usr/bin/env python3
"""Strict-load a production checkpoint into the refactored package layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.trainer.compatibility import validate_checkpoint_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = validate_checkpoint_model(args.config, args.checkpoint, args.device)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("CHECKPOINT_LAYOUT_COMPATIBILITY_OK", flush=True)


if __name__ == "__main__":
    main()
