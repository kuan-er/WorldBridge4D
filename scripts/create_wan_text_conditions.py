#!/usr/bin/env python3
"""Offline-encode the three fixed Wan UMT5 conditions with provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import torch

PROMPTS = {
    "kubric": "A rendered monocular video of multiple rigid objects moving independently in a three-dimensional scene. The camera viewpoint may change over time, and objects may become occluded and reappear.",
    "pointodyssey": "A rendered monocular video of articulated characters and objects undergoing diverse rigid and non-rigid motion in a three-dimensional scene. The camera viewpoint may change over time, and objects may become occluded and reappear.",
    "dynamic_replica": "A rendered monocular video of articulated people moving through a furnished indoor three-dimensional scene. The camera viewpoint may change over time, and people may become occluded and reappear.",
}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            value.update(data)
    return value.hexdigest()


def native_encoder(source: Path):
    if (source / "wan/modules/t5.py").exists():
        sys.path.insert(0, str(source))
        from wan.modules.t5 import T5EncoderModel
        return T5EncoderModel, "official_wan_package"
    if (source / "wan_base/modules/t5.py").exists():
        package = types.ModuleType("_worldbridge_wan_native")
        package.__path__ = [str(source / "wan_base")]
        modules = types.ModuleType("_worldbridge_wan_native.modules")
        modules.__path__ = [str(source / "wan_base" / "modules")]
        sys.modules[package.__name__] = package
        sys.modules[modules.__name__] = modules
        return importlib.import_module("_worldbridge_wan_native.modules.t5").T5EncoderModel, "vendored_wan_base_package"
    raise FileNotFoundError(f"not a Wan2.1 source package: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-source", default=os.getenv("WAN_SOURCE_ROOT"), required=os.getenv("WAN_SOURCE_ROOT") is None)
    parser.add_argument("--checkpoint-dir", default="/data/WorldBridge4D/Wan2.1-T2V-1.3B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source = Path(args.wan_source).resolve()
    checkpoint = Path(args.checkpoint_dir).resolve()
    output = Path(args.output_dir).resolve()
    T5EncoderModel, layout = native_encoder(source)
    t5_checkpoint = checkpoint / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = checkpoint / "google/umt5-xxl"
    if not t5_checkpoint.is_file() or not tokenizer.is_dir():
        raise FileNotFoundError("native UMT5 checkpoint/tokenizer is incomplete")
    device = torch.device(args.device)
    encoder = T5EncoderModel(
        text_len=512, dtype=torch.bfloat16, device=device,
        checkpoint_path=str(t5_checkpoint), tokenizer_path=str(tokenizer),
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, prompt in PROMPTS.items():
        with torch.inference_mode():
            unpadded = encoder([prompt], device)[0].detach()
        if unpadded.ndim != 2 or unpadded.shape[1] != 4096 or unpadded.shape[0] > 512:
            raise RuntimeError(f"{name}: unexpected UMT5 output {tuple(unpadded.shape)}")
        condition = torch.cat((unpadded, unpadded.new_zeros(512 - len(unpadded), 4096)))[None].cpu()
        metadata = {
            "dataset": name, "prompt": prompt, "unpadded_tokens": int(len(unpadded)),
            "source_commit": commit, "source_layout": layout, "source_root": str(source),
            "t5_checkpoint": str(t5_checkpoint), "t5_checkpoint_sha256": sha256(t5_checkpoint),
            "tokenizer": str(tokenizer),
        }
        path = output / f"{name}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"encoder_hidden_states": condition, "metadata": metadata}, temporary)
        temporary.replace(path)
        summary[name] = {"path": str(path), "shape": list(condition.shape), **metadata}
    (output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("WAN_THREE_TEXT_CONDITIONS_READY", flush=True)


if __name__ == "__main__":
    main()
