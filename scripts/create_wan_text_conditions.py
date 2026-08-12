#!/usr/bin/env python3
"""Offline-encode the three fixed Wan UMT5 conditions with provenance."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import torch

TASK_INSTRUCTION = "Estimate dense three-dimensional point trajectories over time from this monocular video."
PROMPTS = {
    "kubric": f"{TASK_INSTRUCTION} The video shows multiple rigid objects moving independently in a rendered three-dimensional scene. The camera viewpoint may change over time, and objects may become occluded and reappear.",
    "pointodyssey": f"{TASK_INSTRUCTION} The video shows articulated characters and objects undergoing diverse rigid and non-rigid motion in a rendered three-dimensional scene. The camera viewpoint may change over time, and objects may become occluded and reappear.",
    "dynamic_replica": f"{TASK_INSTRUCTION} The video shows articulated people moving through a rendered furnished indoor three-dimensional scene. The camera viewpoint may change over time, and people may become occluded and reappear.",
}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            value.update(data)
    return value.hexdigest()


def _load_t5_without_package_init(package_root: Path, module_prefix: str):
    """Load only Wan's relative-import-compatible modules, not inference init."""
    package = types.ModuleType(module_prefix)
    package.__path__ = [str(package_root)]
    modules = types.ModuleType(f"{module_prefix}.modules")
    modules.__path__ = [str(package_root / "modules")]
    sys.modules[package.__name__] = package
    sys.modules[modules.__name__] = modules
    return importlib.import_module(f"{module_prefix}.modules.t5").T5EncoderModel


def native_encoder(source: Path):
    if (source / "wan/modules/t5.py").exists():
        return _load_t5_without_package_init(
            source / "wan", "_worldbridge_wan_official"
        ), "official_wan_package"
    if (source / "wan_base/modules/t5.py").exists():
        return _load_t5_without_package_init(
            source / "wan_base", "_worldbridge_wan_native"
        ), "vendored_wan_base_package"
    raise FileNotFoundError(f"not a Wan2.1 source package: {source}")


def completed_cache(output: Path) -> dict | None:
    """Return a verified complete cache, never a partial concurrent write."""
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        summary = json.loads(metadata_path.read_text())
        for name, prompt in PROMPTS.items():
            record = summary[name]
            path = output / f"{name}.pt"
            if (record.get("dataset") != name or record.get("prompt") != prompt
                    or record.get("shape") != [1, 512, 4096]
                    or not path.is_file() or record.get("condition_sha256") != sha256(path)):
                return None
        return summary
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


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
    output.mkdir(parents=True, exist_ok=True)
    # Several launchers may notice the same absent cache at once. Serialize the
    # expensive UMT5 load/encode and let later processes reuse verified output.
    lock_handle = (output / ".create.lock").open("a+b")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    if cached := completed_cache(output):
        print(json.dumps(cached, indent=2))
        print("WAN_THREE_TEXT_CONDITIONS_READY", flush=True)
        return
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
    summary = {}
    names = list(PROMPTS)
    with torch.inference_mode():
        encoded = encoder([PROMPTS[name] for name in names], device)
    if len(encoded) != len(names):
        raise RuntimeError(f"native UMT5 returned {len(encoded)}/{len(names)} conditions")
    for name, unpadded in zip(names, encoded):
        prompt = PROMPTS[name]
        unpadded = unpadded.detach()
        if unpadded.ndim != 2 or unpadded.shape[1] != 4096 or unpadded.shape[0] > 512:
            raise RuntimeError(f"{name}: unexpected UMT5 output {tuple(unpadded.shape)}")
        condition = torch.cat((unpadded, unpadded.new_zeros(512 - len(unpadded), 4096)))[None].cpu()
        metadata = {
            "dataset": name, "prompt": prompt, "unpadded_tokens": int(len(unpadded)),
            "source_commit": commit, "source_layout": layout, "source_root": str(source),
            "t5_checkpoint": str(t5_checkpoint), "t5_checkpoint_sha256": sha256(t5_checkpoint),
            "tokenizer": str(tokenizer), "shape": list(condition.shape),
        }
        path = output / f"{name}.pt"
        temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
        torch.save({"encoder_hidden_states": condition, "metadata": metadata}, temporary)
        temporary.replace(path)
        summary[name] = {"path": str(path), **metadata, "condition_sha256": sha256(path)}
    metadata_path = output / "metadata.json"
    metadata_temporary = output / f"metadata.json.{os.getpid()}.tmp"
    metadata_temporary.write_text(json.dumps(summary, indent=2) + "\n")
    metadata_temporary.replace(metadata_path)
    print(json.dumps(summary, indent=2))
    print("WAN_THREE_TEXT_CONDITIONS_READY", flush=True)


if __name__ == "__main__":
    main()
