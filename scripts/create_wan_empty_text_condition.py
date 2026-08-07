#!/usr/bin/env python3
"""Create WAN's native empty-UMT5 condition once, outside the repository."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys
import types

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-source", default=os.getenv("WAN_SOURCE_ROOT"),
                    help="Checkout of https://github.com/Wan-Video/Wan2.1")
    ap.add_argument("--checkpoint-dir", default="/dataset/Wan2.1-T2V-1.3B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.wan_source:
        raise ValueError("--wan-source (or WAN_SOURCE_ROOT) is required; native WAN code is not guessed or downloaded")
    source = pathlib.Path(args.wan_source).resolve()
    if (source / "wan/modules/t5.py").exists():
        sys.path.insert(0, str(source))
        from wan.modules.t5 import T5EncoderModel
        source_layout = "official_wan_package"
    elif (source / "wan_base/modules/t5.py").exists():
        # This server already has an exact vendored Wan2.1 source package in
        # LaCT. Load only its modules package so wan_base.__init__ cannot pull
        # unrelated inference wrappers or optional distributed dependencies.
        package = types.ModuleType("_h004_wan_native")
        package.__path__ = [str(source / "wan_base")]
        modules = types.ModuleType("_h004_wan_native.modules")
        modules.__path__ = [str(source / "wan_base" / "modules")]
        sys.modules[package.__name__] = package
        sys.modules[modules.__name__] = modules
        T5EncoderModel = importlib.import_module("_h004_wan_native.modules.t5").T5EncoderModel
        source_layout = "vendored_wan_base_package"
    else:
        raise FileNotFoundError(f"not a Wan2.1 source package: {source}")

    checkpoint = pathlib.Path(args.checkpoint_dir)
    device = torch.device(args.device)
    encoder = T5EncoderModel(
        text_len=512, dtype=torch.bfloat16, device=device,
        checkpoint_path=str(checkpoint / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(checkpoint / "google/umt5-xxl"),
    )
    with torch.inference_mode():
        unpadded = encoder([""], device)[0].detach()
    if unpadded.ndim != 2 or unpadded.shape[1] != 4096 or unpadded.shape[0] > 512:
        raise RuntimeError(f"native empty condition has unexpected shape {tuple(unpadded.shape)}")
    condition = torch.cat([unpadded, unpadded.new_zeros(512 - unpadded.shape[0], 4096)], dim=0)[None].cpu()
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True,
                                                stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unavailable"
    payload = {
        "encoder_hidden_states": condition,
        "metadata": {"prompt": "", "source_commit": source_commit, "source_layout": source_layout,
                     "source_root": str(source), "text_len": 512,
                     "unpadded_tokens": int(unpadded.shape[0]),
                     "t5_checkpoint": str((checkpoint / "models_t5_umt5-xxl-enc-bf16.pth").resolve()),
                     "tokenizer": str((checkpoint / "google/umt5-xxl").resolve())},
    }
    output = pathlib.Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp"); torch.save(payload, tmp); tmp.replace(output)
    print(json.dumps({"output": str(output), "shape": list(condition.shape), **payload["metadata"]}, indent=2))
    print(f"WAN_EMPTY_CONDITION_CREATED: {output}", flush=True)


if __name__ == "__main__":
    main()
