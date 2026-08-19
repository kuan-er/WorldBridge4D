#!/usr/bin/env python3
"""Fail-fast package/API check for a WorldBridge4D training environment."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PACKAGES = (
    ("numpy", "numpy", ">=1.26,<2.0"),
    ("PyYAML", "yaml", ">=6.0,<7"),
    ("torch", "torch", ">=2.4,<2.11"),
    ("tensorflow-cpu", "tensorflow", ">=2.15,<2.17"),
    ("diffusers", "diffusers", "==0.36.0"),
    ("safetensors", "safetensors", ">=0.7,<0.8"),
    ("transformers", "transformers", ">=4.41,<5"),
    ("accelerate", "accelerate", ">=1.1,<2"),
    ("sentencepiece", "sentencepiece", ">=0.2,<0.3"),
    ("ftfy", "ftfy", ">=6.1,<7"),
    ("wandb", "wandb", ">=0.19,<1"),
    ("Pillow", "PIL", ">=10,<12"),
    ("jsonschema", "jsonschema", ">=4.19,<5"),
    ("packaging", "packaging", ">=23,<26"),
    ("pytest", "pytest", ">=8,<10"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true",
                        help="also require a CUDA GPU with BF16 support")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "apis": {},
        "cuda": {},
        "errors": [],
    }
    errors: list[str] = result["errors"]

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    python_version = Version(".".join(map(str, sys.version_info[:3])))
    if python_version not in SpecifierSet(">=3.10,<3.12"):
        errors.append(f"Python {python_version} is unsupported; use >=3.10,<3.12")

    loaded: dict[str, Any] = {}
    for distribution, module_name, specifier in PACKAGES:
        entry: dict[str, Any] = {"required": specifier}
        try:
            version = importlib.metadata.version(distribution)
            entry["version"] = version
            entry["import"] = "ok"
            loaded[module_name] = importlib.import_module(module_name)
            if Version(version) not in SpecifierSet(specifier):
                errors.append(f"{distribution} {version} does not satisfy {specifier}")
        except Exception as exc:  # make one report instead of failing at first import
            entry["import"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{distribution}/{module_name}: {entry['error']}")
        result["packages"][distribution] = entry

    try:
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
        from diffusers.loaders.single_file_utils import (
            convert_wan_transformer_to_diffusers,
            convert_wan_vae_to_diffusers,
            load_single_file_checkpoint,
        )
        symbols = (
            AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel,
            convert_wan_transformer_to_diffusers, convert_wan_vae_to_diffusers,
            load_single_file_checkpoint,
        )
        result["apis"]["diffusers_wan"] = [symbol.__name__ for symbol in symbols]
    except Exception as exc:
        errors.append(f"Diffusers Wan API check failed: {type(exc).__name__}: {exc}")
        result["apis"]["diffusers_wan"] = "failed"

    try:
        import jsonschema
        schema = json.loads((ROOT / "docs/dataset_manifest_v1.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        result["apis"]["dataset_manifest_schema"] = "ok"
    except Exception as exc:
        errors.append(f"dataset manifest schema check failed: {type(exc).__name__}: {exc}")
        result["apis"]["dataset_manifest_schema"] = "failed"

    try:
        tensorflow = loaded.get("tensorflow") or importlib.import_module("tensorflow")
        _ = tensorflow.train.Example()
        _ = tensorflow.data.TFRecordDataset
        result["apis"]["tensorflow_tfrecord"] = "ok"
    except Exception as exc:
        errors.append(f"TensorFlow TFRecord API check failed: {type(exc).__name__}: {exc}")
        result["apis"]["tensorflow_tfrecord"] = "failed"

    project_modules = (
        "worldbridge.data", "worldbridge.geometry", "worldbridge.dense4d",
        "worldbridge.dense4d_runtime", "worldbridge.dynamic_replica",
        "worldbridge.pointodyssey", "worldbridge.text_conditions",
        "worldbridge.training256", "worldbridge.wan",
    )
    imported = []
    for module_name in project_modules:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            errors.append(f"project import {module_name} failed: {type(exc).__name__}: {exc}")
    result["apis"]["project_imports"] = imported

    torch = loaded.get("torch")
    if torch is not None:
        result["cuda"] = {
            "torch_build": str(torch.version.cuda),
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["cuda"].update({
                "device_0": properties.name,
                "device_0_total_gib": round(properties.total_memory / 1024 ** 3, 3),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            })
        if args.require_cuda:
            if not torch.cuda.is_available():
                errors.append("CUDA is required but torch.cuda.is_available() is false")
            elif not torch.cuda.is_bf16_supported():
                errors.append("CUDA device does not support BF16")

    result["ok"] = not errors
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
    print("WORLD_BRIDGE_ENVIRONMENT_OK", flush=True)


if __name__ == "__main__":
    main()
