"""DR 512x512 cache CPU gate: tests + manifest + GPU1 health."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from worldbridge.data.cache.native import CONTRACT_DR512, file_sha256
from worldbridge.data.native_inputs import build_dr512_manifest, validate_manifest
from worldbridge.utils.io import atomic_json

CONFIG = "configs/h031_dr512_cache.yaml"
OUTPUT = Path("/data/WorldBridge4D-runs/h031-dr512-cache-20260919")
UUID = "GPU-e5923570-eb29-25bc-46f2-298f98a1706b"  # physical GPU 1


def health() -> str:
    raw = subprocess.check_output(["nvidia-smi", "-i", "1", "-q", "-x"], timeout=20)
    gpus = ET.fromstring(raw).findall("gpu")
    assert [g.findtext("uuid") for g in gpus] == [UUID]
    for g in gpus:
        for key in ("dram_uncorrectable", "sram_uncorrectable_parity", "sram_uncorrectable_secded"):
            assert g.findtext("ecc_errors/volatile/" + key) == "0", (g.findtext("uuid"), key)
        for key in ("remapped_row_pending", "remapped_row_failure"):
            assert g.findtext("remapped_rows/" + key) == "No", (g.findtext("uuid"), key)
    return raw.decode()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    vae = file_sha256(cfg["vae_checkpoint"])
    manifest = build_dr512_manifest(cfg, vae)
    validate_manifest(manifest)
    if args.gate:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/test_dr512_cache.py", "tests/test_native_latents.py",
                        "tests/test_native_rgb.py", "tests/test_native_rgb_stream.py",
                        "tests/test_native_gt_demand.py", "tests/test_native_cache_pair.py"], check=True)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "dynamic_replica.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (OUTPUT / "gpu_health.xml").write_text(health())
        report = dict(event="H031_DR512_CACHE_GATE_OK", config_sha256=file_sha256(CONFIG),
                      manifest_sha256=manifest["sha256"], contract=CONTRACT_DR512,
                      transform=manifest["transform"], native_hw=manifest["native_hw"],
                      latent_shape=manifest["latent_shape"], records=len(manifest["records"]),
                      cache_gpu=cfg["cache_gpu"], vae_sha256=vae,
                      cache_root=cfg["cache_root"], rgb_root=cfg["rgb_root"])
        atomic_json(OUTPUT / "gate_complete.json", report)
        print(json.dumps(report), flush=True)
        return
    raise SystemExit("this gate is CPU-only; the producer is h031_dr512_cache.py")


if __name__ == "__main__":
    main()
