"""Matched 300-update clip=5 control from the low-LR FULL step-185500 state."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json

CONFIG = "configs/h031_k512_dr512_full_k9_clip5_185500_300.yaml"
CONFIG_SHA = "1bea4feb0fb5dfbd2207d6e6c9f41b502ce91876992150914839cd5e086a9a05"
BASE_CONFIG = "configs/h031_k512_dr512_full_k9_unfreeze_183000.yaml"
BASE_CONFIG_SHA = "7928fd682a0d6f0d5a0b90f25ed367fd5a08fe3a984089c36cb4a4feb730cbb3"
STEP = 185500
TARGET = 185800
SOURCE = Path("/data/WorldBridge4D-runs/h031-k512-dr512-full-k9-unfreeze-lowlr-183000-to193000-gpu16-r3-20260919/checkpoint-0185500.pt")
SOURCE_SHA = "c3bc7f90e3b1cb4c8fbc1c5e9b0036a967d832c8fce16dc1ddb513e2aface402"
HANDOFF = Path("/data/WorldBridge4D-runs/h031-k512-dr512-full-k9-clip5-185500-gpu45-handoff-20260920")
OUTPUT = Path("/data/WorldBridge4D-runs/h031-k512-dr512-full-k9-clip5-185500-to185800-gpu45-20260920")
UUIDS = ["GPU-25430d5b-58a4-d1c1-701b-ade48355308f", "GPU-5e216d50-d919-1bb1-fa53-0192f2cf3101"]


def health() -> str:
    raw = subprocess.check_output(["nvidia-smi", "-i", "4,5", "-q", "-x"], timeout=20)
    gpus = ET.fromstring(raw).findall("gpu")
    assert [g.findtext("uuid") for g in gpus] == UUIDS
    for gpu in gpus:
        for key in ("dram_uncorrectable", "sram_uncorrectable_parity", "sram_uncorrectable_secded"):
            assert gpu.findtext("ecc_errors/volatile/" + key) == "0", (gpu.findtext("uuid"), key)
        for key in ("remapped_row_pending", "remapped_row_failure"):
            assert gpu.findtext("remapped_rows/" + key) == "No", (gpu.findtext("uuid"), key)
    return raw.decode()


def gate() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONPATH"] = str(_ROOT / "src") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    assert file_sha256(CONFIG) == CONFIG_SHA
    assert file_sha256(BASE_CONFIG) == BASE_CONFIG_SHA
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    base = yaml.safe_load(Path(BASE_CONFIG).read_text())
    validate_config(cfg, 2)
    assert cfg["gradient_clip"] == 5.0 and base["gradient_clip"] == 1.0
    assert cfg["max_steps"] == TARGET and TARGET - STEP == 300
    assert cfg["selected_checkpoint_step"] == base["selected_checkpoint_step"] == 183000
    assert cfg["lr_restart"] == base["lr_restart"]
    assert cfg["dataset_mix_counts"] == base["dataset_mix_counts"]
    assert cfg["trainable_mode"] == "full"
    assert not HANDOFF.exists() and not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 38400000000
    subprocess.run([
        sys.executable, "-m", "pytest", "-q", "tests/test_clip5_control.py",
        "tests/test_native512_full.py", "tests/test_fp32_master.py",
        "tests/test_soft_torchrun.py",
    ], check=True)
    source_stat = SOURCE.lstat()
    assert SOURCE.is_file() and not SOURCE.is_symlink() and source_stat.st_size == 19363914891
    assert file_sha256(SOURCE) == SOURCE_SHA

    import torch
    payload = torch.load(SOURCE, map_location="cpu", mmap=True, weights_only=True)
    state = payload["training_state"]
    assert state["global_step"] == STEP
    assert state["world_size"] == len(state["rng_states"]) == 2
    assert state["dataset_cycle_offset"] == STEP % 20 == 0
    assert state["dataset_mix_phase_origin"] == 183000
    assert state["dataset_mix_counts"] == cfg["dataset_mix_counts"]
    assert payload["config"]["gradient_clip"] == 1.0
    assert payload["config"]["lr_restart"] == cfg["lr_restart"]
    optimizer = payload["optimizer"]
    assert len(optimizer["state"]) == 1053
    groups = {group["name"]: len(group["params"]) for group in optimizer["param_groups"]}
    assert groups == {
        "wan_backbone": 822, "geometry_adapter": 38, "dense_decoder": 132,
        "source_rgb_decay": 18, "source_rgb_no_decay": 43,
    }
    del payload

    temp = HANDOFF.with_name(HANDOFF.name + f".tmp-{os.getpid()}")
    temp.mkdir(exist_ok=False)
    os.link(SOURCE, temp / "resume.pt")
    pinned = (temp / "resume.pt").stat()
    assert (pinned.st_dev, pinned.st_ino, pinned.st_size) == (
        source_stat.st_dev, source_stat.st_ino, source_stat.st_size,
    )
    atomic_json(temp / "train_status.json", {"completed_steps": STEP, "world_size": 2})
    (temp / "gpu_health.xml").write_text(health())
    report = {
        "event": "H031_CLIP5_185500_300_GATE_OK",
        "source": str(SOURCE), "checkpoint_sha256": SOURCE_SHA,
        "checkpoint_bytes": source_stat.st_size, "resume_step": STEP,
        "target_step": TARGET, "updates": TARGET - STEP,
        "config": CONFIG, "config_sha256": CONFIG_SHA,
        "base_config_sha256": BASE_CONFIG_SHA,
        "scientific_delta": "gradient_clip 1.0 -> 5.0 only; exact FULL model/Adam/RNG/counters resume",
        "baseline": "existing R-20260919154805-1f6b19 steps185501-185800",
        "optimizer_states": 1053, "optimizer_group_parameter_tensors": groups,
        "world_size": 2, "B": 1, "A": 4, "K": 9,
        "clips_per_update": 8, "pairs_per_update": 72,
        "physical_gpus": [4, 5], "seed": cfg["seed"], "decoder_seed": cfg["decoder_seed"],
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "health_scope": "UUID volatile ECC and remap snapshot; PRL separately enforces exclusive lease",
    }
    atomic_json(temp / "complete.json", report)
    os.replace(temp, HANDOFF)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.gate:
        gate()
        return
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert file_sha256(CONFIG) == CONFIG_SHA
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    validate_config(cfg, 2)
    assert cfg["gradient_clip"] == 5.0 and cfg["max_steps"] == TARGET
    assert not OUTPUT.exists()
    report = json.loads((HANDOFF / "complete.json").read_text())
    assert report["checkpoint_sha256"] == SOURCE_SHA and report["config_sha256"] == CONFIG_SHA
    assert file_sha256(HANDOFF / "resume.pt") == SOURCE_SHA
    (HANDOFF / "gpu_health_at_launch.xml").write_text(health())
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["PYTHONPATH"] = str(_ROOT / "src") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    print(json.dumps({"event": "H031_CLIP5_185500_300_BEGIN", "gate": report}), flush=True)
    os.execv(sys.executable, [
        sys.executable, "-m", "worldbridge.trainer.soft_torchrun",
        "--standalone", "--nproc-per-node=2", "--log-dir", str(OUTPUT) + "-elastic", "--tee", "3",
        "scripts/train.py", "--config", CONFIG, "--output-dir", str(OUTPUT),
        "--resume", str(HANDOFF / "resume.pt"), "--startup-preflight-updates", "5",
    ])


if __name__ == "__main__":
    main()
