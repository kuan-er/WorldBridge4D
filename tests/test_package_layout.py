import importlib.util
from pathlib import Path
import runpy

import torch
import yaml


def test_production_cache_paths_use_persistent_storage():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml").read_text()
    )
    cache_paths = [config["source_rgb_cache_root"]]
    for values in config["datasets"].values():
        cache_paths.extend(
            value for key, value in values.items() if key.endswith("cache_root")
        )
    assert cache_paths
    assert all(
        Path(path).is_relative_to("/data/WorldBridge4D-persistent")
        for path in cache_paths
    )

    from worldbridge.data import constants

    runtime_paths = (
        constants.KUBRIC_GEOMETRY_CACHE,
        constants.POINTODYSSEY_ANNO_CACHE,
        constants.POINTODYSSEY_ANNO_NPY_CACHE,
        constants.POINTODYSSEY_DEPTH_CACHE,
        constants.DYNAMIC_REPLICA_DEPTH_CACHE,
        constants.DYNAMIC_REPLICA_TRAJECTORY_CACHE,
    )
    assert all(path.is_relative_to("/data/WorldBridge4D-persistent") for path in runtime_paths)


def test_legacy_model_imports_are_identity_compatible():
    from worldbridge.dense4d import DenseQueryDecoder as LegacyDecoder
    from worldbridge.dense4d import DenseQueryWanModel as LegacyModel
    from worldbridge.models import DenseQueryDecoder, DenseQueryWanModel

    assert LegacyDecoder is DenseQueryDecoder
    assert LegacyModel is DenseQueryWanModel


def test_legacy_data_imports_are_identity_compatible():
    from worldbridge.data import MOViFDataset
    from worldbridge.training256 import MOViF256Dataset as LegacyMOViF256Dataset
    from worldbridge.data.datasets import MOViF256Dataset

    assert MOViFDataset.__module__ == "worldbridge.data.movif"
    assert LegacyMOViF256Dataset is MOViF256Dataset


def test_scripts_expose_exactly_five_thin_entrypoints():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    limits = {
        "train.py": 20,
        "infer.py": 20,
        "evaluate.py": 20,
        "prepare_data.py": 60,
        "run_fsdp.sh": 200,
    }
    entrypoints = {
        path.name for path in scripts.iterdir() if path.suffix in {".py", ".sh"}
    }
    assert entrypoints == set(limits)
    for name, limit in limits.items():
        lines = (scripts / name).read_text().splitlines()
        assert len(lines) <= limit, f"{name} contains orchestration logic"


def test_prepare_data_subcommands_resolve_to_package_modules():
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    commands = runpy.run_path(str(script))["COMMANDS"]
    assert len(commands) == 19
    assert all(importlib.util.find_spec(module) is not None for module in commands.values())


def test_refactored_decoder_forward_constructs_typed_output():
    from worldbridge.models import DenseQueryDecoder, DenseQueryOutput

    torch.manual_seed(7)
    model = DenseQueryDecoder(
        num_frames=21, latent_shape=(8, 2, 4, 4), query_dim=16,
        embedding_dim=8, num_layers=1, num_heads=4,
        upsample_channels=(16, 8), output_size=(8, 8), query_grid_size=4,
    )
    output = model(
        torch.randn(1, 8, 2, 4, 4),
        torch.tensor([[0, 2]]), torch.tensor([[1, 3]]),
    )
    assert isinstance(output, DenseQueryOutput)
    assert output.normalized_xyz.shape == (1, 2, 3, 8, 8)
    assert torch.isfinite(output.normalized_xyz).all()


def test_package_layers_import_without_compatibility_modules():
    from worldbridge.data.factory import load_training_datasets
    from worldbridge.evaluation.inference import main as inference_main
    from worldbridge.models.factory import build_real_model
    from worldbridge.trainer.trainer import WorldBridgeTrainer

    assert callable(load_training_datasets)
    assert callable(inference_main)
    assert callable(build_real_model)
    assert hasattr(WorldBridgeTrainer, "fit")
