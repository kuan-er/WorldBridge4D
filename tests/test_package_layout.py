from pathlib import Path

import torch


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


def test_cli_entry_points_are_thin():
    root = Path(__file__).resolve().parents[1]
    limits = {
        "train_three_dataset_256_fsdp.py": 40,
        "infer_three_dataset_256.py": 20,
        "eval_source_rgb_counterfactual_256.py": 20,
    }
    for name, limit in limits.items():
        lines = (root / "scripts" / name).read_text().splitlines()
        assert len(lines) <= limit, f"{name} contains orchestration logic"


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
