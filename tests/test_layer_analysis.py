import numpy as np

from worldbridge.layer_analysis import (
    linear_cka_matrix, mrmr_layers, native_frame_indices, robust_rgb, token_center_pixels,
)


def test_native_frame_indices_and_token_centers_match_h005_contract():
    np.testing.assert_array_equal(native_frame_indices(21, 6), [0, 4, 8, 12, 16, 20])
    centers = token_center_pixels(128, 128, 8, 8)
    assert centers.shape == (64, 2)
    np.testing.assert_array_equal(centers[0], [8, 8])
    np.testing.assert_array_equal(centers[-1], [120, 120])


def test_linear_cka_is_symmetric_and_detects_identical_layers():
    rng = np.random.default_rng(7)
    first = rng.normal(size=(24, 12))
    second = first.copy()
    third = rng.normal(size=(24, 12))
    cka = linear_cka_matrix(np.stack((first, second, third)))
    np.testing.assert_allclose(cka, cka.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(cka), 1.0, atol=1e-10)
    assert cka[0, 1] > 0.999999


def test_mrmr_uses_relevance_then_avoids_redundant_candidate():
    probe = [1.0, 1.01, 1.2]
    correspondence = [1.0, 1.01, 1.2]
    cka = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    selected = mrmr_layers(probe, correspondence, cka, 2, redundancy_weight=5.0)
    assert selected == [0, 2]


def test_robust_rgb_is_finite_and_bounded():
    values = np.arange(90, dtype=np.float32).reshape(5, 6, 3)
    rgb = robust_rgb(values)
    assert np.isfinite(rgb).all()
    assert rgb.min() >= 0 and rgb.max() <= 1
