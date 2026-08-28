import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pywt

from dwt_ann_mav.features import (
    FeatureConfig,
    detail_coefficients,
    extract_features,
    filters,
    minimum_window,
    windows,
)
from dwt_ann_mav.fixed import fixed_features, quantize


def test_db44_orthogonality_and_roundtrip():
    low, high = filters()
    assert len(low) == len(high) == 88
    np.testing.assert_allclose(low.sum(), np.sqrt(2), atol=1e-14)
    np.testing.assert_allclose(high.sum(), 0, atol=1e-14)
    for shift in range(0, 88, 2):
        np.testing.assert_allclose(
            low[: 88 - shift] @ low[shift:], 1 if shift == 0 else 0, atol=1e-14
        )
    bank = pywt.Wavelet("custom_db44", filter_bank=[low, high, low[::-1], high[::-1]])
    signal = np.random.default_rng(8).normal(size=65536)
    coeffs = pywt.wavedec(signal, bank, level=9)
    np.testing.assert_allclose(pywt.waverec(coeffs, bank), signal, atol=1e-13)


@pytest.mark.parametrize("order", [2, 4, 20, 38])
def test_generator_against_independent_pywavelets(order):
    spec = importlib.util.spec_from_file_location(
        "generate", Path(__file__).parents[1] / "scripts/generate_db44.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.testing.assert_allclose(
        module.coefficients(order), pywt.Wavelet(f"db{order}").dec_lo, atol=1e-15
    )


def test_valid_convolution_matches_pywavelets_interior(config):
    signal = np.random.default_rng(2).normal(size=config.window_size)
    current = signal
    for ours in detail_coefficients(signal, config):
        approximation, detail = pywt.dwt(current, config.wavelet, mode="zero")
        taps = len(filters(config.wavelet)[0])
        # PyWavelets starts at convolution sample 1; our valid start is taps-1.
        start = (taps - 2) // 2
        np.testing.assert_allclose(ours, detail[start : start + len(ours)], atol=1e-14)
        current = approximation[start : start + len(ours)]


def test_feature_order_and_fixed_error(config):
    signal = 2 * np.sin(np.arange(config.window_size) * 0.23)
    details = detail_coefficients(signal, config)
    np.testing.assert_allclose(
        extract_features(signal, config), [np.abs(d).mean() for d in details]
    )
    np.testing.assert_allclose(
        fixed_features(signal, config) / 65536, extract_features(signal, config), atol=0.002
    )
    assert len(list(windows(np.zeros(2 * config.window_size + 10), config))) == 2


def test_validation(config):
    assert minimum_window(88, 9) == 44458
    with pytest.raises(ValueError, match="at least"):
        FeatureConfig(window_size=1000, stride=1000)
    with pytest.raises(ValueError):
        extract_features(np.full(config.window_size, np.nan), config)
    with pytest.raises(ValueError):
        quantize([np.inf])
    with pytest.raises(ValueError):
        quantize([1e6])
