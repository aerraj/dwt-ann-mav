"""Bit-accurate signed 32-bit reference and memory export for fpga/rtl.

Signals/ANN use 16 fractional bits; wavelet taps use 30. Products accumulate
without intermediate rounding, arithmetic right shifts truncate toward -inf,
and all stored signed values saturate. Positive mean-absolute sums divide down.
"""

import json
import math
from pathlib import Path

import numpy as np

from .features import filters
from .model import DIMS, load_model
from .sensors import ACS712

MIN_INT, MAX_INT = -(1 << 31), (1 << 31) - 1
FRAC, FILTER_FRAC = 16, 30


def saturate(x):
    return min(MAX_INT, max(MIN_INT, int(x)))


def quantize(values, frac=FRAC, strict=True):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Cannot quantize NaN or infinity")
    scaled = np.rint(values * 2**frac)
    if strict and np.any((scaled < MIN_INT) | (scaled > MAX_INT)):
        raise ValueError("Value exceeds signed fixed-point range; recalibrate/retrain")
    return np.clip(scaled, MIN_INT, MAX_INT).astype(np.int64)


def fixed_features(samples, config, already_quantized=False):
    x = np.asarray(samples, dtype=object) if already_quantized else quantize(samples).astype(object)
    if x.ndim != 1 or len(x) != config.window_size:
        raise ValueError("Invalid fixed-point window")
    low, high = [quantize(c, FILTER_FRAC).astype(object) for c in filters(config.wavelet)]
    result = []
    for _ in range(config.levels):
        detail = [saturate(int(a) >> FILTER_FRAC) for a in np.convolve(x, high, "valid")[::2]]
        x = np.asarray(
            [saturate(int(a) >> FILTER_FRAC) for a in np.convolve(x, low, "valid")[::2]],
            dtype=object,
        )
        result.append(saturate(sum(abs(a) for a in detail) // len(detail)))
    return np.asarray(result, dtype=np.int64)


def fixed_ann(features, model, mean, scale):
    if np.asarray(features).shape != (9,):
        raise ValueError("Expected nine fixed-point features")
    means, inverse = quantize(mean), quantize(1 / scale)
    x = np.asarray(
        [saturate((int(a) - int(b)) * int(c) >> FRAC) for a, b, c in zip(features, means, inverse)],
        dtype=object,
    )
    if x.shape != (9,):
        raise ValueError("Expected nine fixed-point features")
    for index, layer in enumerate(model.layers):
        w = quantize(layer.weight.detach().numpy()).astype(object)
        bias = quantize(layer.bias.detach().numpy()).astype(object)
        totals = w @ x + bias * (1 << FRAC)
        x = np.asarray([saturate(int(total) >> FRAC) for total in totals], dtype=object)
        if index < len(model.layers) - 1:
            x = np.maximum(x, 0)
    return int(x[0])


def sigmoid_lut():
    return quantize(1 / (1 + np.exp(-np.linspace(-8, 8, 257))))


def fixed_probability(logit):
    return int(sigmoid_lut()[min(256, max(0, (int(logit) + (8 << FRAC)) >> 12))])


def write_mem(path, values):
    Path(path).write_text("".join(f"{int(x) & 0xFFFFFFFF:08x}\n" for x in np.asarray(values).flat))


def export_fpga(model_path, directory, threshold=0.5, calibration=None):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be in (0,1)")
    model, mean, scale, config, metadata = load_model(model_path)
    if config.stride != config.window_size:
        raise ValueError(
            "Reference RTL supports non-overlapping windows only; set stride=window_size"
        )
    calibration = calibration or ACS712()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    low, high = filters(config.wavelet)
    arrays = {
        "dwt_low.mem": quantize(low, FILTER_FRAC),
        "dwt_high.mem": quantize(high, FILTER_FRAC),
        "mean.mem": quantize(mean),
        "inverse_scale.mem": quantize(1 / scale),
        "weights.mem": np.concatenate(
            [quantize(layer.weight.detach().numpy()).ravel() for layer in model.layers]
        ),
        "biases.mem": np.concatenate(
            [quantize(layer.bias.detach().numpy()) for layer in model.layers]
        ),
        "sigmoid.mem": sigmoid_lut(),
    }
    threshold_q = int(quantize(math.log(threshold / (1 - threshold))))
    gain, offset = int(quantize(calibration.gain)), int(quantize(calibration.offset))
    for name, value in arrays.items():
        write_mem(directory / name, value)
    lengths = []
    n = config.window_size
    for _ in range(9):
        n = (n - len(low)) // 2 + 1
        lengths.append(n)
    report = {
        "schema_version": 1,
        "config": config.to_dict(),
        "model_metadata": metadata,
        "signal_fraction_bits": FRAC,
        "filter_fraction_bits": FILTER_FRAC,
        "word_bits": 32,
        "accumulator_bits": 96,
        "architecture": list(DIMS),
        "threshold": threshold,
        "threshold_logit_q16": threshold_q,
        "adc": calibration.__dict__,
        "adc_gain_q16": gain,
        "adc_offset_q16": offset,
        "detail_lengths": lengths,
        "filter_taps": len(low),
        "zeroed_filter_taps": int(np.count_nonzero(arrays["dwt_low.mem"] == 0)),
        "acquisition_seconds": config.window_size / config.sample_rate,
        "dwt_estimated_cycles": sum(n * (len(low) + 1) + 2 for n in lengths),
        "board_validated": False,
        "warning": "Reference RTL: validate quantization, synthesis, timing and safety before hardware use.",
    }
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "model_config.svh").write_text(
        f"`define MODEL_WINDOW {config.window_size}\n`define MODEL_TAPS {len(low)}\n"
        f"`define MODEL_THRESHOLD {threshold_q}\n`define ADC_GAIN {gain}\n`define ADC_OFFSET {offset}\n"
        f"`define ADC_BITS {calibration.adc_bits}\n"
    )
    return report
