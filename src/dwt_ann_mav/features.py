"""One shared, causal DWT feature definition for training and RTL.

No boundary extension: convolve in valid mode, retain indices 0,2,4,... .
Inputs are a single calibrated current channel in amperes. Features are
mean(abs(detail)) in D1..D9 order; the final approximation is not an input.
"""

import json
from dataclasses import asdict, dataclass
from importlib.resources import files

import numpy as np
import pywt


@dataclass(frozen=True)
class FeatureConfig:
    wavelet: str = "db44"
    levels: int = 9
    window_size: int = 65536
    stride: int = 65536
    sample_rate: float = 2450.0

    def __post_init__(self):
        if self.levels != 9:
            raise ValueError("The paper ANN and RTL require exactly nine levels")
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not isinstance(self.stride, int) or not 0 < self.stride <= self.window_size:
            raise ValueError("stride must be an integer in [1, window_size]")
        low, _ = filters(self.wavelet)
        if not isinstance(self.window_size, int) or self.window_size < minimum_window(
            len(low), self.levels
        ):
            raise ValueError(
                f"{self.wavelet}/{self.levels} needs at least "
                f"{minimum_window(len(low), self.levels)} samples without padding"
            )

    def to_dict(self):
        return asdict(self)


def filters(name="db44"):
    if name == "db44":
        bank = json.loads(files("dwt_ann_mav").joinpath("resources/db44.json").read_text())
        return np.asarray(bank["dec_lo"]), np.asarray(bank["dec_hi"])
    wavelet = pywt.Wavelet(name)
    if not wavelet.orthogonal:
        raise ValueError("Only orthogonal discrete wavelets are supported")
    return np.asarray(wavelet.dec_lo), np.asarray(wavelet.dec_hi)


def minimum_window(taps, levels):
    n = 1
    for _ in range(levels):
        n = 2 * (n - 1) + taps
    return n


def detail_coefficients(window, config):
    current = np.asarray(window, dtype=np.float64)
    if current.ndim != 1 or len(current) != config.window_size:
        raise ValueError(f"Expected one window of {config.window_size} samples")
    if not np.isfinite(current).all():
        raise ValueError("Current samples contain NaN or infinity")
    low, high = filters(config.wavelet)
    details = []
    for _ in range(config.levels):
        details.append(np.convolve(current, high, mode="valid")[::2])
        current = np.convolve(current, low, mode="valid")[::2]
    return details


def extract_features(window, config):
    return np.asarray([np.mean(np.abs(x)) for x in detail_coefficients(window, config)])


def windows(signal, config):
    signal = np.asarray(signal)
    if signal.ndim != 1 or len(signal) < config.window_size:
        raise ValueError("Recording is shorter than a window or is not one-dimensional")
    for start in range(0, len(signal) - config.window_size + 1, config.stride):
        yield start, signal[start : start + config.window_size]


def baseline_features(window, config, kind):
    x = np.asarray(window, dtype=np.float64)
    if kind == "fft":
        spectrum = np.abs(np.fft.rfft(x)) / len(x)
        freq = np.fft.rfftfreq(len(x), 1 / config.sample_rate)
        return np.asarray(
            [
                np.mean(
                    spectrum[
                        (freq > config.sample_rate / 2 ** (j + 1))
                        & (freq <= config.sample_rate / 2**j)
                    ]
                )
                for j in range(1, 10)
            ]
        )
    if kind != "time":
        raise ValueError("Unknown baseline")
    centered = x - np.mean(x)
    sd = max(float(np.std(x)), 1e-12)
    return np.asarray(
        [
            np.mean(x),
            sd,
            np.sqrt(np.mean(x * x)),
            np.mean(np.abs(x)),
            np.ptp(x),
            np.max(np.abs(x)),
            np.mean((centered / sd) ** 3),
            np.mean((centered / sd) ** 4),
            np.mean(x[1:] * x[:-1] < 0),
        ]
    )
