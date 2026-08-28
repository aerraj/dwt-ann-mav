"""Calibrated ACS712 conversion and software reference for latched protection.

This is a simulation/telemetry policy, not an electrical safety guarantee.
SW420 supplies a digital comparator state, not vibration acceleration.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACS712:
    adc_bits: int = 12
    adc_reference_volts: float = 3.3
    sensor_zero_volts: float = 2.5
    sensitivity_volts_per_amp: float = 0.1
    divider_ratio: float = 0.6

    def __post_init__(self):
        if not isinstance(self.adc_bits, int) or not 2 <= self.adc_bits <= 24:
            raise ValueError("ADC bits must be in [2,24]")
        values = (
            self.adc_reference_volts,
            self.sensor_zero_volts,
            self.sensitivity_volts_per_amp,
            self.divider_ratio,
        )
        if not all(math.isfinite(x) and x > 0 for x in values) or self.divider_ratio > 1:
            raise ValueError("Invalid sensor calibration")

    @property
    def gain(self):
        return self.adc_reference_volts / (
            (2**self.adc_bits - 1) * self.divider_ratio * self.sensitivity_volts_per_amp
        )

    @property
    def offset(self):
        return self.sensor_zero_volts / self.sensitivity_volts_per_amp

    def convert(self, codes):
        codes = np.asarray(codes, dtype=np.float64)
        if (
            not np.isfinite(codes).all()
            or np.any(codes != np.floor(codes))
            or np.any((codes < 0) | (codes >= 2**self.adc_bits))
        ):
            raise ValueError("ADC codes must be finite unsigned integers within the ADC range")
        return codes * self.gain - self.offset


class ProtectionController:
    def __init__(self, threshold=0.5, watchdog_seconds=60.0):
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be in (0,1)")
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise ValueError("watchdog must be positive")
        self.threshold = threshold
        self.watchdog = watchdog_seconds
        self.last_result = None
        self.last_time = None
        self.last_probability = None
        self.tripped = True
        self.reason = "startup_unarmed"
        self.acknowledge_previous = False

    def update(
        self, timestamp, probability=None, vibration=False, sensor_ok=True, acknowledge=False
    ):
        if (
            not math.isfinite(timestamp)
            or timestamp < 0
            or (self.last_time is not None and timestamp < self.last_time)
        ):
            raise ValueError("Timestamps must be finite, nonnegative and monotonic")
        self.last_time = timestamp
        acknowledge_edge = acknowledge and not self.acknowledge_previous
        self.acknowledge_previous = acknowledge
        invalid = probability is not None and (
            not math.isfinite(probability) or not 0 <= probability <= 1
        )
        if probability is not None and not invalid:
            self.last_result = timestamp
            self.last_probability = probability
        reason = None
        if not sensor_ok or invalid:
            reason = "sensor_or_model_invalid"
        elif vibration:
            reason = "vibration"
        elif self.last_result is None or timestamp >= self.last_result + self.watchdog:
            reason = "watchdog"
        elif self.last_probability >= self.threshold:
            reason = "ann_fault"
        if reason:
            self.tripped, self.reason = True, reason
        elif acknowledge_edge:
            self.tripped, self.reason = False, "armed"
        return {
            "timestamp": timestamp,
            "trip": self.tripped,
            "relay_enable": not self.tripped,
            "reason": self.reason,
            "probability": self.last_probability,
        }
