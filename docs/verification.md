# Verification record

Local validation on 2026-08-28, Python 3.12, CPU inference. This is implementation validation, not reproduction of the motor study.

## Checks run

`python -m pytest -q`: **24 passed** (no RTL skips).

- Numerical/db44 checks: generator agrees with PyWavelets db2/db4/db20/db38; db44 passes orthogonality and nine-level reconstruction checks.
- Executable RTL: repeated db4 and full db44 windows match Python integer features exactly; six-layer ANN matches exported fixed-point weights, including extreme signed inputs; ADC-to-DWT-to-ANN and sigmoid telemetry match the reference.
- Protection: startup, independent vibration, latched trips, fault priority, inference watchdog and acknowledgment-edge behavior tested.
- Training: deterministic replay, training-only normalization, disjoint recording groups and duplicate-content leakage rejection tested.
- Packaging: wheel builds and includes db44 coefficients and upstream MagNet license.
- Code formatting/lint checks pass.

## Synthetic integration experiment

Command: `motor-monitor demo --output runs/paper-smoke --epochs 20`.

Default db44, nine levels, 65,536-sample windows, 2,450 Hz, batch 32, seed 42. There are 12 generated recording groups per class and two windows per recording. The held-out test has only eight windows. All eight classify correctly, with binary cross-entropy approximately 0.509. These deliberately simple simulated faults do **not** establish real motor accuracy or reproduce the paper's reported loss.

On all eight held-out synthetic windows, float/fixed decisions agree. Maximum mean-absolute feature error is approximately `1.64e-5` amperes in coefficient units. Probability telemetry differs because it uses a quantized sigmoid lookup. The db44 export has 19 low-pass taps quantized to zero; this is disclosed rather than treating fixed point as numerically exact to float.

The reference DWT estimate is 5,759,742 cycles. Acquisition alone takes 26.7494 seconds. No sub-millisecond or physical board timing claim is made.

## Pretrained video smoke run

The pinned upstream MagNet checkpoint was downloaded and SHA-256 checked, loaded on CPU, and used to amplify a generated 30-frame, 30-FPS, 128×96 subtle-translation video at alpha 10. The side-by-side output was reopened and every frame decoded: 30 frames, 30 FPS, 256×96. This verifies learned-model/codec integration, not motor-specific diagnostic performance. Unit tests also exercise one supervised training epoch and checkpoint reload.

## Not verified

Original experimental recordings, physical sensor calibration, real motor footage/ground truth, Stratix III synthesis/place-and-route, pin constraints, timing closure, resource use, power, continuous acquisition buffering, and safety qualification. See `hardware.md` for required board integration work.
