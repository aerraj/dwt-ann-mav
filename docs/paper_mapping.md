# Paper-to-code mapping

Source: *FPGA-Based Condition Monitoring of Three Phase Induction motors using DWT-driven Artificial Neural Network, and Motion Amplification Video*, Purbia, Raj, Barnwal and Singh, supplied as `Final_Research_Paper.pdf` (15 pages). The manuscript is not redistributed here.

| Requirement | Implementation | Qualification |
|---|---|---|
| ACS712 current acquisition, pp. 4–5 | `sensors.ACS712`, `convert-adc`, top-level RTL | ADC interface and calibration are board-specific |
| db44, nine levels, pp. 5, 8 | `features.py`, `resources/db44.json`, `dwt_features.sv` | True order 44 / 88 taps, not db4 or a 44-tap filter |
| Nine ANN inputs, p. 5 | Mean absolute detail coefficients D1…D9 | Reduction unspecified in paper; explicit engineering choice |
| 32/64/128/64/32/1 widths, p. 5 | `MotorANN`, `ann_inference.sv` | Five hidden layers plus output; table's “six hidden” conflicts with listed widths |
| ReLU/sigmoid binary classification | BCE-with-logits training and sigmoid inference | Healthy=0, faulty=1; threshold 0.5 |
| Batch 32, epochs 20 | Training defaults | Adam 0.001 and validation-loss selection are implementation choices |
| Laptop weights deployed to FPGA | Filter/normalizer/weight/bias ROM export | Fixed-point accuracy requires representative held-out recordings |
| Independent SW420/relay, pp. 5–6 | Python policy and `protection.sv` | Digital comparator state, not calibrated vibration amplitude |
| Learned encoder/manipulator/decoder, pp. 9–10 | Checkpoint-compatible MagNet | Architecture retained from existing MAV notebook |
| FFT/time-domain comparisons, p. 11 | Same group splits and logistic-regression baselines | Paper does not define exact baselines; these are measured reference baselines |
| 98.70%, loss 0.032, p. 11 | Reference metadata only | Original dataset/split unavailable; not reproduced |
| <1 ms DWT+ANN, p. 11 | Cycle estimate and simulation | Serial reference does not meet this claim; board optimization remains |

## db44 generation

PyWavelets' built-in Daubechies catalog ends at db38. The package generates db44 by factoring

`P(y) = sum[k=0..N-1] binomial(N-1+k,k) y^k`, where `y=(2-z-z^-1)/4`,

using 100 decimal digits and 500 extra working bits for root solving. The minimum-phase spectral factor is normalized to sum to sqrt(2); the high-pass filter is its alternating-sign reverse. Tests compare generated db2/db4/db20/db38 against independent PyWavelets filters, check db44 even-shift orthogonality, and perform nine-level reconstruction through PyWavelets' custom-wavelet API.

References: [PyWavelets filter-bank API](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html), [SciPy's historical spectral-factorization method and double-precision limitation](https://github.com/scipy/scipy/blob/v1.12.0/scipy/signal/_wavelets.py). Our generator uses arbitrary precision instead of double-precision polynomial roots.

```bash
python scripts/generate_db44.py --output /tmp/db44.json
```

## Exact feature contract

1. One contiguous calibrated current window, default 65,536 samples at 2,450 Hz (sample rate follows the notebook; the paper does not establish it).
2. At each level: `convolve(x, dec_lo, valid)[::2]` and `convolve(x, dec_hi, valid)[::2]`.
3. Record `mean(abs(detail))`, recurse on approximation. Order D1 through D9.
4. Fit mean/std only on training windows. Standard deviation is floored at 0.001 for exportable normalization. Store both with the checkpoint.

No autocorrelation, padding, cross-recording concatenation or downsample-by-5000 averaging is hidden in preprocessing. Nine valid db44 levels need at least **44,458 samples**. The default detail lengths are 32,725 / 16,319 / 8,116 / 4,015 / 1,964 / 939 / 426 / 170 / 42.

Boundary handling and mean-absolute reduction are engineering choices, not specified paper details. Changing them requires retraining and matching RTL changes.

## Scientific reproduction

Reproduction needs original recordings/acquisition metadata, motor/session groups, exact features, split/seed, training checkpoint/settings, calibration, and board timing reports. Synthetic tests are not reproduction. Verify that the selected dataset column actually measures current rather than vibration before training; no Kaggle column semantics are assumed.
