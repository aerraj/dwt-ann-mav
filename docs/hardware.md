# FPGA integration

The SystemVerilog is a board-independent research reference validated with Icarus simulation. No Quartus project, bitstream, pin assignment, timing closure, utilization or power result is claimed. The paper names Stratix III but omits exact device, board, ADC, clock and wiring.

**Do not connect motor mains, sensor outputs or a relay to unspecified FPGA pins.** ADC levels, isolation, conditioning, contactor drivers, flyback suppression and independent emergency-stop/overload protection need qualified engineering review. This is not certified motor-protection equipment.

## Build inputs

Run `motor-monitor export-fpga MODEL --output ROM_DIR --calibration CALIBRATION_JSON`.

| Export | Meaning |
|---|---|
| `model_config.svh` | Window/taps, ADC conversion, logit threshold |
| `dwt_low.mem`, `dwt_high.mem` | Signed 32-bit filters, 30 fractional bits |
| `mean.mem`, `inverse_scale.mem` | Nine normalizer parameters, 16 fractional bits |
| `weights.mem`, `biases.mem` | 20,800 weights / 321 biases, output-major |
| `sigmoid.mem` | 257 Q16 probabilities on logits [-8,8], step 1/16 |
| `manifest.json` | Configuration, arithmetic, calibration, model metadata, latency estimate |

Add `fpga/rtl/*.sv` to a SystemVerilog project, add ROM_DIR to include/search paths and resolve `.mem` files for synthesis. Wrap `motor_monitor` with the actual ADC interface, device, pin and clock constraints. Review inferred memories and dividers: this architecture is not optimized for a particular Stratix RAM/DSP layout. Export rejects overlapping-window configurations because this hardware captures non-overlapping windows.

## Handshake and acquisition

`adc_code`, `adc_valid`, `adc_ready` use a single-clock ready/valid handshake. A code is consumed only when both valid and ready are high. Hold unaccepted data stable. Each captured window must contain **contiguous samples at the declared sample rate**, not opportunistic samples from a free-running ADC.

The reference captures a full window, computes DWT, then runs ANN. It cannot simultaneously capture the next window. For continuous acquisition add an appropriately sized FIFO/double buffer in the board wrapper; otherwise there are gaps between windows. Buffer overflow, ADC errors and acquisition failures must drive `sensor_ok` low. There is no SPI/I2C ADC driver because the ADC is unspecified. An incomplete window never emits a prediction; stalled inference eventually trips the watchdog.

## Arithmetic

Stored samples, features, normalization, weights, biases and activations are signed 32-bit values with 16 fractional bits. Filter coefficients have 30 fractional bits. MACs accumulate in 96 bits, arithmetic-shift, then saturate to signed 32 bits. Mean-absolute features divide positive sums by valid coefficient counts. Python `fixed.py` mirrors these rules with arbitrary-precision intermediates; tests compare actual RTL bit for bit.

Export rejects unrepresentable parameters; runtime activations saturate. Tiny db44 taps quantize to zero, counted in the export manifest. Run `verify-fixed` on real held-out data: float accuracy does not validate fixed-point classification. ADC gain and offset are separately quantized and need calibration-error checks. Physical sensor validity is the board wrapper's responsibility.

Classification uses a signed logit threshold. The sigmoid LUT is for telemetry only, using a left-bin lookup and clipping logits to [-8,8]. Quantized decisions can differ near threshold.

## Protection

- Reset de-energizes the relay and leaves it unarmed.
- A healthy, current result plus an acknowledgment **rising edge** permits arming. Holding acknowledgment high cannot continuously rearm.
- Fault inference, synchronized SW420 assertion, `sensor_ok=0` or inference timeout latches trip. Fault conditions defeat acknowledgment.
- SW420 uses a two-flop synchronizer independently of DWT/ANN. Active-high is assumed; invert in the board wrapper if needed. Very short pulses need external capture/stretching.
- `sensor_ok` and `acknowledge` are synchronous inputs. Synchronize/debounce physical switches and satisfy clock/reset requirements in the wrapper.

`WATCHDOG_CYCLES` must be positive and sized from measured clock rate, acquisition length and worst-case compute latency. Default two billion cycles corresponds to 40 seconds at an **illustrative** 50 MHz, not a claimed board clock.

## Timing limits

Acquisition alone takes **26.75 seconds** at the default 65,536 samples / 2,450 Hz. DWT reuses two MACs across taps and takes roughly 5.76 million cycles, about 115 ms at an illustrative 50 MHz; ANN takes roughly 22,000 cycles. Synthesis/timing can impose further limits. This does **not** meet the paper's sub-millisecond DWT+ANN claim.

Meeting a tighter budget requires pipelined/parallel filter banks, buffering, a justified acquisition configuration, synthesis and timing closure. Valid nine-level db44 cannot use fewer than 44,458 samples. Do not hide acquisition delay or call simulation counts hardware measurements.
