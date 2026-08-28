# Learned motion amplification

The original MAV notebook uses [ZhengPeng7's PyTorch implementation](https://github.com/ZhengPeng7/motion_magnification_learning-based) of [Learning-based Video Motion Magnification, ECCV 2018](https://people.csail.mit.edu/tiam/deepmag/). The package retains that architecture/checkpoint with CPU support, safe loading and streaming IO.

## Provenance

- Source commit: `8f3f1e1c4feef4236729f6e8b2cb8f0cd4471ed2`, `magnet.py`.
- MIT copyright/license retained in `src/dwt_ann_mav/resources/MAGNET_LICENSE.txt`.
- Changes: removed unused CUDA data import, formatting, separate IO/training wrappers. Layer names/shapes unchanged.
- Checkpoint release: `v1.0`, `magnet_epoch12_loss7.28e-02.pth`, 3,528,201 bytes.
- SHA-256: `26c0e3ba528152d9219698ca4dac8a1924e3c4528f81277c3296fb7c034d4fae`.
- Download is explicit; weights are not committed. Loader uses `weights_only=True`, strips DataParallel prefixes and strictly checks tensors.

The inherited decoder uses nearest-neighbor upsampling plus convolutions, whereas the motor manuscript mentions transposed convolutions. The manipulator uses `M_b + h(alpha*g(M_b-M_a))`. Keeping the existing checkpoint-compatible architecture is deliberate, not a claim of exact equivalence to all manuscript details.

## Inference

BGR frames become RGB tensors in [-1,1], padded to multiples of four and cropped back. Pixels are clipped before uint8 conversion. Static mode references the first frame; dynamic mode the preceding frame. Only reference/current/output images and network tensors remain in memory, not the entire video.

`alpha` is **additional gain**, approximately `(1+alpha)` displacement in the trained small-motion regime. Zero bypasses reconstruction exactly. Camera motion, large gain, rolling shutter, compression, lighting changes and low FPS can create artifacts. This backend has no temporal band-pass filtering. Output supports qualitative inspection, not automatic diagnosis or calibrated vibration measurement.

Codec dimensions must be even and at least 16×16. MP4 uses `mp4v`, AVI uses MJPG. Constant FPS/dimensions are assumed. Audio and variable timestamps are not preserved. Metadata records counts, hashes, alpha, FPS and processing duration. Original files are never overwritten.

## Supervised training

Manifest columns: `a,b,c,target,alpha`. A is reference, C is the unperturbed second frame, B its texture/noise-perturbed version, and target the corresponding perturbed magnified frame. Paths are relative to the manifest; tuple dimensions must match.

```csv
a,b,c,target,alpha
sample1/a.png,sample1/b.png,sample1/c.png,sample1/magnified.png,10
```

```bash
motor-monitor train-mav data/mav-train.csv --output runs/magnet-custom.pth --epochs 12 --device cpu
```

Utility defaults: batch one, Adam 1e-4, reconstruction L1 plus 0.1 times texture/motion consistency losses. This provides a usable supervised training path, not a claim that a small local run reproduces upstream training. Use the [authors' data/protocol](https://people.csail.mit.edu/tiam/deepmag/) for research evaluation. The paper does not supply that dataset or motor-specific video ground truth.
