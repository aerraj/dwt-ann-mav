"""Command-line entry points; no command actuates physical equipment."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .data import create_demo, load_signal
from .features import FeatureConfig, extract_features, windows
from .fixed import export_fpga, fixed_ann, fixed_features, fixed_probability
from .model import load_model, predict
from .sensors import ACS712, ProtectionController
from .train import plot_run, train


def feature_options(parser):
    parser.add_argument("--wavelet", default="db44")
    parser.add_argument("--window-size", type=int, default=65536)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--sample-rate", type=float, default=2450)


def config_from_args(args):
    return FeatureConfig(
        wavelet=args.wavelet,
        window_size=args.window_size,
        stride=args.window_size if args.stride is None else args.stride,
        sample_rate=args.sample_rate,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DWT/ANN/MAV research pipeline (no physical actuation)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Generate synthetic signals, train and export a smoke-test model"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--epochs", type=int, default=20)
    feature_options(demo)
    fitting = commands.add_parser("train", help="Train binary ANN on a recording manifest")
    fitting.add_argument("manifest", type=Path)
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--epochs", type=int, default=20)
    fitting.add_argument("--batch-size", type=int, default=32)
    fitting.add_argument("--learning-rate", type=float, default=1e-3)
    fitting.add_argument("--seed", type=int, default=42)
    fitting.add_argument("--column", type=int, default=0)
    fitting.add_argument("--skiprows", type=int, default=0)
    fitting.add_argument("--no-baselines", action="store_true")
    fitting.add_argument("--plots", action="store_true")
    feature_options(fitting)
    inference = commands.add_parser(
        "predict", help="Predict complete windows in one calibrated recording"
    )
    inference.add_argument("model", type=Path)
    inference.add_argument("recording", type=Path)
    inference.add_argument("--column", type=int, default=0)
    inference.add_argument("--skiprows", type=int, default=0)
    inference.add_argument("--threshold", type=float, default=0.5)
    exporting = commands.add_parser(
        "export-fpga", help="Export fixed-point ROMs and model_config.svh"
    )
    exporting.add_argument("model", type=Path)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.add_argument("--threshold", type=float, default=0.5)
    exporting.add_argument("--calibration", type=Path, help="ACS712 JSON calibration")
    verifying = commands.add_parser(
        "verify-fixed", help="Compare float/fixed DWT+ANN on complete windows"
    )
    verifying.add_argument("model", type=Path)
    verifying.add_argument("recording", type=Path)
    verifying.add_argument("--column", type=int, default=0)
    verifying.add_argument("--skiprows", type=int, default=0)
    converting = commands.add_parser(
        "convert-adc", help="Convert ADC codes to amperes before training"
    )
    converting.add_argument("recording", type=Path)
    converting.add_argument("--output", type=Path, required=True)
    converting.add_argument("--calibration", type=Path, required=True)
    converting.add_argument("--column", type=int, default=0)
    converting.add_argument("--skiprows", type=int, default=0)
    replay = commands.add_parser(
        "replay", help="Replay JSONL telemetry through latched protection policy"
    )
    replay.add_argument("events", type=Path)
    replay.add_argument("--threshold", type=float, default=0.5)
    replay.add_argument("--watchdog-seconds", type=float, default=60)
    video = commands.add_parser(
        "magnify", help="Learned video magnification using an explicit checkpoint"
    )
    video.add_argument("source", type=Path)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--checkpoint", type=Path, required=True)
    video.add_argument("--alpha", type=float, default=10)
    video.add_argument("--mode", choices=["static", "dynamic"], default="static")
    video.add_argument("--device", default="cpu")
    video.add_argument("--side-by-side", action="store_true")
    mav_train = commands.add_parser(
        "train-mav", help="Train MagNet on supervised A/B/C/target tuples"
    )
    mav_train.add_argument("manifest", type=Path)
    mav_train.add_argument("--output", type=Path, required=True)
    mav_train.add_argument("--epochs", type=int, default=12)
    mav_train.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        result = execute(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


def execute(args):
    if args.command in {"demo", "train"}:
        config = config_from_args(args)
        if args.command == "demo":
            manifest = create_demo(args.output / "data", config, seed=args.seed)
            result = train(
                manifest, args.output / "run", config, epochs=args.epochs, seed=args.seed
            )
            export_fpga(args.output / "run/model.npz", args.output / "fpga")
            return result
        result = train(
            args.manifest,
            args.output,
            config,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
            args.column,
            args.skiprows,
            not args.no_baselines,
        )
        if args.plots:
            plot_run(args.output)
        return result
    if args.command == "export-fpga":
        calibration = (
            ACS712(**json.loads(args.calibration.read_text())) if args.calibration else ACS712()
        )
        return export_fpga(args.model, args.output, args.threshold, calibration)
    if args.command == "convert-adc":
        if args.output.exists() or args.output.suffix != ".npy":
            raise ValueError("Output must be a new .npy file")
        calibration = ACS712(**json.loads(args.calibration.read_text()))
        amps = calibration.convert(load_signal(args.recording, args.column, args.skiprows))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, amps)
        return {"output": str(args.output), "samples": len(amps), "units": "amperes"}
    if args.command in {"predict", "verify-fixed"}:
        model, mean, scale, config, metadata = load_model(args.model)
        signal = load_signal(args.recording, args.column, args.skiprows)
        if args.command == "predict" and (
            not np.isfinite(args.threshold) or not 0 < args.threshold < 1
        ):
            raise ValueError("threshold must be in (0,1)")
        result = []
        for start, window in windows(signal, config):
            began = time.perf_counter()
            feature = extract_features(window, config)
            probability = float(predict(model, feature[None], mean, scale)[0])
            row = {
                "start_sample": start,
                "start_seconds": start / config.sample_rate,
                "probability_faulty": probability,
                "processing_seconds": time.perf_counter() - began,
            }
            if args.command == "predict":
                row["fault"] = probability >= args.threshold
            else:
                fixed_feature = fixed_features(window, config)
                logit = fixed_ann(fixed_feature, model, mean, scale)
                fixed_p = fixed_probability(logit) / 65536
                row.update(
                    {
                        "fixed_logit_q16": logit,
                        "fixed_probability": fixed_p,
                        "feature_max_abs_error": float(
                            np.max(np.abs(feature - fixed_feature / 65536))
                        ),
                        "probability_abs_error": abs(probability - fixed_p),
                        "classification_agrees": (probability >= 0.5) == (logit >= 0),
                    }
                )
            result.append(row)
        return {
            "config": config.to_dict(),
            "model_metadata": metadata,
            "windows": result,
            "trailing_samples_ignored": len(signal)
            - (result[-1]["start_sample"] + config.window_size),
        }
    if args.command == "replay":
        controller = ProtectionController(args.threshold, args.watchdog_seconds)
        with args.events.open() as stream:
            for line in stream:
                if line.strip():
                    print(json.dumps(controller.update(**json.loads(line))))
        return None
    if args.command == "magnify":
        from .video import magnify_video

        return magnify_video(
            args.source,
            args.output,
            args.checkpoint,
            args.alpha,
            args.mode,
            args.device,
            args.side_by_side,
        )
    if args.command == "train-mav":
        from .video import train_magnet

        return train_magnet(args.manifest, args.output, args.epochs, device=args.device)
    raise ValueError("Unknown command")


if __name__ == "__main__":
    sys.exit(main())
