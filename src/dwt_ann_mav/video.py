"""Streaming learned motion magnification, independent of electrical protection."""

import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import file_hash
from .magnet import MagNet


def load_magnet(checkpoint, device="cpu"):
    model = MagNet().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    state = saved.get("state_dict", saved)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if not all(
        isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()
    ):
        raise ValueError("Invalid/nonfinite MagNet checkpoint")
    model.load_state_dict(state, strict=True)
    return model.eval()


def frame_tensor(frame, device="cpu"):
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
        raise ValueError("Frames must be BGR images at least 16x16")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    value = (
        torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None].to(device)
        / 127.5
        - 1
    )
    h, w = frame.shape[:2]
    return F.pad(value, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")


def magnify_pair(model, reference, frame, alpha=10.0, device="cpu"):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative (additional displacement gain)")
    if reference.shape != frame.shape:
        raise ValueError("Video dimensions changed")
    # Exact identity avoids reconstruction artifacts when magnification is disabled.
    if alpha == 0:
        return frame.copy()
    a, b = frame_tensor(reference, device), frame_tensor(frame, device)
    with torch.inference_mode():
        result = model(a, b, None, None, alpha, mode="evaluate")
    if not torch.isfinite(result).all():
        raise ValueError("MagNet returned nonfinite pixels")
    h, w = frame.shape[:2]
    rgb = (
        ((result[0, :, :h, :w].cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def magnify_video(
    source, destination, checkpoint, alpha=10.0, mode="static", device="cpu", side_by_side=False
):
    if mode not in {"static", "dynamic"}:
        raise ValueError("mode must be static or dynamic")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError("Output must be a new file, different from the source")
    if destination.suffix.lower() not in {".mp4", ".avi"}:
        raise ValueError("Output must be .mp4 or .avi")
    model = load_magnet(checkpoint, device)
    capture = cv2.VideoCapture(str(source))
    writer = None
    started, count = time.perf_counter(), 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, reference = capture.read()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot decode input or determine a valid frame rate")
        h, w = reference.shape[:2]
        if h % 2 or w % 2 or min(h, w) < 16:
            raise ValueError(
                "Video codec requires even dimensions at least 16x16; resize input explicitly"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        codec = "mp4v" if destination.suffix.lower() == ".mp4" else "MJPG"
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (w * (2 if side_by_side else 1), h),
        )
        if not writer.isOpened():
            raise ValueError("Could not initialize video encoder")
        writer.write(np.concatenate([reference, reference], axis=1) if side_by_side else reference)
        count = 1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            amplified = magnify_pair(model, reference, frame, alpha, device)
            writer.write(np.concatenate([frame, amplified], axis=1) if side_by_side else amplified)
            if mode == "dynamic":
                reference = frame
            count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Encoder produced no video")
    report = {
        "frames": count,
        "fps": fps,
        "width": w * (2 if side_by_side else 1),
        "height": h,
        "alpha_additional_gain": alpha,
        "mode": mode,
        "device": device,
        "processing_seconds": time.perf_counter() - started,
        "checkpoint_sha256": file_hash(checkpoint),
        "source_sha256": file_hash(source),
        "audio_preserved": False,
        "purpose": "qualitative inspection, not fault classification",
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def train_magnet(manifest, destination, epochs=12, learning_rate=1e-4, seed=42, device="cpu"):
    """Supervised A/B/C/M tuples from the upstream synthetic dataset.

    Manifest: a,b,c,target,alpha. B is perturbed C, target is the perturbed
    magnified image. Batch size 1 permits differing image sizes. This is a
    training utility, not a claim of equivalence to the upstream full training.
    """
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    manifest, destination = Path(manifest), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"a", "b", "c", "target", "alpha"}.issubset(reader.fieldnames or []):
            raise ValueError("MAV manifest requires a,b,c,target,alpha")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty MAV training manifest")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MagNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        losses = []
        for i in rng.permutation(len(rows)):
            row = rows[i]
            frames = [cv2.imread(str(manifest.parent / row[k])) for k in ("a", "b", "c", "target")]
            if any(x is None for x in frames) or len({x.shape for x in frames}) != 1:
                raise ValueError("Training images must be readable with matching dimensions")
            alpha = float(row["alpha"])
            if not math.isfinite(alpha) or alpha < 0:
                raise ValueError("Invalid training alpha")
            a, b, c, target = [frame_tensor(frame, device) for frame in frames]
            optimizer.zero_grad(set_to_none=True)
            prediction, texture_ac, texture_bm, motion_bc = model(a, b, c, target, alpha)
            loss = F.l1_loss(prediction, target) + 0.1 * sum(
                F.l1_loss(*pair) for pair in (texture_ac, texture_bm, motion_bc)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, destination)
    destination.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
    return history
