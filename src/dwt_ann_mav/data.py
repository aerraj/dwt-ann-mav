"""Recording-level dataset manifests. No splitting of correlated windows."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import baseline_features, extract_features, windows


@dataclass(frozen=True)
class Recording:
    path: Path
    label: int
    group: str
    split: str = ""


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path,label,group columns; split is optional")
        rows = []
        for row in reader:
            if row["label"] not in {"0", "1"} or not row["group"].strip():
                raise ValueError("Labels must be 0=healthy or 1=faulty; groups must be nonempty")
            source = (path.parent / row["path"]).resolve()
            if not source.is_file():
                raise ValueError(f"Recording does not exist: {source}")
            split = (row.get("split") or "").strip()
            if split not in {"", "train", "val", "test"}:
                raise ValueError(f"Invalid split: {split}")
            rows.append(Recording(source, int(row["label"]), row["group"].strip(), split))
    if not rows or len({r.path for r in rows}) != len(rows):
        raise ValueError("Manifest is empty or has duplicate recording paths")
    return rows


def split_recordings(records, seed=42):
    group_labels = {}
    for r in records:
        if (
            not all(r.split for r in records)
            and r.group in group_labels
            and group_labels[r.group] != r.label
        ):
            raise ValueError(
                "Each group must have one binary label; use explicit splits for mixed-condition studies"
            )
        group_labels[r.group] = r.label
    if any(r.split for r in records):
        if not all(r.split for r in records):
            raise ValueError("Either every recording or no recording must specify a split")
        assigned = {}
        for r in records:
            if r.group in assigned and assigned[r.group] != r.split:
                raise ValueError(f"Data leakage: group {r.group!r} crosses splits")
            assigned[r.group] = r.split
    else:
        rng = np.random.default_rng(seed)
        assigned = {}
        for label in (0, 1):
            groups = sorted(g for g, y in group_labels.items() if y == label)
            if len(groups) < 3:
                raise ValueError(
                    "Need at least three independent groups per class for train/val/test"
                )
            rng.shuffle(groups)
            holdout = max(1, int(len(groups) * 0.2))
            for i, group in enumerate(groups):
                assigned[group] = "test" if i < holdout else "val" if i < 2 * holdout else "train"
    result = {
        split: [r for r in records if assigned[r.group] == split]
        for split in ("train", "val", "test")
    }
    for split, rows in result.items():
        if {r.label for r in rows} != {0, 1}:
            raise ValueError(f"{split} must contain both classes")
    # Identical content under different filenames must not cross splits either.
    seen = {}
    for split, rows in result.items():
        for r in rows:
            digest = file_hash(r.path)
            if digest in seen and seen[digest] != split:
                raise ValueError("Data leakage: identical recording content crosses splits")
            seen[digest] = split
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_signal(path, column=0, skiprows=0):
    path = Path(path)
    if column < 0 or skiprows < 0:
        raise ValueError("column and skiprows must be nonnegative")
    if path.suffix.lower() == ".npy":
        x = np.load(path, allow_pickle=False)
        if x.ndim == 2:
            x = x[:, column]
        elif column != 0:
            raise ValueError("A one-dimensional NPY file only has column 0")
    elif path.suffix.lower() == ".csv":
        x = np.loadtxt(path, delimiter=",", usecols=column, skiprows=skiprows, ndmin=1)
    else:
        raise ValueError("Recordings must be CSV or NPY")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"Invalid, empty or nonfinite signal: {path}")
    return x


def build_dataset(records, config, column=0, skiprows=0, baselines=False):
    features, labels, provenance = [], [], []
    baseline = {"fft": [], "time": []}
    for record in records:
        signal = load_signal(record.path, column, skiprows)
        digest = file_hash(record.path)
        for start, window in windows(signal, config):
            features.append(extract_features(window, config))
            labels.append(record.label)
            provenance.append(
                {
                    "path": str(record.path),
                    "sha256": digest,
                    "group": record.group,
                    "start": start,
                    "label": record.label,
                }
            )
            if baselines:
                for kind, values in baseline.items():
                    values.append(baseline_features(window, config, kind))
    return (
        np.asarray(features),
        np.asarray(labels),
        provenance,
        {k: np.asarray(v) for k, v in baseline.items()},
    )


def create_demo(directory, config, groups_per_class=12, seed=42):
    """Synthetic integration fixture, never evidence of real motor accuracy."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.csv"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest}")
    rng = np.random.default_rng(seed)
    t = np.arange(config.window_size * 2) / config.sample_rate
    rows = []
    for label in (0, 1):
        for group in range(groups_per_class):
            frequency = rng.uniform(48, 52)
            signal = rng.uniform(2, 4) * np.sin(2 * np.pi * frequency * t + rng.uniform(0, 6.28))
            signal += rng.normal(0, 0.02, len(t))
            if label:
                signal += 0.5 * np.sin(2 * np.pi * frequency * 3 * t)
                signal += 0.25 * np.sin(2 * np.pi * frequency * 7 * t)
                signal += rng.normal(0, 0.08, len(t))
            name = f"synthetic_{label}_{group:03d}.npy"
            np.save(directory / name, signal.astype(np.float32))
            rows.append({"path": name, "label": label, "group": f"synthetic_{label}_{group}"})
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "group"])
        writer.writeheader()
        writer.writerows(rows)
    (directory / "SYNTHETIC_DATA.txt").write_text(
        "Simulated signals for integration tests only. Not the paper dataset.\n"
    )
    return manifest
