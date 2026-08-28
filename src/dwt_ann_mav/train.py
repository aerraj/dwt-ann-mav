"""Deterministic training, validation selection, and untouched held-out evaluation."""

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_dataset, read_manifest, split_recordings
from .model import MotorANN, predict, save_model


def metrics(y, probabilities, threshold=0.5):
    predicted = np.asarray(probabilities) >= threshold
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "samples": int(len(y)),
    }


def train(
    manifest,
    output,
    config,
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    seed=42,
    column=0,
    skiprows=0,
    baselines=True,
):
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.npz").exists():
        raise FileExistsError("Choose a new output directory; a trained model already exists")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    split = split_recordings(read_manifest(manifest), seed)
    data = {
        name: build_dataset(rows, config, column, skiprows, baselines)
        for name, rows in split.items()
    }
    x_train, y_train = data["train"][:2]
    mean = x_train.mean(axis=0)
    # Floor keeps the inverse scale representable in the Q16 hardware path.
    scale = np.maximum(x_train.std(axis=0), 1e-3)
    x = torch.tensor((x_train - mean) / scale, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    model = MotorANN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(x), generator=generator)
        for batch in indices.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        record = {"epoch": epoch}
        for name in ("train", "val"):
            xx, yy = data[name][:2]
            result = metrics(yy, predict(model, xx, mean, scale))
            record[f"{name}_loss"] = result["loss"]
            record[f"{name}_accuracy"] = result["accuracy"]
        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        history.append(record)
    model.load_state_dict(best_state)
    model.eval()
    report = {
        "schema_version": 1,
        "seed": seed,
        "config": config.to_dict(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "selected_epoch": best_epoch,
        "selection": "minimum validation BCE",
        "training_seconds": time.perf_counter() - started,
        "labels": {"0": "healthy", "1": "faulty"},
        "data_is_synthetic": (Path(manifest).parent / "SYNTHETIC_DATA.txt").exists(),
        "paper_reference_only": {"accuracy": 0.987, "loss": 0.032, "reproduced": False},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "splits": {},
        "baselines": {},
    }
    for name, (xx, yy, provenance, _) in data.items():
        report["splits"][name] = metrics(yy, predict(model, xx, mean, scale))
        report["splits"][name]["groups"] = sorted({r.group for r in split[name]})
        (output / f"{name}_windows.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if baselines:
        for kind in ("fft", "time"):
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000)
            )
            classifier.fit(data["train"][3][kind], y_train)
            report["baselines"][kind] = {
                name: metrics(values[1], classifier.predict_proba(values[3][kind])[:, 1])
                for name, values in data.items()
            }
    metadata = {
        "seed": seed,
        "selected_epoch": best_epoch,
        "data_is_synthetic": report["data_is_synthetic"],
    }
    save_model(output / "model.npz", model, mean, scale, config, metadata)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return report


def plot_run(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    history = json.loads((directory / "history.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for axis, metric in zip(axes[:2], ("loss", "accuracy")):
        for split in ("train", "val"):
            axis.plot(
                [row["epoch"] for row in history],
                [row[f"{split}_{metric}"] for row in history],
                label=split,
            )
        axis.set(xlabel="Epoch", ylabel=metric.title())
        axis.legend()
    cm = np.asarray(report["splits"]["test"]["confusion_matrix"])
    axes[2].imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        axes[2].text(j, i, str(value), ha="center", va="center")
    axes[2].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Healthy", "Faulty"],
        yticklabels=["Healthy", "Faulty"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Held-out confusion matrix",
    )
    fig.suptitle(
        "Synthetic integration run" if report["data_is_synthetic"] else "Measured experiment"
    )
    fig.savefig(directory / "evaluation.png", dpi=160)
    plt.close(fig)
