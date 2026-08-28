"""The paper's 9 -> 32 -> 64 -> 128 -> 64 -> 32 -> 1 binary ANN."""

import json
from itertools import pairwise

import numpy as np
import torch
from torch import nn

from .features import FeatureConfig

DIMS = (9, 32, 64, 128, 64, 32, 1)


class MotorANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in pairwise(DIMS)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x).squeeze(-1)  # logits; sigmoid at inference


def save_model(path, model, mean, scale, config, metadata=None):
    arrays = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    np.savez_compressed(
        path,
        **arrays,
        mean=mean,
        scale=scale,
        config=json.dumps(config.to_dict()),
        metadata=json.dumps(metadata or {}),
        format_version=np.asarray(1),
    )


def load_model(path):
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["format_version"]) != 1:
            raise ValueError("Unsupported model artifact version")
        config = FeatureConfig(**json.loads(str(saved["config"])))
        mean, scale = saved["mean"].copy(), saved["scale"].copy()
        if (
            mean.shape != (9,)
            or scale.shape != (9,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Invalid model normalization")
        model = MotorANN()
        state = {key: torch.from_numpy(saved[key].copy()) for key in model.state_dict()}
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Nonfinite model weights")
        model.load_state_dict(state, strict=True)
        metadata = json.loads(str(saved["metadata"]))
    return model.eval(), mean, scale, config, metadata


def predict(model, x, mean, scale):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
        raise ValueError("Expected finite [batch,9] features")
    with torch.inference_mode():
        return torch.sigmoid(
            model(torch.as_tensor((x - mean) / scale, dtype=torch.float32))
        ).numpy()
