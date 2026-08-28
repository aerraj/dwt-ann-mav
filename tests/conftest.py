import numpy as np
import pytest
import torch

from dwt_ann_mav.features import FeatureConfig
from dwt_ann_mav.model import MotorANN, save_model


@pytest.fixture
def config():
    return FeatureConfig(wavelet="db4", window_size=4096, stride=4096)


@pytest.fixture
def artifact(tmp_path, config):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = MotorANN().eval()
    mean = np.linspace(0.1, 0.9, 9)
    scale = np.linspace(0.2, 1, 9)
    path = tmp_path / "model.npz"
    save_model(path, model, mean, scale, config, {"data_is_synthetic": True})
    return path
