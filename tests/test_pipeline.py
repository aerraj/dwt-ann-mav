import json

import numpy as np
import pytest

from dwt_ann_mav.data import Recording, create_demo, load_signal, read_manifest, split_recordings
from dwt_ann_mav.fixed import export_fpga, fixed_ann, quantize
from dwt_ann_mav.model import load_model, predict, save_model
from dwt_ann_mav.sensors import ACS712, ProtectionController
from dwt_ann_mav.train import train


def test_training_roundtrip_and_no_leakage(tmp_path, config):
    manifest = create_demo(tmp_path / "data", config, groups_per_class=4)
    report = train(manifest, tmp_path / "run", config, epochs=2, baselines=True)
    groups = [set(report["splits"][split]["groups"]) for split in ("train", "val", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    model, mean, scale, loaded_config, metadata = load_model(tmp_path / "run/model.npz")
    assert loaded_config == config
    assert metadata["data_is_synthetic"]
    assert report["paper_reference_only"]["reproduced"] is False
    assert set(report["baselines"]) == {"fft", "time"}
    assert np.isfinite(predict(model, np.ones((2, 9)), mean, scale)).all()
    other = train(manifest, tmp_path / "repeat", config, epochs=2, baselines=False)
    assert other["splits"] == report["splits"]
    means = []
    for row in json.loads((tmp_path / "run/train_windows.json").read_text()):
        from dwt_ann_mav.features import extract_features

        x = load_signal(row["path"])[row["start"] : row["start"] + config.window_size]
        means.append(extract_features(x, config))
    np.testing.assert_allclose(mean, np.mean(means, axis=0))


def test_leakage_rejected(tmp_path, config):
    manifest = create_demo(tmp_path / "data", config, groups_per_class=3)
    rows = read_manifest(manifest)
    assigned = [
        Recording(r.path, r.label, r.group, ("train", "val", "test")[i % 3])
        for i, r in enumerate(rows)
    ]
    assigned[1] = Recording(assigned[1].path, 0, assigned[0].group, "val")
    with pytest.raises(ValueError, match="leakage"):
        split_recordings(assigned)
    rows[1].path.write_bytes(rows[0].path.read_bytes())
    assigned = [
        Recording(r.path, r.label, r.group, ("train", "val", "test")[i % 3])
        for i, r in enumerate(rows)
    ]
    with pytest.raises(ValueError, match="identical"):
        split_recordings(assigned)


def test_export_and_fixed_ann(artifact, tmp_path):
    model, mean, scale, config, _ = load_model(artifact)
    features = np.linspace(0.1, 0.9, 9)
    logit = fixed_ann(quantize(features), model, mean, scale) / 65536
    p = predict(model, features[None], mean, scale)[0]
    assert abs(1 / (1 + np.exp(-logit)) - p) < 0.002
    result = export_fpga(artifact, tmp_path / "rom")
    assert len((tmp_path / "rom/weights.mem").read_text().splitlines()) == 20800
    assert result["board_validated"] is False
    assert result["config"] == config.to_dict()


def test_acs712():
    sensor = ACS712()
    codes = np.array([0, 1861, 4095])
    np.testing.assert_allclose(sensor.convert(codes), codes * sensor.gain - sensor.offset)
    for invalid in ([4096], [-1], [3.5], [np.nan]):
        with pytest.raises(ValueError):
            sensor.convert(invalid)


@pytest.mark.parametrize("underflow", ["adc", "normalizer"])
def test_export_rejects_scale_underflow_before_writing_roms(artifact, tmp_path, underflow):
    calibration = ACS712(adc_bits=24) if underflow == "adc" else ACS712()
    if underflow == "normalizer":
        model, mean, _, config, metadata = load_model(artifact)
        save_model(artifact, model, mean, np.full(9, 1e8), config, metadata)
    destination = tmp_path / "rom-underflow"
    with pytest.raises(ValueError, match="rounds to zero"):
        export_fpga(artifact, destination, calibration=calibration)
    assert not list(destination.glob("*.mem"))
    assert not (destination / "model_config.svh").exists()


def test_protection_latched_and_independent():
    p = ProtectionController(watchdog_seconds=1)
    assert p.update(0)["trip"]
    assert not p.update(0.1, probability=0.1, acknowledge=True)["trip"]
    assert p.update(0.2, vibration=True, acknowledge=True)["trip"]
    assert p.update(0.3)["trip"]
    assert not p.update(0.4, probability=0.1, acknowledge=True)["trip"]
    assert p.update(1.4)["trip"]
    assert p.update(1.5, probability=0.9, acknowledge=True)["trip"]
    assert p.update(1.6, probability=float("nan"), acknowledge=True)["trip"]
    with pytest.raises(ValueError):
        p.update(0.5)


def test_held_acknowledgment_cannot_automatically_rearm():
    p = ProtectionController()
    assert not p.update(0, probability=0.1, acknowledge=True)["trip"]
    assert p.update(1, vibration=True, acknowledge=True)["trip"]
    assert p.update(2, probability=0.1, acknowledge=True)["trip"]
    assert p.update(3, acknowledge=False)["trip"]
    assert not p.update(4, acknowledge=True)["trip"]
