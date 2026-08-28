import csv

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")
from dwt_ann_mav.magnet import MagNet
from dwt_ann_mav.video import load_magnet, magnify_pair, magnify_video, train_magnet


def test_pair_shapes_and_identity():
    torch.set_num_threads(1)
    model = MagNet().eval()
    frame = np.full((33, 35, 3), 120, np.uint8)
    np.testing.assert_array_equal(magnify_pair(model, frame, frame, 0), frame)
    output = magnify_pair(model, frame, frame, 2)
    assert output.shape == frame.shape and output.dtype == np.uint8
    with pytest.raises(ValueError):
        magnify_pair(model, frame, frame, float("nan"))


def test_stream_preserves_fps_count_and_dimensions(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "model.pth"
    torch.save(MagNet().state_dict(), checkpoint)
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 17, (32, 32))
    assert writer.isOpened()
    for i in range(5):
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[8:20, 5 + i : 15 + i] = 180
        writer.write(frame)
    writer.release()
    output = tmp_path / "output.avi"
    report = magnify_video(source, output, checkpoint, alpha=1, side_by_side=True)
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FPS) == 17
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (32, 64, 3)
        count += 1
    capture.release()
    assert count == report["frames"] == 5
    assert report["audio_preserved"] is False
    with pytest.raises(FileExistsError):
        magnify_video(source, source, checkpoint)


def test_mav_training_and_reload(tmp_path):
    torch.set_num_threads(1)
    frame = np.random.default_rng(5).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "image.png"), frame)
    manifest = tmp_path / "mav.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["a", "b", "c", "target", "alpha"])
        writer.writerow(["image.png"] * 4 + [1])
    history = train_magnet(manifest, tmp_path / "trained.pth", epochs=1)
    assert np.isfinite(history[0]["loss"])
    assert isinstance(load_magnet(tmp_path / "trained.pth"), MagNet)
