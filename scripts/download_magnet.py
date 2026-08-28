"""Explicitly download the legacy notebook's pretrained MagNet, checking SHA-256."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://github.com/ZhengPeng7/motion_magnification_learning-based/releases/download/v1.0/magnet_epoch12_loss7.28e-02.pth"
SHA256 = "26c0e3ba528152d9219698ca4dac8a1924e3c4528f81277c3296fb7c034d4fae"


def download(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(temporary)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("xb") as stream:
            while chunk := response.read(65536):
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError("Checkpoint checksum mismatch; refusing to install")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(download(parser.parse_args().output))
