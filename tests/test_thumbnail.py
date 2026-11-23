import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for name in ("cv2", "imageio"):
    sys.modules.setdefault(name, types.ModuleType(name))

import main  # noqa: E402


class FakeItem:
    def __init__(self):
        self.id = "item-1"
        self.datasetId = "dataset-1"
        self.metadata = {"system": {}}
        self.updates = 0

    def update(self, system_metadata=False):
        self.updates += 1
        return self


@pytest.fixture
def runner():
    return main.VideoPreprocess.__new__(main.VideoPreprocess)


def test_file_above_thumbnail_limit_is_skipped_without_failing(runner, tmp_path, monkeypatch):
    video = tmp_path / "big.mp4"
    video.write_bytes(b"0")
    monkeypatch.setattr(main.os.path, "getsize", lambda _: (main.MAX_GEN_THUMB_SIZE_MB + 1) * 1024 * 1024)
    monkeypatch.setattr(runner, "_build_gif", lambda *a: pytest.fail("GIF must not be built for large files"))
    item = FakeItem()

    runner._generate_thumbnail(item, str(video), str(tmp_path), 128)

    etl = item.metadata["system"]["etl"]
    assert [e["stage"] for e in etl["errors"]] == ["thumbnail_size"]
    assert "failed" not in etl
    assert "videoEtl" not in item.metadata["system"]
    assert "thumbnailId" not in item.metadata["system"]
    assert item.updates == 1


def test_thumbnail_build_error_still_fails(runner, tmp_path):
    video = tmp_path / "small.mp4"
    video.write_bytes(b"0")

    def broken(*_):
        raise RuntimeError("decode error")

    runner._build_gif = broken
    item = FakeItem()

    with pytest.raises(RuntimeError, match="decode error"):
        runner._generate_thumbnail(item, str(video), str(tmp_path), 128)
    assert item.metadata["system"]["etl"]["failed"] is True
