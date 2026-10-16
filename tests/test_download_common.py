"""Tests for the resumable download helper. The network is replaced by a fake server.

Run from the repo root:  python -m pytest tests/test_download_common.py -v
"""
import json

import pytest
import requests

from src.download import common

DATA = bytes(range(256)) * 40          # 10,240 bytes of known content


class FakeResponse:
    def __init__(self, body: bytes, status: int, fail_after: int = None):
        self.status_code, self._body, self._fail_after = status, body, fail_after
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 416:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for start in range(0, len(self._body), 1000):
            if self._fail_after is not None and start >= self._fail_after:
                raise requests.ConnectionError("connection dropped")
            yield self._body[start:start + 1000]


def fake_server(monkeypatch, fail_first_after=None, support_range=True):
    """Serve DATA. The first request drops after `fail_first_after` bytes. Returns the list of Range headers seen."""
    seen = []

    def fake_get(url, stream, timeout, headers):
        seen.append(headers.get("Range"))
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        if headers and not support_range:
            return FakeResponse(DATA, 200)
        fail = fail_first_after if len(seen) == 1 else None
        return FakeResponse(DATA[start:], 206 if headers else 200, fail_after=fail)

    monkeypatch.setattr(common.requests, "get", fake_get)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    return seen


def test_plain_download_writes_file_and_metadata(tmp_path, monkeypatch):
    fake_server(monkeypatch)
    out = tmp_path / "f.bin"
    common.stream_to_file("http://x/f", out, {"source": "test", "licence": "CC BY 4.0"})
    assert out.read_bytes() == DATA
    meta = json.loads((tmp_path / "f.bin.json").read_text())
    assert meta["source"] == "test" and meta["url"] == "http://x/f" and "downloaded" in meta
    assert not (tmp_path / "f.bin.part").exists()


def test_dropped_connection_resumes_and_gives_identical_file(tmp_path, monkeypatch):
    seen = fake_server(monkeypatch, fail_first_after=4000)
    out = tmp_path / "f.bin"
    common.stream_to_file("http://x/f", out, {})
    assert out.read_bytes() == DATA                      # no gaps and no repeated bytes
    assert seen[0] is None and seen[1] == "bytes=4000-"  # second request resumed where the first stopped


def test_existing_part_file_is_resumed_on_a_new_run(tmp_path, monkeypatch):
    seen = fake_server(monkeypatch)
    (tmp_path / "f.bin.part").write_bytes(DATA[:3000])
    out = tmp_path / "f.bin"
    common.stream_to_file("http://x/f", out, {})
    assert out.read_bytes() == DATA and seen == ["bytes=3000-"]


def test_server_without_range_support_restarts_cleanly(tmp_path, monkeypatch):
    fake_server(monkeypatch, support_range=False)
    (tmp_path / "f.bin.part").write_bytes(DATA[:3000])
    out = tmp_path / "f.bin"
    common.stream_to_file("http://x/f", out, {})
    assert out.read_bytes() == DATA


def test_gives_up_after_the_retry_limit_and_keeps_the_part_file(tmp_path, monkeypatch):
    def always_fail(url, stream, timeout, headers):
        raise requests.Timeout("timed out")
    monkeypatch.setattr(common.requests, "get", always_fail)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    (tmp_path / "f.bin.part").write_bytes(DATA[:500])
    with pytest.raises(IOError):
        common.stream_to_file("http://x/f", tmp_path / "f.bin", {}, retries=3)
    assert not (tmp_path / "f.bin").exists()
    assert (tmp_path / "f.bin.part").read_bytes() == DATA[:500]


def test_finished_file_is_skipped(tmp_path, monkeypatch):
    seen = fake_server(monkeypatch)
    (tmp_path / "f.bin").write_bytes(b"already here")
    common.stream_to_file("http://x/f", tmp_path / "f.bin", {})
    assert seen == [] and (tmp_path / "f.bin").read_bytes() == b"already here"
