"""Tests for src/download/boundaries.py. The network is replaced by a fake.

Run from the repo root:  python -m pytest tests/test_boundaries.py -v
"""
import pytest
import requests

from src.download import boundaries


class FakeReply:
    def __init__(self, status, payload=None, content=b""):
        self.status_code, self._payload, self.content = status, payload, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


def test_level_missing_in_geoboundaries_is_skipped_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(boundaries.requests, "get", lambda url, timeout: FakeReply(404))
    boundaries.download("NGA", "ADM3", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_other_http_errors_still_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(boundaries.requests, "get", lambda url, timeout: FakeReply(500))
    with pytest.raises(requests.HTTPError):
        boundaries.download("NGA", "ADM2", tmp_path)


def test_level_is_downloaded_and_written(tmp_path, monkeypatch):
    def fake_get(url, timeout):
        if "geoboundaries" in url:
            return FakeReply(200, {"gjDownloadURL": "http://x/file.geojson", "boundaryLicense": "CC BY 4.0"})
        return FakeReply(200, content=b"{}")
    monkeypatch.setattr(boundaries.requests, "get", fake_get)
    boundaries.download("KEN", "ADM2", tmp_path)
    assert (tmp_path / "KEN_ADM2.geojson").read_bytes() == b"{}"
