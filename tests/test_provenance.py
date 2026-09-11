import pytest

from wds_sentinel.data.provenance import sha256_of_file, verify_checksum


def test_sha256_known_value(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_bytes(b"hello world")
    digest = sha256_of_file(f)
    assert len(digest) == 64
    assert digest == sha256_of_file(f)  # deterministic


def test_verify_checksum_detects_tampering(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_bytes(b"original content")
    good = sha256_of_file(f)
    assert verify_checksum(f, good) is True

    f.write_bytes(b"tampered content")
    assert verify_checksum(f, good) is False


def test_verify_checksum_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        verify_checksum(missing, "deadbeef")
