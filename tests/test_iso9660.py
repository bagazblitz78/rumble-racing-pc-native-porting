"""Tests for ISO9660/XA volume and directory parsing (T1.3)."""
import pytest

from ps2toolkit.disc.sector import RawDiscImage
from ps2toolkit.disc.iso9660 import ISO9660, ISOError

from tests.disc_fixtures import (
    HELLO_CONTENT,
    HELLO_LBA,
    INNER_CONTENT,
    build_test_disc,
)


@pytest.fixture()
def iso(tmp_path):
    img = tmp_path / "test.bin"
    img.write_bytes(build_test_disc())
    return ISO9660(RawDiscImage(img))


def test_volume_identity(iso):
    assert iso.volume_id == "TESTVOL"
    assert iso.block_size == 2048
    assert iso.volume_space_size == 24


def test_playstation_svd_detected(iso):
    assert iso.is_playstation is True


def test_playstation_via_pvd_only(tmp_path):
    # Some real discs (e.g. Rumble Racing) carry PLAYSTATION in the PVD
    img = tmp_path / "pvd.bin"
    img.write_bytes(build_test_disc(pvd_system=b"PLAYSTATION", svd_system=b"CDROM"))
    iso = ISO9660(RawDiscImage(img))
    assert iso.is_playstation is True


def test_root_listing(iso):
    entries = {f.name: f for f in iso.list_dir(iso.root)}
    assert "HELLO.TXT;1" in entries
    assert "SUB" in entries
    assert entries["HELLO.TXT;1"].is_dir is False
    assert entries["HELLO.TXT;1"].size == len(HELLO_CONTENT)
    assert entries["HELLO.TXT;1"].lba == HELLO_LBA
    assert entries["SUB"].is_dir is True


def test_read_root_file(iso):
    f = iso.resolve("HELLO.TXT;1")
    assert iso.read_file(f) == HELLO_CONTENT


def test_subdir_listing_and_read(iso):
    sub = iso.resolve("SUB")
    assert sub.is_dir is True
    entries = {f.name: f for f in iso.list_dir(sub)}
    assert "INNER.TXT;1" in entries
    inner = iso.resolve("SUB/INNER.TXT;1")
    assert iso.read_file(inner) == INNER_CONTENT


def test_resolve_missing_raises(iso):
    with pytest.raises(FileNotFoundError):
        iso.resolve("NOPE.TXT;1")


def test_resolve_into_file_raises(iso):
    with pytest.raises(NotADirectoryError):
        iso.resolve("HELLO.TXT;1/SUB")


def test_bad_signature_raises(tmp_path):
    from tests.disc_fixtures import build_test_disc
    data = bytearray(build_test_disc())
    # corrupt CD001 signature in PVD (LBA 16 user data offset 1..5)
    # user data starts at raw offset 16*2352 + 24
    off = 16 * 2352 + 24
    data[off + 1 : off + 6] = b"XXXXX"
    img = tmp_path / "bad.bin"
    img.write_bytes(bytes(data))
    with pytest.raises(ISOError):
        ISO9660(RawDiscImage(img))
