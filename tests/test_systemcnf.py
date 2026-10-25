"""Tests for SYSTEM.CNF parsing and boot-ELF location (T1.5)."""
import pytest

from ps2toolkit.disc.iso9660 import ISO9660
from ps2toolkit.disc.sector import RawDiscImage
from ps2toolkit.disc.systemcnf import find_boot_elf, parse_system_cnf, strip_disc_path

from tests.disc_fixtures import FAKE_ELF, build_disc_with_boot

SYSTEM_CNF = """BOOT2 = cdrom0:\\SLUS_201.74;1
VER = 1.00
VMODE = NTSC
"""


def test_parse_system_cnf():
    cfg = parse_system_cnf(SYSTEM_CNF)
    assert cfg["BOOT2"] == "cdrom0:\\SLUS_201.74;1"
    assert cfg["VER"] == "1.00"
    assert cfg["VMODE"] == "NTSC"


def test_parse_system_cnf_forward_slash():
    cfg = parse_system_cnf("BOOT2 = cdrom0:/SYSTEM.CNF;1\n")
    assert cfg["BOOT2"] == "cdrom0:/SYSTEM.CNF;1"


def test_strip_disc_path():
    assert strip_disc_path("cdrom0:\\SLUS_201.74;1") == "SLUS_201.74;1"
    assert strip_disc_path("cdrom0:/SLUS_201.74;1") == "SLUS_201.74;1"
    assert strip_disc_path("cdrom0:\\subdir\\GAME.ELF;1") == "subdir/GAME.ELF;1"
    assert strip_disc_path("SLUS_201.74;1") == "SLUS_201.74;1"


@pytest.fixture()
def iso(tmp_path):
    img = tmp_path / "test.bin"
    img.write_bytes(build_disc_with_boot())
    return ISO9660(RawDiscImage(img))


def test_find_boot_elf(iso):
    entry, offset = find_boot_elf(iso)
    assert entry.name == "SLUS_201.74;1"
    assert offset == 0
    data = iso.read_file(entry)
    assert data[offset : offset + 4] == b"\x7fELF"


def test_find_boot_elf_with_header_offset(tmp_path):
    # variant disc: ELF magic at 0x800 within the boot file (some PS2 discs
    # prefix the ELF with a 0x800-byte header)
    from tests.disc_fixtures import build_disc_with_boot
    img = tmp_path / "b.bin"
    img.write_bytes(build_disc_with_boot(elf_offset=0x800))
    iso2 = ISO9660(RawDiscImage(img))
    entry, offset = find_boot_elf(iso2)
    assert entry.name == "SLUS_201.74;1"
    assert offset == 0x800
