"""Tests for raw 2352-byte disc sector handling (T1.2)."""
import pytest

from ps2toolkit.disc.sector import (
    RAW_SECTOR_SIZE,
    SyncError,
    decode_bcd,
    encode_bcd,
    extract_user_data,
    lba_to_msf,
    msf_to_lba,
    RawDiscImage,
)


def _cdxa_subheader(file_no=0, channel=0, submode=0x08, coding=0):
    # submode 0x08 = data; duplicated twice per spec
    return bytes([file_no, channel, submode, coding, file_no, channel, submode, coding])


def make_raw_sector(mm, ss, ff, mode=2, form1=True, user_data=b""):
    """Build a raw 2352-byte Mode2 sector (test fixture helper)."""
    header = bytes([encode_bcd(mm), encode_bcd(ss), encode_bcd(ff), mode])
    if mode == 2:
        submode = 0x08 if form1 else 0x08 | 0x40  # data; form2 bit 0x40
        subheader = _cdxa_subheader(submode=submode)
        if form1:
            data = user_data[:2048].ljust(2048, b"\x00")
            trailer = b"\x00" * (4 + 276)  # EDC + ECC placeholder
        else:
            data = user_data[:2324].ljust(2324, b"\x00")
            trailer = b"\x00" * 4  # EDC only
        payload = subheader + data + trailer
    else:  # Mode1/2352
        payload = user_data[:2048].ljust(2048, b"\x00") + b"\x00" * (4 + 8 + 276)
    sector = b"\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00" + header + payload
    assert len(sector) == RAW_SECTOR_SIZE
    return sector


def test_bcd_roundtrip():
    assert encode_bcd(0) == 0x00
    assert encode_bcd(59) == 0x59
    assert encode_bcd(74) == 0x74
    assert decode_bcd(0x59) == 59
    assert decode_bcd(0x74) == 74


def test_msf_lba_math():
    assert msf_to_lba(0, 2, 0) == 0       # MSF 00:02:00 is LBA 0 (150-frame pregap)
    assert msf_to_lba(0, 2, 1) == 1
    assert msf_to_lba(1, 0, 0) == 75 * 60 - 150  # 01:00:00 minus pregap
    assert lba_to_msf(0) == (0, 2, 0)
    assert lba_to_msf(75 * 60 - 150) == (1, 0, 0)
    mm, ss, ff = lba_to_msf(12345)
    assert msf_to_lba(mm, ss, ff) == 12345


def test_extract_form1_user_data():
    sector = make_raw_sector(0, 2, 0, mode=2, form1=True, user_data=b"A" * 2048)
    data = extract_user_data(sector)
    assert len(data) == 2048
    assert data == b"A" * 2048


def test_extract_form2_user_data():
    sector = make_raw_sector(0, 2, 0, mode=2, form1=False, user_data=b"B" * 2324)
    data = extract_user_data(sector)
    assert len(data) == 2324
    assert data == b"B" * 2324


def test_extract_mode1_user_data():
    sector = make_raw_sector(0, 2, 0, mode=1, user_data=b"C" * 2048)
    data = extract_user_data(sector)
    assert len(data) == 2048
    assert data == b"C" * 2048


def test_extract_rejects_bad_size():
    with pytest.raises(ValueError):
        extract_user_data(b"\x00" * 100)


def test_extract_rejects_missing_sync():
    sector = bytearray(make_raw_sector(0, 2, 0, user_data=b"D" * 16))
    sector[1] = 0x00  # break sync pattern
    with pytest.raises(SyncError):
        extract_user_data(bytes(sector))


def test_image_reader(tmp_path):
    sectors = [
        make_raw_sector(0, 2, 0, user_data=b"first-sector-data".ljust(2048, b"\x00")),
        make_raw_sector(0, 2, 1, user_data=b"second-sector-data".ljust(2048, b"\x00")),
        make_raw_sector(0, 2, 2, user_data=b"third-sector-data".ljust(2048, b"\x00")),
    ]
    img = tmp_path / "test.bin"
    img.write_bytes(b"".join(sectors))
    disc = RawDiscImage(img)
    assert disc.sector_count == 3
    assert disc.read_user_data(0).startswith(b"first-sector-data")
    assert disc.read_user_data(2).startswith(b"third-sector-data")
    raw = disc.read_sector(1)
    assert len(raw) == RAW_SECTOR_SIZE
    with pytest.raises(IndexError):
        disc.read_sector(3)
