"""Raw CD/DVD sector handling for PS2 disc images (task T1.2).

Supports raw 2352-byte sectors (MODE1/2352, MODE2/2352 with CD-XA
Form1/Form2) and plain 2048-byte ISO sectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAW_SECTOR_SIZE = 2352
ISO_SECTOR_SIZE = 2048
SYNC_PATTERN = b"\x00" + b"\xff" * 10 + b"\x00"
PREGAP_FRAMES = 150  # 2 seconds of pregap (MSF 00:02:00) before LBA 0
FRAMES_PER_SECOND = 75


class SyncError(ValueError):
    """Raised when a raw sector lacks the 12-byte sync pattern."""


def encode_bcd(value: int) -> int:
    """Encode a 0-99 integer as BCD (used in sector headers)."""
    if not 0 <= value <= 99:
        raise ValueError(f"BCD value out of range: {value}")
    return ((value // 10) << 4) | (value % 10)


def decode_bcd(value: int) -> int:
    """Decode a BCD byte to an integer."""
    return (value >> 4) * 10 + (value & 0x0F)


def msf_to_lba(mm: int, ss: int, ff: int) -> int:
    """Absolute MSF (as written in sector headers) to logical block address."""
    return (mm * 60 + ss) * FRAMES_PER_SECOND + ff - PREGAP_FRAMES


def lba_to_msf(lba: int) -> tuple[int, int, int]:
    """Logical block address to absolute MSF (as written in sector headers)."""
    frames = lba + PREGAP_FRAMES
    ff = frames % FRAMES_PER_SECOND
    seconds = frames // FRAMES_PER_SECOND
    ss = seconds % 60
    mm = seconds // 60
    return mm, ss, ff


@dataclass
class XASubheader:
    file_number: int
    channel: int
    submode: int
    coding_info: int

    @property
    def is_form2(self) -> bool:
        return bool(self.submode & 0x40)

    @property
    def is_data(self) -> bool:
        return bool(self.submode & 0x08)


def extract_user_data(sector: bytes) -> bytes:
    """Extract the user-data payload from one raw sector.

    - Mode1/2352: 2048 bytes at offset 16.
    - Mode2/2352 Form1 (CD-XA): 2048 bytes at offset 24 (sync+header+subheader).
    - Mode2/2352 Form2 (CD-XA): 2324 bytes at offset 24.
    - Plain 2048-byte ISO sector: returned as-is.
    """
    if len(sector) == ISO_SECTOR_SIZE:
        return sector
    if len(sector) != RAW_SECTOR_SIZE:
        raise ValueError(f"unexpected sector size {len(sector)}")
    if sector[:12] != SYNC_PATTERN:
        raise SyncError("sector does not start with CD sync pattern")
    mode = sector[15]
    if mode == 1:
        return sector[16 : 16 + 2048]
    if mode == 2:
        submode = sector[16 + 2]  # third byte of duplicated XA subheader
        if submode & 0x40:  # Form2 bit
            return sector[24 : 24 + 2324]
        return sector[24 : 24 + 2048]
    raise ValueError(f"unsupported sector mode {mode}")


class RawDiscImage:
    """Random-access reader over a raw sector disc image."""

    def __init__(self, path: str | Path, sector_size: int = RAW_SECTOR_SIZE):
        self.path = Path(path)
        if sector_size not in (RAW_SECTOR_SIZE, ISO_SECTOR_SIZE):
            raise ValueError(f"unsupported sector size {sector_size}")
        self.sector_size = sector_size
        file_size = self.path.stat().st_size
        if file_size % sector_size != 0:
            raise ValueError(
                f"{self.path}: size {file_size} is not a multiple of {sector_size}"
            )
        self.sector_count = file_size // sector_size
        self._fh = None

    def _handle(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")
        return self._fh

    def read_sector(self, lba: int) -> bytes:
        """Read one raw sector by logical block address."""
        if not 0 <= lba < self.sector_count:
            raise IndexError(f"LBA {lba} out of range (0..{self.sector_count - 1})")
        fh = self._handle()
        fh.seek(lba * self.sector_size)
        data = fh.read(self.sector_size)
        if len(data) != self.sector_size:
            raise IOError(f"short read at LBA {lba}")
        return data

    def read_user_data(self, lba: int) -> bytes:
        """Read and de-sectorize one sector's user data."""
        return extract_user_data(self.read_sector(lba))

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
