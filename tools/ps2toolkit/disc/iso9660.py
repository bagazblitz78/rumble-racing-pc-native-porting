"""ISO9660 (+CD-XA marker) volume parser for PS2 disc images (task T1.3).

Reads volume descriptors from LBA 16 onward, exposes the root directory,
and supports listing/reading files and directories by path. PS2 discs are
ISO9660 with a Supplementary Volume Descriptor identified by the
"PLAYSTATION" system identifier, which this parser detects.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .sector import RawDiscImage

VD_LBA_START = 16


class ISOError(Exception):
    """Raised when the image is not a valid ISO9660 volume."""


@dataclass
class ISOFile:
    name: str  # raw ISO name, e.g. "SLUS_203.36;1"
    lba: int
    size: int
    is_dir: bool


def _both16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _both32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _decode_name(raw: bytes) -> str:
    return raw.decode("latin-1").rstrip("\x00 ")


def _parse_dir_record(
    data: bytes, offset: int, skip_dot: bool = True
) -> tuple[ISOFile | None, int]:
    """Parse one directory record at offset.

    Returns (entry, record_length). entry is None for zero-length records
    and for "." / ".." entries when skip_dot is set. record_length is 0
    when the record terminates the sector.
    """
    length = data[offset]
    if length == 0:
        return None, 0
    rec = data[offset : offset + length]
    lba = _both32(rec, 2)
    size = _both32(rec, 10)
    flags = rec[25]
    name_len = rec[32]
    name = _decode_name(rec[33 : 33 + name_len])
    if skip_dot and name in ("\x00", "\x01"):
        return None, length  # "." and ".." entries inside directory extents
    return ISOFile(name=name, lba=lba, size=size, is_dir=bool(flags & 0x02)), length


class ISO9660:
    def __init__(self, disc: RawDiscImage):
        self.disc = disc
        self.volume_id = ""
        self.system_id = ""
        self.block_size = 2048
        self.volume_space_size = 0
        self.is_playstation = False
        self.root: ISOFile | None = None
        self._read_volume_descriptors()

    def _read_volume_descriptors(self):
        for lba in range(VD_LBA_START, self.disc.sector_count):
            sector = self.disc.read_user_data(lba)
            if sector[1:6] != b"CD001":
                if self.root is not None:
                    break  # descriptor set ended without terminator; tolerate
                raise ISOError(f"LBA {lba}: missing CD001 signature")
            vd_type = sector[0]
            if vd_type == 0:
                break  # terminator
            if vd_type == 1 and self.root is None:
                self.system_id = _decode_name(sector[8:40]).strip()
                self.volume_id = _decode_name(sector[40:72]).strip()
                if self.system_id == "PLAYSTATION":
                    self.is_playstation = True
                self.volume_space_size = _both32(sector, 80)
                self.block_size = _both16(sector, 128)
                root, _ = _parse_dir_record(sector, 156, skip_dot=False)
                self.root = root
                if self.root is None or not self.root.is_dir:
                    raise ISOError("PVD has no valid root directory record")
            elif vd_type == 2:
                if _decode_name(sector[8:40]).strip() == "PLAYSTATION":
                    self.is_playstation = True
        if self.root is None:
            raise ISOError("no Primary Volume Descriptor found")

    def _read_dir_extent(self, dir_file: ISOFile) -> list[ISOFile]:
        entries: list[ISOFile] = []
        size = dir_file.size
        if dir_file.lba + (size + self.block_size - 1) // self.block_size > self.disc.sector_count:
            raise ISOError(f"directory {dir_file.name} extent out of image bounds")
        offset_in_dir = 0
        while offset_in_dir < size:
            lba = dir_file.lba + offset_in_dir // self.block_size
            sector = self.disc.read_user_data(lba)
            pos = offset_in_dir % self.block_size
            while pos < self.block_size:
                entry, rec_len = _parse_dir_record(sector, pos)
                if rec_len == 0:
                    break
                if entry is not None:
                    entries.append(entry)
                pos += rec_len
            offset_in_dir = (offset_in_dir // self.block_size + 1) * self.block_size
        return entries

    def list_dir(self, dir_file: ISOFile) -> list[ISOFile]:
        if not dir_file.is_dir:
            raise NotADirectoryError(dir_file.name)
        return self._read_dir_extent(dir_file)

    def resolve(self, path: str) -> ISOFile:
        """Resolve a path like 'SUB/INNER.TXT;1' from the root directory."""
        if self.root is None:
            raise ISOError("no root directory")
        current = self.root
        components = [c for c in path.replace("\\", "/").split("/") if c]
        for i, component in enumerate(components):
            if not current.is_dir:
                raise NotADirectoryError("/".join(components[: i + 1]))
            matches = [e for e in self.list_dir(current) if e.name.upper() == component.upper()]
            if not matches:
                raise FileNotFoundError(f"{component!r} not found in disc image")
            current = matches[0]
        return current

    def read_file(self, file: ISOFile) -> bytes:
        if file.is_dir:
            raise IsADirectoryError(file.name)
        if file.lba + (file.size + self.block_size - 1) // self.block_size > self.disc.sector_count:
            raise ISOError(f"file {file.name} extent out of image bounds")
        out = bytearray()
        remaining = file.size
        lba = file.lba
        while remaining > 0:
            sector = self.disc.read_user_data(lba)
            chunk = sector[: min(remaining, self.block_size)]
            out += chunk
            remaining -= len(chunk)
            lba += 1
        return bytes(out)

    def walk(self, dir_file: ISOFile | None = None, prefix: str = ""):
        """Yield (path, ISOFile) for every entry, depth-first."""
        if dir_file is None:
            dir_file = self.root
        for entry in self.list_dir(dir_file):
            path = f"{prefix}/{entry.name}" if prefix else entry.name
            yield path, entry
            if entry.is_dir:
                yield from self.walk(entry, path)
