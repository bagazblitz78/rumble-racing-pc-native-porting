"""PS2 (MIPS R5900) ELF32 parser (task T2.1).

Parses the little-endian ELF32 executables used on PS2 discs
(SLUS_*/SCUS_* etc.), exposes program headers/sections, and can produce a
flat guest memory image from PT_LOAD segments.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

ELF_MAGIC = b"\x7fELF"
EM_MIPS = 8
ET_EXEC = 2
PT_LOAD = 1

EhdrOffsets = struct.Struct  # for readability

_EHDR = struct.Struct("<16sHHIIIIIHHHHHH")
_PHDR = struct.Struct("<IIIIIIII")
_SHDR = struct.Struct("<IIIIIIIIII")


class ELFError(Exception):
    """Raised for malformed or unsupported ELF images."""


@dataclass
class Segment:
    type: int
    offset: int
    vaddr: int
    paddr: int
    filesz: int
    memsz: int
    flags: int
    align: int


@dataclass
class Section:
    name: str
    type: int
    addr: int
    offset: int
    size: int
    flags: int


@dataclass
class MemoryImage:
    """Flat guest memory image covering all loaded segments."""

    base: int
    data: bytearray = field(default_factory=bytearray)

    def read(self, vaddr: int, size: int) -> bytes:
        off = vaddr - self.base
        if off < 0 or off + size > len(self.data):
            raise ELFError(f"read 0x{vaddr:X}+{size} outside memory image")
        return bytes(self.data[off : off + size])

    def write(self, vaddr: int, blob: bytes):
        off = vaddr - self.base
        if off < 0 or off + len(blob) > len(self.data):
            raise ELFError(f"write 0x{vaddr:X}+{len(blob)} outside memory image")
        self.data[off : off + len(blob)] = blob


class PS2ELF:
    def __init__(self, data: bytes):
        if len(data) < _EHDR.size:
            raise ELFError("file too small for ELF header")
        (
            ident,
            self.type,
            self.machine,
            self.version,
            self.entry,
            self.phoff,
            self.shoff,
            self.flags,
            self.ehsize,
            self.phentsize,
            self.phnum,
            self.shentsize,
            self.shnum,
            self.shstrndx,
        ) = _EHDR.unpack_from(data, 0)
        if ident[:4] != ELF_MAGIC:
            raise ELFError("not an ELF file (bad magic)")
        if ident[4] != 1 or ident[5] != 1:
            raise ELFError("only ELF32 little-endian images are supported")
        if self.machine != EM_MIPS:
            raise ELFError(f"not a MIPS ELF (e_machine={self.machine})")
        self.data = data
        self.segments: list[Segment] = []
        self.sections: list[Section] = []
        self._parse_segments()
        self._parse_sections()

    @classmethod
    def from_bytes(cls, data: bytes) -> "PS2ELF":
        return cls(data)

    @classmethod
    def from_file(cls, path: str) -> "PS2ELF":
        with open(path, "rb") as fh:
            return cls(fh.read())

    def _parse_segments(self):
        if self.phoff == 0 or self.phnum == 0:
            return
        for i in range(self.phnum):
            off = self.phoff + i * self.phentsize
            if off + _PHDR.size > len(self.data):
                raise ELFError("program header out of bounds")
            fields = _PHDR.unpack_from(self.data, off)
            self.segments.append(Segment(*fields))

    def _parse_sections(self):
        if self.shoff == 0 or self.shnum == 0:
            return
        raw: list[tuple] = []
        for i in range(self.shnum):
            off = self.shoff + i * self.shentsize
            if off + _SHDR.size > len(self.data):
                raise ELFError("section header out of bounds")
            raw.append(_SHDR.unpack_from(self.data, off))
        # resolve names via shstrtab
        strtab = b""
        if self.shstrndx < len(raw):
            _, _, _, _, stroff, strsize, *_ = raw[self.shstrndx]
            strtab = self.data[stroff : stroff + strsize]
        for nameoff, s_type, _flags, addr, offset, size, *_rest in raw:
            end = strtab.find(b"\x00", nameoff)
            name = strtab[nameoff:end].decode("latin-1") if 0 <= nameoff < len(strtab) else ""
            self.sections.append(Section(name, s_type, addr, offset, size, _flags))

    def section_by_name(self, name: str) -> Section | None:
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def load_memory_image(self) -> MemoryImage:
        """Build the guest memory image from PT_LOAD segments (BSS zero-filled)."""
        loads = [s for s in self.segments if s.type == PT_LOAD]
        if not loads:
            raise ELFError("no PT_LOAD segments")
        base = min(s.vaddr for s in loads)
        end = max(s.vaddr + s.memsz for s in loads)
        image = MemoryImage(base=base, data=bytearray(end - base))
        for seg in loads:
            payload = self.data[seg.offset : seg.offset + seg.filesz]
            if len(payload) != seg.filesz:
                raise ELFError("segment data truncated")
            image.write(seg.vaddr, payload)
            # remainder up to memsz stays zero (BSS)
        return image
