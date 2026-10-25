"""Tests for the PS2 (R5900) ELF parser (T2.1)."""
import struct

import pytest

from ps2toolkit.elf.ps2elf import PS2ELF, ELFError

EM_MIPS = 8
PT_LOAD = 1
SHT_PROGBITS = 1
SHT_STRTAB = 3


def build_elf32(
    entry=0x100000,
    segments=None,
    sections=None,
    machine=EM_MIPS,
    flags=0x00000000,
) -> bytes:
    """Build a minimal little-endian ELF32 (R5900) image for tests."""
    if segments is None:
        segments = [(PT_LOAD, 0x1000, 0x00100000, 0x80, 0x100, 5)]
    if sections is None:
        sections = [
            (b"", 0, 0, 0, 0, 0, 0),
            (b".text", SHT_PROGBITS, 0x00100000, 0x1000, 0x80, 0x6, 0),
            (b".shstrtab", SHT_STRTAB, 0, 0, 0, 0, 0),
        ]
    ehsize, phentsize, shentsize = 52, 32, 40
    phoff = ehsize
    shoff = phoff + phentsize * len(segments)
    # section data area starts after headers
    data_off = shoff + shentsize * len(sections)

    shstr = b"\x00"
    name_offs = []
    for name, *_ in sections:
        name_offs.append(len(shstr))
        shstr += name + b"\x00"

    header = struct.pack(
        "<16sHHIIIIIHHHHHH",
        bytes([0x7F, 0x45, 0x4C, 0x46, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        2,  # ET_EXEC
        machine,
        1,  # version
        entry,
        phoff,
        shoff,
        flags,
        ehsize,
        phentsize,
        len(segments),
        shentsize,
        len(sections),
        len(sections) - 1,  # shstrndx
    )
    body = bytearray()
    ph_bytes = b""
    for p_type, offset, vaddr, filesz, memsz, align in segments:
        ph_bytes += struct.pack("<IIIIIIII", p_type, offset, vaddr, vaddr, filesz, memsz, 5, align)
    sh_bytes = b""
    strtab_index = len(sections) - 1
    for i, (name, s_type, addr, offset, size, flags_s, link) in enumerate(sections):
        sh_bytes += struct.pack(
            "<IIIIIIIIII",
            name_offs[i],
            s_type,
            flags_s,
            addr,
            offset,
            size,
            0,  # link (shstrndx for strtab)
            0,  # info
            1,
            0,
        )
    # file data: headers then segment payload at segment offset
    max_off = max((s[1] + s[3] for s in segments), default=data_off)
    file_data = bytearray(max_off)
    file_data[: len(header)] = header
    file_data[phoff : phoff + len(ph_bytes)] = ph_bytes
    file_data[shoff : shoff + len(sh_bytes)] = sh_bytes
    # place distinct payload bytes at each segment offset
    for i, (p_type, offset, vaddr, filesz, memsz, align) in enumerate(segments):
        payload = bytes([(i + 1) * 16 + j % 16 for j in range(filesz)])
        file_data[offset : offset + filesz] = payload
    # append the section-name string table and patch its shdr offset/size
    strtab_off = len(file_data)
    file_data += shstr
    last_shdr_off = shoff + shentsize * strtab_index
    struct.pack_into("<I", file_data, last_shdr_off + 16, strtab_off)  # sh_offset
    struct.pack_into("<I", file_data, last_shdr_off + 20, len(shstr))  # sh_size
    return bytes(file_data)


def test_parse_header():
    elf = PS2ELF.from_bytes(build_elf32())
    assert elf.entry == 0x00100000
    assert elf.machine == EM_MIPS
    assert len(elf.segments) == 1
    seg = elf.segments[0]
    assert seg.type == PT_LOAD
    assert seg.vaddr == 0x00100000
    assert seg.filesz == 0x80
    assert seg.memsz == 0x100


def test_parse_sections():
    elf = PS2ELF.from_bytes(build_elf32())
    names = [s.name for s in elf.sections]
    assert names == ["", ".text", ".shstrtab"]
    text = elf.section_by_name(".text")
    assert text.addr == 0x00100000
    assert text.size == 0x80


def test_load_memory_image():
    elf = PS2ELF.from_bytes(build_elf32())
    image = elf.load_memory_image()
    assert image.base == 0x00100000
    data = image.read(0x00100000, 0x80)
    assert len(data) == 0x80
    assert data[0] == 0x10  # first segment payload marker


def test_load_memory_image_zero_fills_bss():
    segs = [(PT_LOAD, 0x1000, 0x00100000, 0x40, 0x100, 5)]  # filesz 0x40 < memsz 0x100
    elf = PS2ELF.from_bytes(build_elf32(segments=segs))
    image = elf.load_memory_image()
    assert image.read(0x00100000 + 0x40, 0x10) == b"\x00" * 0x10


def test_rejects_bad_magic():
    with pytest.raises(ELFError):
        PS2ELF.from_bytes(b"NOT_ELF" + b"\x00" * 100)


def test_rejects_truncated():
    with pytest.raises(ELFError):
        PS2ELF.from_bytes(b"\x7fELF" + b"\x00" * 10)


def test_rejects_non_mips():
    with pytest.raises(ELFError):
        PS2ELF.from_bytes(build_elf32(machine=3))  # EM_386
