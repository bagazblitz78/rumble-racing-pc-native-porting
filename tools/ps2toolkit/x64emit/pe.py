"""Minimal Windows x64 (PE32+) executable writer (task T2.6).

Builds standalone console PE executables from a caller-supplied .text
payload plus an automatic import table (KERNEL32.dll!ExitProcess by
default), so recompiled PS2 stubs can be launched natively for testing.

Design decisions (kept deliberately minimal but loadable by Windows):

- One ".text" section holding everything: payload code, the import
  structures, and a zero-filled bss tail. The section carries
  CNT_CODE|MEM_EXECUTE|MEM_READ|MEM_WRITE (0xE0000060) so callers may
  store data into it (context structures, patched addresses) without a
  second section.
- The import structures (descriptors, ILT, IAT, hint/name and DLL-name
  strings) are appended automatically AFTER the bytes returned by
  build_text(); build_text must not assume where they sit - use
  layout.iat_rva to reach the IAT slots.
- build_text() is invoked twice: a first probe call to measure the
  payload length (the final RVAs depend on it; the probe Layout carries
  placeholder IAT values of 0 for the requested imports), then a second
  call with the resolved Layout. It must return the same number of bytes
  both times.
- The bss tail is NOT stored in the file: .text's VirtualSize covers
  text+imports+bss while SizeOfRawData covers only the stored bytes, so
  the loader zero-fills the tail (a 32MB bss costs no file bytes).
- No relocation table and DllCharacteristics=0 (no ASLR): the image must
  load at its preferred base 0x140000000, which is virtually always free
  for short-lived test executables.
"""
from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

IMAGE_BASE = 0x140000000
SECTION_ALIGNMENT = 0x1000
FILE_ALIGNMENT = 0x200

_DOS_SIZE = 0x80  # e_lfanew: PE signature sits right behind the DOS header
_MACHINE_AMD64 = 0x8664
_FILE_EXECUTABLE = 0x22  # IMAGE_FILE_EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
_SUBSYSTEM_CONSOLE = 3
_SECTION_CODE = 0xE0000060  # CNT_CODE | MEM_EXECUTE | MEM_READ | MEM_WRITE
_SIZE_OF_OPT64 = 240  # 112 fixed bytes + 16 data directories
_N_DIRS = 16
_IMPORT_DIR = 1
_IAT_DIR = 12
_THUNK = 8  # PE32+ thunk size
_DESC = 20  # IMAGE_IMPORT_DESCRIPTOR size

_COFF = struct.Struct("<HHIIIHH")
_OPT64 = struct.Struct("<HBBIIIIIQIIHHHHHHIIIIHHQQQQII")
_SECHDR = struct.Struct("<8sIIIIIIHHI")
_DESCS = struct.Struct("<IIIII")
_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_U16 = struct.Struct("<H")
_II = struct.Struct("<II")


def _align(value: int, alignment: int) -> int:
    """Round value up to a multiple of alignment (both powers of two)."""
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass
class Layout:
    """Addresses handed to the .text builder before the file is assembled."""

    image_base: int
    text_rva: int
    text_vsize: int
    bss_rva: int
    bss_size: int
    entry_rva: int
    iat_rva: dict[tuple[str, str], int]  # (dll, function) -> IAT slot RVA


def build_console_exe(
    build_text: Callable[[Layout], bytes],
    bss_size: int,
    imports: list[tuple[str, str]] = [("KERNEL32.DLL", "ExitProcess")],
    entry_text_offset: int = 0,
) -> bytes:
    """Build a standalone Windows x64 console PE executable.

    build_text(layout) returns the raw bytes of the .text payload; the
    entry point is the payload bytes at entry_text_offset. The import
    structures are appended after the payload inside .text (build_text
    must locate its IAT slots through layout.iat_rva, not by assuming a
    position). build_text is called twice - a probe call to measure the
    payload length, then the final call with the resolved layout - and
    must return the same number of bytes both times. bss_size bytes of
    zero-filled, writable memory follow the payload in memory
    (layout.bss_rva) but are never stored in the file.
    """
    if not imports:
        raise ValueError("at least one import is required")
    if bss_size < 0:
        raise ValueError("bss_size must not be negative")
    if entry_text_offset < 0:
        raise ValueError("entry_text_offset must not be negative")

    text_rva = SECTION_ALIGNMENT

    # Group imports per DLL, preserving order; duplicate pairs import once.
    groups: dict[str, list[str]] = {}
    for dll, func in imports:
        funcs = groups.setdefault(dll, [])
        if func not in funcs:
            funcs.append(func)

    # Probe call: measure the payload. Final RVAs are unknown yet, so the
    # probe layout carries placeholder IAT values (0) for every import.
    probe = Layout(
        image_base=IMAGE_BASE,
        text_rva=text_rva,
        text_vsize=0,
        bss_rva=0,
        bss_size=bss_size,
        entry_rva=text_rva + entry_text_offset,
        iat_rva={(dll, func): 0 for dll, funcs in groups.items() for func in funcs},
    )
    text_len = len(build_text(probe))
    if not 0 <= entry_text_offset < text_len:
        raise ValueError(
            f"entry_text_offset {entry_text_offset} outside payload of "
            f"{text_len} bytes")

    # Import structures follow the payload inside .text.
    off = text_len
    desc_off = off
    off += _DESC * (len(groups) + 1)  # + null terminator descriptor
    ilt_off: dict[str, int] = {}
    for dll, funcs in groups.items():
        ilt_off[dll] = off
        off += _THUNK * (len(funcs) + 1)  # + null terminator thunk
    iat_off: dict[str, int] = {}
    for dll, funcs in groups.items():
        iat_off[dll] = off
        off += _THUNK * (len(funcs) + 1)
    hint_off: dict[tuple[str, str], int] = {}
    for dll, funcs in groups.items():
        for func in funcs:
            off = _align(off, 2)  # hint/name entries are word aligned
            hint_off[(dll, func)] = off
            off += 2 + len(func.encode("ascii")) + 1
    dll_off: dict[str, int] = {}
    for dll in groups:
        dll_off[dll] = off
        off += len(dll.encode("ascii")) + 1
    stored_len = off  # payload + import structures

    # Fill the import structures (bytearray zero-fills the terminators).
    blob = bytearray(stored_len - text_len)
    for i, (dll, funcs) in enumerate(groups.items()):
        _DESCS.pack_into(
            blob, desc_off - text_len + _DESC * i,
            text_rva + ilt_off[dll],   # OriginalFirstThunk (ILT)
            0, 0,                      # TimeDateStamp, ForwarderChain
            text_rva + dll_off[dll],   # Name
            text_rva + iat_off[dll],   # FirstThunk (IAT)
        )
    iat_rva_map: dict[tuple[str, str], int] = {}
    for dll, funcs in groups.items():
        for j, func in enumerate(funcs):
            name_rva = text_rva + hint_off[(dll, func)]
            _U64.pack_into(blob, ilt_off[dll] - text_len + _THUNK * j, name_rva)
            _U64.pack_into(blob, iat_off[dll] - text_len + _THUNK * j, name_rva)
            iat_rva_map[(dll, func)] = text_rva + iat_off[dll] + _THUNK * j
    for (dll, func), pos in hint_off.items():
        p = pos - text_len
        name = func.encode("ascii")
        blob[p:p + 2 + len(name) + 1] = b"\x00\x00" + name + b"\x00"
    for dll, pos in dll_off.items():
        p = pos - text_len
        name = dll.encode("ascii")
        blob[p:p + len(name) + 1] = name + b"\x00"

    # Final layout: virtual size covers payload+imports+bss, but only the
    # stored bytes go to disk (the loader zero-fills the bss tail).
    vsize = stored_len + bss_size
    rawsize = _align(stored_len, FILE_ALIGNMENT)
    layout = Layout(
        image_base=IMAGE_BASE,
        text_rva=text_rva,
        text_vsize=vsize,
        bss_rva=text_rva + stored_len,
        bss_size=bss_size,
        entry_rva=text_rva + entry_text_offset,
        iat_rva=iat_rva_map,
    )
    text = build_text(layout)
    if len(text) != text_len:
        raise ValueError(
            f"build_text returned {len(text)} bytes on the final call but "
            f"{text_len} on the probe call; the payload length must not "
            "depend on the layout")

    size_of_image = _align(text_rva + vsize, SECTION_ALIGNMENT)
    size_of_headers = _align(
        _DOS_SIZE + 4 + _COFF.size + _SIZE_OF_OPT64 + _SECHDR.size,
        FILE_ALIGNMENT)

    dos = bytearray(_DOS_SIZE)
    dos[0:2] = b"MZ"
    _U32.pack_into(dos, 0x3C, _DOS_SIZE)  # e_lfanew
    coff = _COFF.pack(
        _MACHINE_AMD64, 1, 0, 0, 0, _SIZE_OF_OPT64, _FILE_EXECUTABLE)
    opt = _OPT64.pack(
        0x20B,        # Magic (PE32+)
        14, 0,        # linker version (arbitrary)
        rawsize,      # SizeOfCode
        0,            # SizeOfInitializedData
        bss_size,     # SizeOfUninitializedData (the bss tail)
        layout.entry_rva,
        text_rva,     # BaseOfCode
        IMAGE_BASE,
        SECTION_ALIGNMENT,
        FILE_ALIGNMENT,
        6, 0,         # Major/MinorOperatingSystemVersion
        0, 0,         # Major/MinorImageVersion
        6, 0,         # Major/MinorSubsystemVersion
        0,            # Win32VersionValue
        size_of_image,
        size_of_headers,
        0,            # CheckSum (not required for user-mode exes)
        _SUBSYSTEM_CONSOLE,
        0,            # DllCharacteristics: no ASLR (image has no relocs)
        0x100000,     # SizeOfStackReserve
        0x1000,       # SizeOfStackCommit
        0x100000,     # SizeOfHeapReserve
        0x1000,       # SizeOfHeapCommit
        0,            # LoaderFlags
        _N_DIRS,      # NumberOfRvaAndSizes
    )
    dirs = bytearray(_N_DIRS * 8)
    _II.pack_into(dirs, _IMPORT_DIR * 8, text_rva + desc_off,
                  _DESC * (len(groups) + 1))
    iat_size = sum(_THUNK * (len(funcs) + 1) for funcs in groups.values())
    _II.pack_into(dirs, _IAT_DIR * 8,
                  text_rva + iat_off[next(iter(groups))], iat_size)
    sechdr = _SECHDR.pack(
        b".text\0\0\0",
        vsize,
        text_rva,
        rawsize,
        size_of_headers,
        0, 0,        # PointerToRelocations, PointerToLinenumbers
        0, 0,        # NumberOfRelocations, NumberOfLinenumbers
        _SECTION_CODE,
    )

    header = bytes(dos) + b"PE\x00\x00" + coff + opt + bytes(dirs) + sechdr
    body = bytes(text) + bytes(blob)  # len == stored_len <= rawsize
    return header.ljust(size_of_headers, b"\x00") + body.ljust(rawsize, b"\x00")
