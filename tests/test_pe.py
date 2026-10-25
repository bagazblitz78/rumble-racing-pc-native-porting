"""Tests for the minimal Windows x64 PE executable writer (task T2.6).

The payloads are hand-assembled x64 bytes (no dependency on other project
modules) and the built executables are actually executed on Windows to
verify the import table, entry point, and bss handling end to end.
"""
import struct
import subprocess

import pytest

from ps2toolkit.x64emit.pe import Layout, build_console_exe

DLL = "KERNEL32.DLL"
FUNC = "ExitProcess"
IMPORTS = [(DLL, FUNC)]


# --------------------------------------------------------------------------
# helpers: hand-assembled payloads and an independent PE reader
# --------------------------------------------------------------------------

def exit_stub(layout: Layout, code: int) -> bytes:
    """`mov ecx, code; jmp qword ptr [rip+rel32]` through ExitProcess' IAT slot.

    Win64 passes the first argument in ecx; ExitProcess' exit code becomes
    the process exit code. The jmp is 6 bytes, so rip = entry + 5 + 6.
    """
    iat = layout.iat_rva[(DLL, FUNC)]
    head = b"\xb9" + struct.pack("<I", code)
    disp = iat - (layout.entry_rva + len(head) + 6)
    return head + b"\xff\x25" + struct.pack("<i", disp)


def bss_probe_stub(layout: Layout) -> bytes:
    """Prove bss starts zero and is writable: exit 55 on success, 99 on fault.

    Reads dword [bss+0x1000] (must be 0), writes 0x77, reads it back
    (must equal 0x77), then exits with 55; any mismatch exits with 99.
    """
    bss = layout.image_base + layout.bss_rva + 0x1000
    iat = layout.iat_rva[(DLL, FUNC)]
    body = bytearray()
    body += b"\x48\xb8" + struct.pack("<Q", bss)  # mov rax, imm64
    body += b"\x8b\x08"                            # mov ecx, dword ptr [rax]
    body += b"\x85\xc9"                            # test ecx, ecx
    jnz1 = len(body)
    body += b"\x0f\x85\x00\x00\x00\x00"            # jnz bad
    body += b"\xc7\x00\x77\x00\x00\x00"            # mov dword ptr [rax], 0x77
    body += b"\x8b\x08"                            # mov ecx, dword ptr [rax]
    body += b"\x83\xf9\x77"                        # cmp ecx, 0x77
    jnz2 = len(body)
    body += b"\x0f\x85\x00\x00\x00\x00"            # jnz bad
    body += b"\xb9" + struct.pack("<I", 55)        # mov ecx, 55
    jmp1 = len(body)
    body += b"\xff\x25\x00\x00\x00\x00"            # jmp [rip+IAT]
    bad = len(body)
    body += b"\xb9" + struct.pack("<I", 99)        # mov ecx, 99
    jmp2 = len(body)
    body += b"\xff\x25\x00\x00\x00\x00"            # jmp [rip+IAT]
    entry = layout.entry_rva
    struct.pack_into("<i", body, jnz1 + 2, bad - (jnz1 + 6))
    struct.pack_into("<i", body, jnz2 + 2, bad - (jnz2 + 6))
    struct.pack_into("<i", body, jmp1 + 2, iat - (entry + jmp1 + 6))
    struct.pack_into("<i", body, jmp2 + 2, iat - (entry + jmp2 + 6))
    return bytes(body)


def build(payload, bss_size=0, **kwargs):
    """build_console_exe() while capturing the Layout handed to build_text()."""
    seen = {}

    def build_text(layout):
        seen["layout"] = layout
        return payload(layout) if callable(payload) else payload

    pe = build_console_exe(build_text, bss_size, **kwargs)
    return pe, seen["layout"]


def parse_pe(pe: bytes) -> dict:
    """Independent struct-based reader for the fields these tests check."""
    assert pe[0:2] == b"MZ", "missing DOS magic"
    lfanew = struct.unpack_from("<I", pe, 0x3C)[0]
    assert pe[lfanew:lfanew + 4] == b"PE\x00\x00", "missing PE signature"
    machine, nsections, _, _, _, optsize, chars = struct.unpack_from(
        "<HHIIIHH", pe, lfanew + 4)
    opt = lfanew + 24
    magic = struct.unpack_from("<H", pe, opt)[0]
    entry = struct.unpack_from("<I", pe, opt + 16)[0]
    image_base = struct.unpack_from("<Q", pe, opt + 24)[0]
    secalign, filealign = struct.unpack_from("<II", pe, opt + 32)
    size_image, size_headers = struct.unpack_from("<II", pe, opt + 56)
    subsystem = struct.unpack_from("<H", pe, opt + 68)[0]
    ndirs = struct.unpack_from("<I", pe, opt + 108)[0]
    dirs = [struct.unpack_from("<II", pe, opt + 112 + 8 * i) for i in range(ndirs)]
    sh = opt + optsize
    name = pe[sh:sh + 8].rstrip(b"\x00").decode("ascii")
    vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", pe, sh + 8)
    seccs = struct.unpack_from("<I", pe, sh + 36)[0]
    return {
        "lfanew": lfanew, "machine": machine, "nsections": nsections,
        "optsize": optsize, "chars": chars, "magic": magic, "entry": entry,
        "image_base": image_base, "secalign": secalign, "filealign": filealign,
        "size_image": size_image, "size_headers": size_headers,
        "subsystem": subsystem, "ndirs": ndirs, "dirs": dirs,
        "name": name, "vsize": vsize, "vaddr": vaddr, "rawsize": rawsize,
        "rawptr": rawptr, "seccs": seccs,
    }


def rva_to_off(pe: bytes, rva: int, sec: dict) -> int:
    """File offset of an RVA inside the stored part of .text."""
    assert sec["vaddr"] <= rva < sec["vaddr"] + sec["rawsize"], \
        f"RVA {rva:#x} not in stored section data"
    return sec["rawptr"] + (rva - sec["vaddr"])


def cstr(data: bytes, off: int) -> str:
    return data[off:data.index(b"\x00", off)].decode("ascii")


def walk_imports(pe: bytes, sec: dict):
    """Walk the import directory per the PE spec.

    Returns (imports, iat_slots): imports maps dll -> [(hint, func)],
    iat_slots maps (dll, func) -> RVA of the 8-byte IAT slot the loader
    overwrites with the resolved function address.
    """
    lfanew = struct.unpack_from("<I", pe, 0x3C)[0]
    opt = lfanew + 24
    imp_rva, imp_size = struct.unpack_from("<II", pe, opt + 112 + 8)
    assert imp_rva != 0 and imp_size >= 20, "import directory empty"
    imports, slots = {}, {}
    off = rva_to_off(pe, imp_rva, sec)
    while True:
        oft, _ts, _fc, name_rva, ft = struct.unpack_from("<IIIII", pe, off)
        if oft == 0 and name_rva == 0 and ft == 0:
            break
        dll = cstr(pe, rva_to_off(pe, name_rva, sec))
        funcs = []
        i = 0
        while True:
            name_va = struct.unpack_from(
                "<Q", pe, rva_to_off(pe, oft + 8 * i, sec))[0]
            if name_va == 0:
                break
            assert name_va >> 63 == 0, "ordinal import not expected"
            hint_off = rva_to_off(pe, name_va, sec)
            hint = struct.unpack_from("<H", pe, hint_off)[0]
            func = cstr(pe, hint_off + 2)
            funcs.append((hint, func))
            slots[(dll, func)] = ft + 8 * i
            i += 1
        imports[dll] = funcs
        off += 20
    return imports, slots


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_parses_structurally():
    """Headers, section, and import directory match the PE32+ spec.

    Catches wrong machine/subsystem/entry point, missing data directories,
    misaligned headers, and a bss tail that SizeOfImage fails to cover.
    """
    bss_size = 0x2000
    pe, layout = build(lambda l: exit_stub(l, 42), bss_size=bss_size)
    sec = parse_pe(pe)

    assert sec["lfanew"] == 0x80
    assert sec["machine"] == 0x8664
    assert sec["nsections"] == 1
    assert sec["chars"] == 0x22  # EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
    assert sec["magic"] == 0x20B  # PE32+
    assert sec["image_base"] == 0x140000000
    assert sec["secalign"] == 0x1000
    assert sec["filealign"] == 0x200
    assert sec["subsystem"] == 3  # console
    assert sec["ndirs"] == 16
    assert sec["dirs"][1] != (0, 0), "import directory not populated"
    assert sec["dirs"][12] != (0, 0), "IAT directory not populated"
    assert all(d == (0, 0) for i, d in enumerate(sec["dirs"])
               if i not in (1, 12)), "unexpected data directory"

    # headers must fit before the section data, aligned to FileAlignment
    assert sec["size_headers"] == 0x200
    assert sec["size_headers"] % sec["filealign"] == 0
    assert sec["rawptr"] == sec["size_headers"]
    assert len(pe) == sec["rawptr"] + sec["rawsize"]

    # entry point at the start of .text, encoding hand-checked
    assert layout.entry_rva == sec["vaddr"] == 0x1000
    eoff = rva_to_off(pe, sec["entry"], sec)
    assert pe[eoff:eoff + 5] == b"\xb9\x2a\x00\x00\x00"  # mov ecx, 42
    assert pe[eoff + 5:eoff + 7] == b"\xff\x25"          # jmp [rip+rel32]

    # section: writable code section; stored size covers payload + imports
    assert sec["name"] == ".text"
    assert sec["seccs"] == 0xE0000060  # CODE|EXECUTE|READ|WRITE
    assert sec["rawsize"] % sec["filealign"] == 0
    assert sec["vsize"] >= sec["rawsize"] - (sec["filealign"] - 1)
    assert sec["vsize"] >= len(exit_stub(layout, 42)) + bss_size

    # SizeOfImage covers .text plus the zero-filled bss tail
    assert sec["size_image"] % sec["secalign"] == 0
    assert sec["size_image"] >= sec["vaddr"] + sec["vsize"]
    assert sec["size_image"] > sec["vaddr"] + bss_size

    # import table reads back per spec, and the stub jumps at the reported slot
    imports, slots = walk_imports(pe, sec)
    assert imports == {DLL: [(0, FUNC)]}
    slot = slots[(DLL, FUNC)]
    assert layout.iat_rva[(DLL, FUNC)] == slot
    disp = struct.unpack_from("<i", pe, eoff + 7)[0]
    assert sec["entry"] + 11 + disp == slot


def test_runs_and_exits_with_code(tmp_path):
    """The built image loads, resolves ExitProcess, and exits with 42.

    Catches any defect that stops Windows from mapping or transferring
    control to the image, or from resolving the IAT slot.
    """
    pe, _ = build(lambda l: exit_stub(l, 42))
    exe = tmp_path / "stub42.exe"
    exe.write_bytes(pe)
    result = subprocess.run([str(exe)], timeout=30)
    assert result.returncode == 42


def test_bss_is_zero_filled_and_usable(tmp_path):
    """bss starts zero-filled, lies inside the mapping, and is writable.

    Exits 55 only when the first read of bss+0x1000 yields 0 and a
    written 0x77 reads back; 99 (or a crash) fails the test.
    """
    pe, layout = build(bss_probe_stub, bss_size=0x2000)
    assert layout.bss_size == 0x2000
    # the bss tail must not be stored in the file
    sec = parse_pe(pe)
    assert sec["rawsize"] < 0x2000
    assert sec["vsize"] >= (layout.bss_rva - layout.text_rva) + 0x2000
    exe = tmp_path / "bss.exe"
    exe.write_bytes(pe)
    result = subprocess.run([str(exe)], timeout=30)
    assert result.returncode == 55


def test_large_bss_small_file():
    """A 32MB bss is zero-filled at load, not stored in the file."""
    bss_size = 33554432
    pe, layout = build(b"\x90" * 8, bss_size=bss_size)
    sec = parse_pe(pe)
    assert len(pe) < 100_000
    assert layout.bss_size == bss_size
    assert sec["vsize"] >= bss_size + 8
    assert sec["rawsize"] <= 0x1000, "bss tail leaked into section data"
    assert sec["size_image"] > 33_000_000  # > 32MiB plus .text
    assert sec["size_image"] >= sec["vaddr"] + sec["vsize"]


def test_iat_rva_mapping():
    """layout.iat_rva points at the loader-writable IAT slot itself.

    The slot's file bytes must be the hint/name RVA of ExitProcess (the
    loader replaces them with the real address), not e.g. the ILT entry.
    """
    pe, layout = build(lambda l: exit_stub(l, 42))
    sec = parse_pe(pe)
    slot = layout.iat_rva[(DLL, FUNC)]
    name_rva = struct.unpack_from("<Q", pe, rva_to_off(pe, slot, sec))[0]
    assert name_rva != 0
    hint_off = rva_to_off(pe, name_rva, sec)
    assert pe[hint_off:hint_off + 2] == b"\x00\x00"  # hint 0
    assert cstr(pe, hint_off + 2) == FUNC
    # FirstThunk (the IAT the loader rewrites) must hold this slot RVA
    _, slots = walk_imports(pe, sec)
    assert slots[(DLL, FUNC)] == slot


def test_entry_text_offset_selects_entry():
    """AddressOfEntryPoint honours entry_text_offset within the payload."""
    pe, layout = build(lambda l: exit_stub(l, 7), entry_text_offset=5)
    sec = parse_pe(pe)
    assert layout.entry_rva == sec["entry"] == sec["vaddr"] + 5
    eoff = rva_to_off(pe, sec["entry"], sec)
    assert pe[eoff:eoff + 2] == b"\xff\x25"  # start of `jmp [rip+IAT]`


def test_rejects_bad_arguments():
    """Invalid offsets and empty import lists fail loudly."""
    with pytest.raises(ValueError):
        build_console_exe(lambda l: b"\x90", 0, imports=[])
    with pytest.raises(ValueError):
        build_console_exe(lambda l: b"\x90", 0, entry_text_offset=1)
    with pytest.raises(ValueError):
        build_console_exe(lambda l: b"\x90", -1)
    with pytest.raises(ValueError):
        # payload length must not depend on the layout
        build_console_exe(lambda l: b"\x90" * (1 if l.iat_rva[(DLL, FUNC)] else 2), 0)
