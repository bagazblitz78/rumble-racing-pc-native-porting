"""Build standalone Windows x64 executables that run a compiled guest
program natively and exit with a guest-computed register value.

This is the milestone M-B acceptance evidence: real PE files whose .text
section contains x86-64 machine code translated from EE instructions, with
no emulator anywhere in the execution path.

Image layout (all runtime data in bss, zero-filled by the loader):
  bss+0x000: ctx struct (regs, hi/lo, ram, fpr, fcr31, acc) - 688 bytes
  bss+0x2B0: guest RAM                                     - 32 MiB

The entry stub (in .text, before the guest code):
  sub  rsp, 0x28                 ; Win64 shadow + 16-byte alignment
  movabs rcx, ctx_va
  movabs rax, ram_va
  mov  [rcx + 0x220], rax        ; ctx.ram_ptr = guest RAM
  call guest_code                ; -> int run(void* ctx), returns in eax
  add  rsp, 0x28                 ; restore entry alignment for the tail jump
  movabs rcx, ctx_va
  mov  ecx, [rcx + 0x20]         ; v0 = guest-computed exit code
  jmp  [rip + IAT ExitProcess]
"""
from __future__ import annotations

import struct
from pathlib import Path

from .codegen import Program
from .pe import Layout, build_console_exe

RAM_SIZE = 32 * 1024 * 1024
_STUB_LEN = 59
_V0_SLOT = 0x20    # ctx.regs[2] (low 32 bits of a 16-byte slot)
_RAM_PTR_SLOT = 544


def _stub(layout: Layout, entry_offset: int = 0) -> bytes:
    ctx_va = layout.image_base + layout.bss_rva
    ram_va = ctx_va + 0x2B0
    iat_va_rva = layout.iat_rva[("KERNEL32.DLL", "ExitProcess")]
    jmp_disp = iat_va_rva - (layout.text_rva + _STUB_LEN)
    call_disp = _STUB_LEN + entry_offset - 36  # call target = guest entry
    code = bytearray()
    code += b"\x48\x83\xEC\x28"                          # sub rsp, 0x28
    code += b"\x48\xB9" + struct.pack("<Q", ctx_va)      # movabs rcx, ctx
    code += b"\x48\xB8" + struct.pack("<Q", ram_va)      # movabs rax, ram
    code += b"\x48\x89\x81" + struct.pack("<I", _RAM_PTR_SLOT)  # [rcx+0x110]
    code += b"\xE8" + struct.pack("<i", call_disp)       # call guest
    code += b"\x48\x83\xC4\x28"                          # add rsp, 0x28
    code += b"\x48\xB9" + struct.pack("<Q", ctx_va)      # movabs rcx, ctx
    code += b"\x8B\x49" + struct.pack("<B", _V0_SLOT)    # mov ecx, [rcx+0x10]
    code += b"\xFF\x25" + struct.pack("<i", jmp_disp)    # jmp [rip+ExitProcess]
    assert len(code) == _STUB_LEN, len(code)
    return bytes(code)


def build_exe(program: Program, out_path: Path) -> Path:
    """Write a standalone console EXE that runs `program` and exits with v0."""
    def build_text(layout: Layout) -> bytes:
        return _stub(layout, program.entry_offset) + program.code

    data = build_console_exe(
        build_text,
        bss_size=0x2B0 + RAM_SIZE,
        imports=[("KERNEL32.DLL", "ExitProcess")],
        entry_text_offset=0,
    )
    out_path = Path(out_path)
    out_path.write_bytes(data)
    return out_path
