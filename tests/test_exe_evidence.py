"""Standalone-EXE evidence (milestone M-B).

Compiles EE programs to native x64, wraps each in a real Windows PE
(virtualized 32MB bss guest RAM, KERNEL32!ExitProcess via IAT), writes the
.exe to disk and lets Windows execute it. The process exit code is the
guest-computed $v0 register — proof that translated EE instructions ran as
native x86-64 machine code with no emulator involved.
"""
import subprocess
from pathlib import Path

import pytest

from ps2toolkit.golden.interpreter import EEMemory
from ps2toolkit.r5900.decoder import decode
from ps2toolkit.x64emit.codegen import compile_program
from ps2toolkit.x64emit.evidence import build_exe
from tests import asm as a
from tests.difftest import CODE, STACK


def _compile_exe(words, path: Path) -> Path:
    mem = EEMemory()
    a.assemble(mem, CODE, *words)
    stop = CODE + 4 * len(words)
    insn_at = lambda addr: decode(mem.fetch32(addr), addr)  # noqa: E731
    program = compile_program(insn_at, CODE, stop_addrs={stop})
    return build_exe(program, path)


def _run(words, tmp_path: Path, name: str) -> int:
    exe = _compile_exe(words, tmp_path / name)
    size = exe.stat().st_size
    assert size < 64 * 1024, f"{name} unexpectedly large ({size} bytes)"
    proc = subprocess.run([str(exe)], capture_output=True, timeout=60)
    assert proc.returncode >= 0, f"{name} crashed (rc={proc.returncode:#x})"
    return proc.returncode


def test_exe_mult_produces_42(tmp_path):
    rc = _run([
        a.addiu(a.T0, a.ZERO, 6),
        a.addiu(a.T1, a.ZERO, 7),
        a.multu(a.T0, a.T1),
        a.mflo(a.V0),
        a.jr(a.RA),
        a.NOP,
    ], tmp_path, "mult42.exe")
    assert rc == 42


def test_exe_loop_sums_55(tmp_path):
    rc = _run([
        a.addiu(a.T0, a.ZERO, 0),        # sum
        a.addiu(a.T1, a.ZERO, 1),        # i
        a.addiu(a.T2, a.ZERO, 11),       # limit
        a.addu(a.T0, a.T0, a.T1),        # body
        a.addiu(a.T1, a.T1, 1),
        a.bne(a.T1, a.T2, -3),           # loop
        a.addiu(a.T3, a.ZERO, 0),        # delay
        a.addu(a.V0, a.T0, a.ZERO),
        a.jr(a.RA),
        a.NOP,
    ], tmp_path, "loop55.exe")
    assert rc == 55


def test_exe_memory_roundtrip(tmp_path):
    rc = _run([
        a.addiu(a.T0, a.ZERO, 0x2A),
        a.sw(a.T0, 0, a.SP),
        a.lw(a.V0, 0, a.SP),
        a.jr(a.RA),
        a.NOP,
    ], tmp_path, "mem42.exe")
    assert rc == 42


def test_exe_jal_function_call(tmp_path):
    rc = _run([
        a.jal(CODE + 0x30),
        a.NOP,                           # delay
        a.jr(a.RA),
        a.NOP,
        a.NOP, a.NOP, a.NOP, a.NOP,      # pad 0x10..0x1c
        a.NOP, a.NOP, a.NOP, a.NOP,      # pad 0x20..0x2c
        a.addiu(a.V0, a.ZERO, 99),       # @0x30 callee
        a.jr(a.RA),
        a.NOP,
    ], tmp_path, "call99.exe")
    assert rc == 99


def test_exe_mmi_and_quadword_memory(tmp_path):
    # SQ/LQ move a full 128-bit GPR, PADDUB copies it (rt == 0), and the
    # low byte of v0 becomes the process exit code.
    rc = _run([
        a.addiu(a.T0, a.ZERO, 0x2A),
        a.sq(a.T0, 0, a.SP),
        a.lq(a.T1, 0, a.SP),
        a.paddub(a.V0, a.T1, a.ZERO),
        a.jr(a.RA),
        a.NOP,
    ], tmp_path, "mmi42.exe")
    assert rc == 42


def test_exe_entry_not_at_code_start(tmp_path):
    # A callee placed below main is emitted first, so the guest entry sits at
    # a non-zero offset inside Program.code; the stub must call that address.
    callee = [
        a.addiu(a.V0, a.ZERO, 7),
        a.jr(a.RA),
        a.NOP,
    ] + [a.NOP] * 5                    # pad 0x0c..0x1c
    main = [
        a.jal(CODE),                     # @0x20
        a.NOP,                           # delay
        a.jr(a.RA),
        a.NOP,
    ]
    words = callee + main
    mem = EEMemory()
    a.assemble(mem, CODE, *words)
    stop = CODE + 4 * len(words)
    insn_at = lambda addr: decode(mem.fetch32(addr), addr)  # noqa: E731
    program = compile_program(insn_at, CODE + 0x20, stop_addrs={stop})
    assert program.entry_offset == program.function_offsets[CODE + 0x20] > 0
    exe = build_exe(program, tmp_path / "entry20.exe")
    proc = subprocess.run([str(exe)], capture_output=True, timeout=60)
    assert proc.returncode == 7


def test_exe_branch_and_unaligned_memory(tmp_path):
    # exercise a branch plus the unaligned pair idiom, result in v0
    rc = _run([
        a.lui(a.T0, 0xDEAD),
        a.ori(a.T0, a.T0, 0xBEEF),
        a.sw(a.T0, 0, a.SP),
        a.addiu(a.T1, a.ZERO, 0x1234),
        a.lwl(a.T1, 1, a.SP),            # merge bytes 1..3
        a.lwr(a.T1, 4, a.SP),            # merge byte from next word
        a.beq(a.T1, a.T1, 2),            # always taken
        a.addiu(a.V0, a.T1, 0),          # delay: capture result
        a.addiu(a.V0, a.ZERO, 0xEE),     # skipped
        a.jr(a.RA),
        a.NOP,
    ], tmp_path, "unaligned.exe")
    # lwl r1,[SP+1] on word DEADBEEF: r1 = (BE bytes 1..3 << 8) | 0x1234 & 0xff
    #   bytes 1..3 = AD BE EF -> 0xADBEEF00 | 0x34 = 0xADBEEF34
    # lwr r1,[SP+4] (next word is zero): keeps high 3 bytes, byte 0 = 0x00
    #   -> 0xADBE0000 -> low byte0 = 0x00, so 0xADBE0034? compute exactly:
    # after lwl: r1 = 0xADBEEF34; lwr n=1 (offset 4 -> word-aligned, byte 0):
    #   r1 = (r1 & 0xffffff00) | 0x00 = 0xADBEEF00
    assert (rc & 0xFFFFFFFF) == 0xDEADBE00
