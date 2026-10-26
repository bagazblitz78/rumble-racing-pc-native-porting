"""Execute compiled EE programs natively on Windows x64 and compare with the
golden interpreter (correctness oracle).

Native side:
  - code bytes are copied into a PAGE_EXECUTE_READWRITE allocation
    (VirtualAlloc) and called as ``int run(void* ctx)`` (Win64: rcx = ctx).
  - ctx layout matches codegen: regs[i] = i*16 (128-bit, low then high u64),
    hi = 512, lo = 528 (128-bit each), ram_ptr = 544.
  - the guest RAM base (r14) points directly at the live bytearray buffer of
    an ``EEMemory``, so no 32MB copies are needed per run.
  - return value: -1 (0xFFFFFFFF) on normal completion, else the guest
    syscall exit code.

Golden side: the interpreter runs until pc reaches the stop address (or the
call sentinel), or a syscall is raised.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes

from ..golden.interpreter import (
    EEMemory,
    EEState,
    GuestSyscall,
    step,
)
from .codegen import EXIT_NORMAL, Program

RAM_SIZE = 32 * 1024 * 1024
SENTINEL = 0x00300000  # pc marker meaning "returned from main"

_KERNEL32 = ctypes.windll.kernel32
_KERNEL32.VirtualAlloc.restype = ctypes.c_void_p
_KERNEL32.VirtualAlloc.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD
]
_KERNEL32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
_KERNEL32.GetCurrentProcess.restype = ctypes.c_void_p
_KERNEL32.FlushInstructionCache.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD
]

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_READWRITE = 0x04
_PAGE_EXECUTE_READWRITE = 0x40


class Ctx(ctypes.Structure):
    # GPRs are 128-bit: two u64 slots per register (low, high).
    _fields_ = [
        ("regs", ctypes.c_uint64 * 64),
        ("hi", ctypes.c_uint64 * 2),
        ("lo", ctypes.c_uint64 * 2),
        ("ram_ptr", ctypes.c_void_p),
    ]


def run_native(program: Program, guest_mem: EEMemory, initial: EEState):
    """Run compiled code; returns (regs32, hi, lo, exit_code)."""
    code = program.code
    addr = _KERNEL32.VirtualAlloc(
        None, len(code), _MEM_COMMIT | _MEM_RESERVE, _PAGE_EXECUTE_READWRITE
    )
    if not addr:
        raise OSError(f"VirtualAlloc failed for {len(code)} bytes")
    try:
        ctypes.memmove(addr, code, len(code))
        _KERNEL32.FlushInstructionCache(
            _KERNEL32.GetCurrentProcess(), addr, len(code), 0)
        runner = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(addr)

        view = (ctypes.c_char * RAM_SIZE).from_buffer(guest_mem.data)
        ctx = Ctx()
        for i in range(32):
            ctx.regs[2 * i] = initial.regs[i] & 0xFFFFFFFFFFFFFFFF
            ctx.regs[2 * i + 1] = (initial.regs[i] >> 64) & 0xFFFFFFFFFFFFFFFF
        ctx.hi[0] = initial.hi & 0xFFFFFFFFFFFFFFFF
        ctx.hi[1] = (initial.hi >> 64) & 0xFFFFFFFFFFFFFFFF
        ctx.lo[0] = initial.lo & 0xFFFFFFFFFFFFFFFF
        ctx.lo[1] = (initial.lo >> 64) & 0xFFFFFFFFFFFFFFFF
        ctx.ram_ptr = ctypes.addressof(view)

        ret = runner(ctypes.addressof(ctx))
        regs = [ctx.regs[2 * i] | (ctx.regs[2 * i + 1] << 64)
                for i in range(32)]
        hi = ctx.hi[0] | (ctx.hi[1] << 64)
        lo = ctx.lo[0] | (ctx.lo[1] << 64)
        return regs, hi, lo, ret & 0xFFFFFFFF
    finally:
        _KERNEL32.VirtualFree(addr, 0, _MEM_RELEASE)


def run_golden(guest_mem: EEMemory, initial: EEState, entry: int, stop: int,
               max_steps: int = 200_000):
    """Run the golden interpreter; returns (regs32, hi, lo, exit_code)."""
    st = EEState(pc=entry)
    st.regs[:] = initial.regs[:]
    st.hi, st.lo = initial.hi, initial.lo
    steps = 0
    code = EXIT_NORMAL
    while steps < max_steps:
        if st.pc in (SENTINEL, stop):
            break
        try:
            steps += step(st, guest_mem)
        except GuestSyscall as exc:
            code = exc.code
            break
    else:
        raise AssertionError(
            f"golden interpreter hit max_steps={max_steps} (loop did not terminate?)"
        )
    return st.regs[:], st.hi, st.lo, code


def _fmt_diff(kind: str, expected, got) -> str:
    if kind in ("regs",):
        diffs = [
            f"r{i}: golden={a:#034x} native={b:#034x}"
            for i, (a, b) in enumerate(zip(expected, got)) if a != b
        ]
        return "; ".join(diffs)
    return f"golden={expected:#x} native={got:#x}"


def compare_results(golden, native):
    """Raise AssertionError describing any state divergence."""
    g_regs, g_hi, g_lo, g_code = golden
    n_regs, n_hi, n_lo, n_code = native

    if g_regs != n_regs:
        raise AssertionError(f"regs diverge: {_fmt_diff('regs', g_regs, n_regs)}")
    if g_hi != n_hi:
        raise AssertionError(f"hi diverges: {_fmt_diff('int', g_hi, n_hi)}")
    if g_lo != n_lo:
        raise AssertionError(f"lo diverges: {_fmt_diff('int', g_lo, n_lo)}")
    if g_code != n_code:
        raise AssertionError(
            f"exit code diverges: {_fmt_diff('int', g_code, n_code)}"
        )
