"""Shared helpers for differential tests: golden interpreter vs native x64."""
from __future__ import annotations

import random

from ps2toolkit.golden.interpreter import EEState, EEMemory
from ps2toolkit.r5900.decoder import decode
from ps2toolkit.x64emit.codegen import compile_program
from ps2toolkit.x64emit.harness import compare_results, run_golden, run_native
from tests import asm as a

CODE = 0x00100000
STACK = 0x00110000

# Registers the generator may freely clobber: exclude $zero (always 0 by
# construction), $sp (memory base) and $ra (return sentinel).
CLOBBERABLE = [i for i in range(31) if i not in (0, 29, 31)]


def make_state(seed: int | None = None) -> EEState:
    st = EEState(pc=CODE)
    if seed is not None:
        rnd = random.Random(seed)
        for i in range(31):
            st.regs[i] = rnd.getrandbits(64)
        st.hi = rnd.getrandbits(32)
        st.lo = rnd.getrandbits(32)
    st.regs[0] = 0  # $zero is hardwired; codegen treats rs==0 as literal 0
    st.regs[29] = STACK
    st.regs[31] = 0x00300000  # sentinel: pc here == main returned
    return st


def run_case(words, seed: int | None = None, state: EEState | None = None,
             max_steps: int = 200_000, label: str = ""):
    """Assemble, run golden + native, assert identical architectural state."""
    mem = EEMemory()
    a.assemble(mem, CODE, *words)
    stop = CODE + 4 * len(words)
    initial = state if state is not None else make_state(seed)

    insn_at = lambda addr: decode(mem.read32(addr), addr)  # noqa: E731
    # golden must not leak its memory writes into the native run (and vice
    # versa): snapshot guest RAM around each execution
    snapshot = bytes(mem.data)
    golden = run_golden(mem, initial, CODE, stop, max_steps)
    mem.data[:] = snapshot
    program = compile_program(insn_at, CODE, stop_addrs={stop})
    native = run_native(program, mem, initial)
    try:
        compare_results(golden, native)
    except AssertionError as exc:
        detail = f"{label}: {exc}" if label else str(exc)
        detail += f"\n  program: {[insn_at(CODE + 4*i).mnemonic for i in range(len(words))]}"
        raise AssertionError(detail) from None
    return golden, native, program
