"""Differential tests for the SSE2 MMI/LQ/SQ lowering (T3.7).

Every case runs the same program through the golden interpreter and the
native x64 recompiler and requires the full 128-bit register file, HI/LO
and memory writes to match exactly.
"""
import random

import pytest

from tests import asm as a
from tests.difftest import make_state, run_case

M128 = (1 << 128) - 1


def mmi(funct: int, sa: int, rd: int, rs: int, rt: int) -> int:
    return a.enc_r(0x1C, rs, rt, rd, sa, funct)


# (name, funct, sa)
MMI_CASES = [
    ("paddb", 0x08, 8), ("psubb", 0x08, 9),
    ("paddh", 0x08, 4), ("psubh", 0x08, 5),
    ("paddw", 0x08, 0), ("psubw", 0x08, 1),
    ("paddsb", 0x08, 24), ("psubsb", 0x08, 25),
    ("paddsh", 0x08, 20), ("psubsh", 0x08, 21),
    ("paddub", 0x28, 24), ("psubub", 0x28, 25),
    ("padduh", 0x28, 20), ("psubuh", 0x28, 21),
    ("pcgtb", 0x08, 10), ("pcgth", 0x08, 6), ("pcgtw", 0x08, 2),
    ("pceqb", 0x28, 10), ("pceqh", 0x28, 6), ("pceqw", 0x28, 2),
    ("pmaxh", 0x08, 7), ("pminh", 0x28, 7),
    ("pextlw", 0x08, 18), ("pextuw", 0x28, 18),
    ("pextlh", 0x08, 22), ("pextuh", 0x28, 22),
    ("pextlb", 0x08, 26), ("pextub", 0x28, 26),
    ("pcpyld", 0x09, 14), ("pcpyud", 0x29, 14),
    ("pcpyh", 0x29, 27),
    ("pexeh", 0x09, 26), ("prevh", 0x09, 27),
    ("pexew", 0x09, 30), ("prot3w", 0x09, 31),
    ("pexch", 0x29, 26), ("pexcw", 0x29, 30),
    ("pand", 0x09, 18), ("por", 0x29, 18),
    ("pxor", 0x09, 19), ("pnor", 0x29, 19),
]


def _edge_value(rnd: random.Random) -> int:
    """Random 128-bit value biased toward saturation/byte edges."""
    if rnd.random() < 0.5:
        return rnd.getrandbits(128)
    byte = rnd.choice((0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF))
    return int.from_bytes(bytes([byte] * 16), "little")


@pytest.mark.parametrize("name,funct,sa", MMI_CASES,
                         ids=[c[0] for c in MMI_CASES])
def test_mmi_matches_golden(name, funct, sa):
    for seed in range(12):
        rnd = random.Random(seed * 131 + funct + sa)
        st = make_state()
        st.regs[a.T0] = _edge_value(rnd)
        st.regs[a.T1] = _edge_value(rnd)
        st.regs[a.T2] = rnd.getrandbits(128)
        run_case([
            mmi(funct, sa, a.T2, a.T0, a.T1),
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"{name} seed={seed}")


def test_lq_sq_full_quadword_matches_golden():
    for seed in range(20):
        rnd = random.Random(seed)
        st = make_state()
        st.regs[a.T0] = rnd.getrandbits(128)
        run_case([
            a.sq(a.T0, 0, a.SP),
            a.lq(a.T1, 0, a.SP),
            a.sq(a.T1, 8, a.SP),
            a.lq(a.T2, 8, a.SP),
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"lq/sq seed={seed}")


def test_plzcw_matches_golden():
    for seed in range(10):
        rnd = random.Random(seed)
        st = make_state()
        st.regs[a.T0] = rnd.getrandbits(128)
        st.regs[a.T1] = rnd.choice((0, M128, 1 << 63, (1 << 64) - 1))
        run_case([
            mmi(0x04, 0, a.T2, a.T0, a.ZERO),
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"plzcw seed={seed}")


def test_mult1_hi_lo_matches_golden():
    for seed in range(20):
        rnd = random.Random(seed)
        st = make_state()
        st.regs[a.T0] = rnd.getrandbits(32)
        st.regs[a.T1] = rnd.getrandbits(32)
        run_case([
            mmi(0x18, 0, a.T2, a.T0, a.T1),   # MULT1
            mmi(0x10, 0, a.T3, a.ZERO, a.ZERO),  # MFHI1
            mmi(0x12, 0, a.T4, a.ZERO, a.ZERO),  # MFLO1
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"mult1 seed={seed}")


def test_mthi1_mtlo1_matches_golden():
    for seed in range(10):
        rnd = random.Random(seed)
        st = make_state()
        st.regs[a.T0] = rnd.getrandbits(64)
        st.regs[a.T1] = rnd.getrandbits(64)
        run_case([
            mmi(0x11, 0, 0, a.T0, a.ZERO),    # MTHI1
            mmi(0x13, 0, 0, a.T1, a.ZERO),    # MTLO1
            mmi(0x10, 0, a.T2, a.ZERO, a.ZERO),  # MFHI1
            mmi(0x12, 0, a.T3, a.ZERO, a.ZERO),  # MFLO1
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"mthi1 seed={seed}")
