"""R5900 MMI semantics (T3.5), cross-checked against PCSX2's interpreter."""
import pytest

from ps2toolkit.golden.interpreter import EEMemory, EEState, step
from ps2toolkit.r5900.decoder import decode
from tests import asm as a

CODE = 0x00100000
M128 = (1 << 128) - 1


def mmi(funct: int, sa: int, rd: int, rs: int, rt: int) -> int:
    """Build an MMI word: opcode 0x1C, group in funct, operation in sa."""
    return a.enc_r(0x1C, rs, rt, rd, sa, funct)


def run1(word: int, **regs):
    st = EEState(pc=CODE)
    for name, value in regs.items():
        st.regs[getattr(a, name.upper())] = value
    mem = EEMemory()
    a.assemble(mem, CODE, word, a.NOP)
    step(st, mem)
    return st


def lane(reg: int, bits: int, index: int) -> int:
    return (reg >> (bits * index)) & ((1 << bits) - 1)


def test_paddub_is_the_compiler_copy_idiom():
    value = 0x0102030405060708090A0B0C0D0E0F10
    st = run1(mmi(0x28, 24, a.T2, a.T0, a.ZERO), t0=value)
    assert st.regs[a.T2] == value


def test_paddub_saturates_per_byte():
    st = run1(mmi(0x28, 24, a.T2, a.T0, a.T1),
              t0=0xF0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0,
              t1=0x20202020202020202020202020202020)
    assert st.regs[a.T2] == 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


def test_psubub_saturates_at_zero():
    st = run1(mmi(0x28, 25, a.T2, a.T0, a.T1),
              t0=0x10, t1=0x20)
    assert st.regs[a.T2] == 0


def test_paddw_psubw_and_paddsw():
    st = run1(mmi(0x08, 0, a.T2, a.T0, a.T1),
              t0=0x7FFFFFFF00000001, t1=0x0000000200000003)
    assert lane(st.regs[a.T2], 32, 0) == 4
    assert lane(st.regs[a.T2], 32, 1) == 0x80000001
    st = run1(mmi(0x08, 0, a.T2, a.T0, a.ZERO), t0=1)
    assert st.regs[a.T2] == 1

    st = run1(mmi(0x08, 16, a.T2, a.T0, a.T1),  # PADDSW
              t0=0x7FFFFFFF7FFFFFFF, t1=0x0000000100000001)
    assert st.regs[a.T2] == 0x7FFFFFFF7FFFFFFF

    st = run1(mmi(0x08, 16, a.T2, a.T0, a.T1),  # negative saturation
              t0=0x8000000080000000, t1=0xFFFFFFFFFFFFFFFF)
    assert st.regs[a.T2] == 0x8000000080000000


def test_compare_min_max():
    st = run1(mmi(0x08, 2, a.T2, a.T0, a.T1),  # PCGTW
              t0=0x00000002FFFFFFFF, t1=0x00000001FFFFFFFF)
    assert lane(st.regs[a.T2], 32, 0) == 0
    assert lane(st.regs[a.T2], 32, 1) == 0xFFFFFFFF

    st = run1(mmi(0x28, 3, a.T2, a.T0, a.T1),  # PMINW
              t0=0x00000002FFFFFFFF, t1=0x00000001FFFFFFFE)
    assert st.regs[a.T2] == 0x00000001FFFFFFFE

    st = run1(mmi(0x28, 2, a.T2, a.T0, a.T1),  # PCEQW
              t0=0x1111111122222222, t1=0x1111111133333333)
    assert st.regs[a.T2] == 0xFFFFFFFFFFFFFFFFFFFFFFFF00000000  # all 4 lanes


def test_pextlw_pextuw_pcpyld_pcpyud():
    # t0 lanes: [1, 2, 3, 4]; t1 lanes: [5, 6, 7, 8]
    t0 = 0x00000004000000030000000200000001
    t1 = 0x00000008000000070000000600000005

    st = run1(mmi(0x08, 18, a.T2, a.T0, a.T1), t0=t0, t1=t1)  # PEXTLW
    assert st.regs[a.T2] == 0x00000002000000060000000100000005

    st = run1(mmi(0x28, 18, a.T2, a.T0, a.T1), t0=t0, t1=t1)  # PEXTUW
    assert st.regs[a.T2] == 0x00000004000000080000000300000007

    st = run1(mmi(0x09, 14, a.T2, a.T0, a.T1), t0=t0, t1=t1)  # PCPYLD
    assert st.regs[a.T2] == 0x00000002000000010000000600000005

    st = run1(mmi(0x29, 14, a.T2, a.T0, a.T1), t0=t0, t1=t1)  # PCPYUD
    assert st.regs[a.T2] == 0x00000008000000070000000400000003


def test_pcpyh():
    st = run1(mmi(0x29, 27, a.T2, a.ZERO, a.T1),  # PCPYH
              t1=0x00070006000500040003000200010000)
    assert st.regs[a.T2] == 0x00040004000400040000000000000000


def test_pand_por_pxor_pnor():
    a_val = 0xF0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0
    b_val = 0x0FF00FF00FF00FF00FF00FF00FF00FF0
    assert run1(mmi(0x09, 18, a.T2, a.T0, a.T1), t0=a_val, t1=b_val).regs[a.T2] == (a_val & b_val)
    assert run1(mmi(0x29, 18, a.T2, a.T0, a.T1), t0=a_val, t1=b_val).regs[a.T2] == (a_val | b_val)
    assert run1(mmi(0x09, 19, a.T2, a.T0, a.T1), t0=a_val, t1=b_val).regs[a.T2] == (a_val ^ b_val)
    assert run1(mmi(0x29, 19, a.T2, a.T0, a.T1), t0=a_val, t1=b_val).regs[a.T2] == (~(a_val | b_val) & M128)


def test_mult1_div1_and_mflo1_mfhi1():
    st = run1(mmi(0x18, 0, a.T2, a.T0, a.T1),  # MULT1
              t0=0xFFFFFFFE, t1=3)
    assert (st.lo >> 64) & 0xFFFFFFFFFFFFFFFF == 0xFFFFFFFFFFFFFFFA
    assert (st.hi >> 64) & 0xFFFFFFFFFFFFFFFF == 0xFFFFFFFFFFFFFFFF
    assert st.regs[a.T2] & 0xFFFFFFFFFFFFFFFF == 0xFFFFFFFFFFFFFFFA

    st = run1(mmi(0x18, 0, a.T2, a.T0, a.T1), t0=7, t1=5)
    st.pc = CODE
    mem = EEMemory()
    a.assemble(mem, CODE, mmi(0x12, 0, a.T3, a.ZERO, a.ZERO), a.NOP)  # MFLO1
    step(st, mem)
    assert st.regs[a.T3] == 35

    st = run1(mmi(0x1A, 0, a.T2, a.T0, a.T1), t0=0xFFFFFFFE, t1=3)  # DIV1
    assert (st.lo >> 64) & 0xFFFFFFFFFFFFFFFF == 0  # -2 / 3 truncates to 0
    assert (st.hi >> 64) & 0xFFFFFFFFFFFFFFFF == 0xFFFFFFFFFFFFFFFE


def test_shift_and_plzcw():
    st = run1(mmi(0x3C, 4, a.T2, a.ZERO, a.T1),  # PSLLW by 4
              t1=0x8000000100000002)
    assert st.regs[a.T2] == 0x0000001000000020

    st = run1(mmi(0x28, 1, a.T2, a.ZERO, a.T1),  # PABSW (MMI1 index 1)
              t1=0x0000000080000000)
    assert st.regs[a.T2] == 0x000000007FFFFFFF

    st = run1(mmi(0x04, 0, a.T2, a.T0, a.ZERO),  # PLZCW
              t0=0x0000FFFF00000000)
    assert lane(st.regs[a.T2], 32, 0) == 31
    assert lane(st.regs[a.T2], 32, 1) == 15


def test_mmi_decode_fields():
    insn = decode(mmi(0x28, 24, 17, 4, 0), 0x121630)
    assert insn.mnemonic == "paddub"
    assert (insn.rs, insn.rt, insn.rd, insn.shamt) == (4, 0, 17, 24)
    assert "paddub" in insn.disasm()
